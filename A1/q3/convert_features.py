#!/usr/bin/env python3
"""
convert_features.py  -  graphs  ->  binary (N X 50) feature matrix.
Output: a single 2-D numpy int8 array saved to <path_features>.
"""

import sys
import pickle
import numpy as np
from collections import defaultdict, Counter
import time

# Parsing
def parse_graph_dataset(path):
    graphs = []
    cur = {"nodes": [], "edges": []}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if cur["nodes"] or cur["edges"]:
                    graphs.append(cur)
                cur = {"nodes": [], "edges": []}
            elif line.startswith("v"):
                _, nid, lbl = line.split()
                cur["nodes"].append((int(nid), int(lbl)))
            elif line.startswith("e"):
                _, u, v, lbl = line.split()
                cur["edges"].append((int(u), int(v), int(lbl)))
    if cur["nodes"] or cur["edges"]:
        graphs.append(cur)
    return graphs


def remove_duplicate_edges(graphs):
    
    cleaned = []
    for g in graphs:
        seen = set()
        edges = []
        for u, v, lbl in g["edges"]:
            key = (min(u, v), max(u, v), lbl)
            if key not in seen:
                seen.add(key)
                edges.append((u, v, lbl))
        cleaned.append({"nodes": g["nodes"], "edges": edges})
    return cleaned

def build_adj(graph):
    """
    Returns:
      node_label : {node_id: label}
      adj        : {node_id: {neighbour_id: edge_label}}
    """
    node_label = {n[0]: n[1] for n in graph["nodes"]}
    adj = defaultdict(dict)
    for u, v, lbl in graph["edges"]:
        adj[u][v] = lbl
        adj[v][u] = lbl
    return node_label, dict(adj)


#quick reject

def precompute_stats(graph):
    """Stats used by quick_reject.  Call once per graph."""
    node_labels  = Counter(n[1] for n in graph["nodes"])
    edge_labels  = Counter(e[2] for e in graph["edges"])

    # degree sequence (sorted ascending)
    adj = defaultdict(set)
    for u, v, _ in graph["edges"]:
        adj[u].add(v)
        adj[v].add(u)
    degrees = sorted(len(adj[n[0]]) for n in graph["nodes"])

    return {
        "n_nodes":     len(graph["nodes"]),
        "n_edges":     len(graph["edges"]),
        "node_labels": node_labels,
        "edge_labels": edge_labels,
        "degrees":     degrees,
    }


def quick_reject(p_stats, h_stats):
    """
    Return True if we can *definitely* say pattern ⊄ host.
    Return False if we must run VF2.
    All checks are necessary conditions - returning False is never a false negative.
    """
    if p_stats["n_nodes"] > h_stats["n_nodes"]:
        return True
    if p_stats["n_edges"] > h_stats["n_edges"]:
        return True

    # node-label multiset containment
    for lbl, cnt in p_stats["node_labels"].items():
        if h_stats["node_labels"].get(lbl, 0) < cnt:
            return True

    # edge-label multiset containment
    for lbl, cnt in p_stats["edge_labels"].items():
        if h_stats["edge_labels"].get(lbl, 0) < cnt:
            return True

    # degree-sequence feasibility: for every pattern degree d there must
    
    p_deg = p_stats["degrees"]   
    h_deg = h_stats["degrees"]   
    hi = 0
    for pd in p_deg:
        while hi < len(h_deg) and h_deg[hi] < pd:
            hi += 1
        if hi >= len(h_deg):
            return True
        hi += 1                  

    return False




def is_subgraph(pattern, host):
    
    p_nlabel, p_adj = build_adj(pattern)
    h_nlabel, h_adj = build_adj(host)

    p_nodes = list(p_nlabel.keys())
    h_nodes = list(h_nlabel.keys())

    if len(p_nodes) > len(h_nodes):
        return False

    
    p_deg = {n: len(p_adj.get(n, {})) for n in p_nodes}
    h_deg = {n: len(h_adj.get(n, {})) for n in h_nodes}

    
    p_nodes.sort(key=lambda n: p_deg[n], reverse=True)

    mapping = {}          
    used    = set()       

    def backtrack(idx):
        if idx == len(p_nodes):
            return True

        pn   = p_nodes[idx]
        p_lbl = p_nlabel[pn]
        p_d   = p_deg[pn]

        for hn in h_nodes:
            if hn in used:
                continue
            if h_nlabel[hn] != p_lbl:
                continue
            if h_deg[hn] < p_d:
                continue
            ok = True
            for prev_idx in range(idx):
                prev_pn = p_nodes[prev_idx]
                prev_hn = mapping[prev_pn]

                p_edge = p_adj.get(pn, {}).get(prev_pn)
                h_edge = h_adj.get(hn, {}).get(prev_hn)

                if p_edge is not None:
                    if h_edge != p_edge:
                        ok = False
                        break
            if not ok:
                continue

            mapping[pn] = hn
            used.add(hn)
            if backtrack(idx + 1):
                return True
            del mapping[pn]
            used.discard(hn)

        return False

    return backtrack(0)



# Main
def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_features.py <graphs.dat> <subgraphs.pkl> <out_features.npy>")
        sys.exit(1)

    graphs_path, subs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # ---- load ----------------------------------------------------------
    print("Loading graphs …", flush=True)
    graphs = remove_duplicate_edges(parse_graph_dataset(graphs_path))

    with open(subs_path, "rb") as f:
        subgraphs = pickle.load(f)

    N = len(graphs)
    K = len(subgraphs)
    print(f" Graphs: {N}   Subgraph features: {K}")

    # ---- precompute stats for quick rejection -------------------------
    print("Precomputing graph stats …", flush=True)
    g_stats = [precompute_stats(g) for g in graphs]
    p_stats = [precompute_stats(p) for p in subgraphs]

    # ---- feature extraction --------------------------------------------
    features = np.zeros((N, K), dtype=np.int8)

    rejected = 0
    vf2_calls = 0
    start = time.time()

    print("Extracting features …", flush=True)
    for j in range(K):
        ps = p_stats[j]
        pattern = subgraphs[j]

        for i in range(N):
            if quick_reject(ps, g_stats[i]):
                rejected += 1
                continue

            vf2_calls += 1
            if is_subgraph(pattern, graphs[i]):
                features[i, j] = 1

        # progress every 10 subgraphs
        if (j + 1) % 10 == 0 or j == K - 1:
            elapsed = time.time() - start
            pct     = (j + 1) / K * 100
            eta     = (elapsed / (j + 1)) * (K - j - 1)
            print(f"  [{j+1:>3}/{K}]  {pct:5.1f}%   "
                  f"elapsed={elapsed:6.1f}s   eta={eta:6.1f}s", flush=True)

    elapsed = time.time() - start
    total_checks = N * K
    print(f"\nDone in {elapsed:.1f}s")
    print(f" Quick-rejections: {rejected:,}/{total_checks:,}  "
          f"({rejected/total_checks*100:.1f}%)")
    print(f" VF2 calls:        {vf2_calls:,}")
    print(f" Matches:          {int(features.sum()):,}")

    # save
    np.save(out_path, features)
    print(f" Saved {features.shape}  :- {out_path}")


if __name__ == "__main__":
    main()
