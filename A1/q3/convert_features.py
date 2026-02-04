#!/usr/bin/env python3
"""
Optimized feature extraction with faster subgraph isomorphism checking.
"""

import sys
import pickle
import numpy as np
from collections import defaultdict, Counter
import time

# ============================================================================
# Graph Parsing
# ============================================================================

def parse_graph_dataset(path):
    """Parse graph dataset."""
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
    """Remove duplicate edges."""
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


# ============================================================================
# Graph Statistics for Quick Filtering
# ============================================================================

def precompute_stats(graph):
    """Compute statistics for quick rejection."""
    node_labels = Counter(n[1] for n in graph["nodes"])
    edge_labels = Counter(e[2] for e in graph["edges"])
    
    # Degree sequence
    adj = defaultdict(set)
    for u, v, _ in graph["edges"]:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = sorted(len(adj[n[0]]) for n in graph["nodes"])
    
    return {
        "n_nodes": len(graph["nodes"]),
        "n_edges": len(graph["edges"]),
        "node_labels": node_labels,
        "edge_labels": edge_labels,
        "degrees": degrees,
    }


def quick_reject(p_stats, h_stats):
    """
    Quick rejection test.
    Returns True if pattern definitely NOT in host.
    """
    # Size checks
    if p_stats["n_nodes"] > h_stats["n_nodes"]:
        return True
    if p_stats["n_edges"] > h_stats["n_edges"]:
        return True
    
    # Label containment
    for lbl, cnt in p_stats["node_labels"].items():
        if h_stats["node_labels"].get(lbl, 0) < cnt:
            return True
    
    for lbl, cnt in p_stats["edge_labels"].items():
        if h_stats["edge_labels"].get(lbl, 0) < cnt:
            return True
    
    # Degree sequence check
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


# ============================================================================
# Optimized Subgraph Isomorphism (VF2-like)
# ============================================================================

def build_adj(graph):
    """Build adjacency structure."""
    node_label = {n[0]: n[1] for n in graph["nodes"]}
    adj = defaultdict(dict)
    
    for u, v, lbl in graph["edges"]:
        adj[u][v] = lbl
        adj[v][u] = lbl
    
    return node_label, dict(adj)


def is_subgraph(pattern, host):
    """
    Optimized subgraph isomorphism test.
    Returns True if pattern is subgraph of host.
    """
    p_nlabel, p_adj = build_adj(pattern)
    h_nlabel, h_adj = build_adj(host)
    
    p_nodes = list(p_nlabel.keys())
    h_nodes = list(h_nlabel.keys())
    
    if len(p_nodes) > len(h_nodes):
        return False
    
    # Compute degrees
    p_deg = {n: len(p_adj.get(n, {})) for n in p_nodes}
    h_deg = {n: len(h_adj.get(n, {})) for n in h_nodes}
    
    # Sort pattern nodes by degree (descending) for better pruning
    p_nodes.sort(key=lambda n: p_deg[n], reverse=True)
    
    # Backtracking search
    mapping = {}
    used = set()
    
    def backtrack(idx):
        if idx == len(p_nodes):
            return True
        
        pn = p_nodes[idx]
        p_lbl = p_nlabel[pn]
        p_d = p_deg[pn]
        
        for hn in h_nodes:
            if hn in used:
                continue
            
            # Label and degree check
            if h_nlabel[hn] != p_lbl or h_deg[hn] < p_d:
                continue
            
            # Check consistency with existing mapping
            valid = True
            for prev_idx in range(idx):
                prev_pn = p_nodes[prev_idx]
                prev_hn = mapping[prev_pn]
                
                p_edge = p_adj.get(pn, {}).get(prev_pn)
                h_edge = h_adj.get(hn, {}).get(prev_hn)
                
                if p_edge is not None and h_edge != p_edge:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Try this mapping
            mapping[pn] = hn
            used.add(hn)
            
            if backtrack(idx + 1):
                return True
            
            del mapping[pn]
            used.discard(hn)
        
        return False
    
    return backtrack(0)


# ============================================================================
# Main Feature Extraction
# ============================================================================

def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_features.py <graphs.dat> <subgraphs.pkl> <out_features.npy>")
        sys.exit(1)
    
    graphs_path, subs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Load data
    print("Loading graphs...", flush=True)
    graphs = remove_duplicate_edges(parse_graph_dataset(graphs_path))
    
    with open(subs_path, "rb") as f:
        subgraphs = pickle.load(f)
    
    N = len(graphs)
    K = len(subgraphs)
    print(f"Graphs: {N}, Features: {K}")
    
    # Precompute statistics
    print("Precomputing graph statistics...", flush=True)
    g_stats = [precompute_stats(g) for g in graphs]
    p_stats = [precompute_stats(p) for p in subgraphs]
    
    # Extract features
    features = np.zeros((N, K), dtype=np.int8)
    
    rejected = 0
    vf2_calls = 0
    start = time.time()
    
    print("Extracting features...", flush=True)
    
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
        
        # Progress reporting
        if (j + 1) % 10 == 0 or j == K - 1:
            elapsed = time.time() - start
            pct = (j + 1) / K * 100
            eta = (elapsed / (j + 1)) * (K - j - 1)
            print(f"  [{j+1:>3}/{K}] {pct:5.1f}% | "
                  f"elapsed={elapsed:6.1f}s eta={eta:6.1f}s", flush=True)
    
    elapsed = time.time() - start
    total_checks = N * K
    
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Quick rejections: {rejected:,}/{total_checks:,} ({rejected/total_checks*100:.1f}%)")
    print(f"  VF2 calls: {vf2_calls:,}")
    print(f"  Matches found: {int(features.sum()):,}")
    
    # Save
    np.save(out_path, features)
    print(f"\nSaved feature matrix {features.shape} → {out_path}")


if __name__ == "__main__":
    main()