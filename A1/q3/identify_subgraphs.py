#!/usr/bin/env python3

import sys
import os
import subprocess
import pickle
import math
from collections import defaultdict, Counter


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
        seen_edges = set()
        unique_edges = []
        for u, v, lbl in g["edges"]:
            key = (min(u, v), max(u, v), lbl)
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append((u, v, lbl))
        cleaned.append({"nodes": g["nodes"], "edges": unique_edges})
    return cleaned



# Gaston interaction

def write_gaston_format(graphs, outp):
    with open(outp, "w") as f:
        for gid, g in enumerate(graphs):
            f.write(f"t # {gid}\n")
            for nid, lbl in g["nodes"]:
                f.write(f"v {nid} {lbl}\n")
            for u, v, lbl in g["edges"]:
                f.write(f"e {u} {v} {lbl}\n")


def run_gaston(binpath, infile, minsup, outfile):
    """Run gaston.  minsup is an absolute count (int)."""
    try:
        subprocess.run(
            [binpath, str(minsup), infile, outfile],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600          # 10 min per support level
        )
        return os.path.exists(outfile) and os.path.getsize(outfile) > 0
    except Exception as exc:
        print(f"    gaston exception: {exc}")
        return False


def parse_gaston_output(path):
    """Parse gaston's output file into a list of subgraph dicts."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    subgraphs = []
    sg = {"nodes": [], "edges": [], "support": 0}

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            if line.startswith("t"):
                if sg["nodes"] or sg["edges"]:
                    subgraphs.append(sg)
                sg = {"nodes": [], "edges": [], "support": 0}
                # gaston puts support count at end of 't' line
                tokens = line.split()
                for tok in reversed(tokens):
                    if tok.isdigit():
                        sg["support"] = int(tok)
                        break
                continue

            if line.startswith("v"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        sg["nodes"].append((int(parts[1]), int(parts[2])))
                    except ValueError:
                        pass
                continue

            if line.startswith(("e", "u", "d")):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        sg["edges"].append(
                            (int(parts[1]), int(parts[2]), int(parts[3]))
                        )
                    except ValueError:
                        pass

    if sg["nodes"] or sg["edges"]:
        subgraphs.append(sg)

    return subgraphs


# Scoring  –  entropy-based discriminativeness

def entropy(p):
    """Binary entropy H(p). Safe at boundaries."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def discriminative_score(sg, total_graphs):
    freq = sg["support"] / total_graphs if total_graphs > 0 else 0.0
    h = entropy(freq)                       

    # Mild size bonus for moderate-complexity fragments
    n_edges = len(sg["edges"])
    if n_edges == 0:
        size_w = 0.1
    elif n_edges == 1:
        size_w = 0.6
    elif 2 <= n_edges <= 5:
        size_w = 1.0 + 0.05 * n_edges          
    else:
        size_w = 1.0 / (1.0 + 0.1 * (n_edges - 5))  

    return h * size_w



# Deduplication & diversity

def canonical_key(sg):
    """Deterministic key for deduplication across support levels."""
    return (tuple(sorted(sg["nodes"])), tuple(sorted(sg["edges"])))


def structural_fingerprint(sg):
    """Compact fingerprint for diversity bucketing."""
    return (
        tuple(sorted(Counter(n[1] for n in sg["nodes"]).items())),
        tuple(sorted(Counter(e[2] for e in sg["edges"]).items())),
        len(sg["nodes"]),
        len(sg["edges"]),)


def greedy_diverse_selection(scored_sgs, k):
    buckets = defaultdict(list)          
    for sg in scored_sgs:
        buckets[len(sg["edges"])].append(sg)

    bucket_iters = {ec: iter(lst) for ec, lst in sorted(buckets.items())}
    bucket_keys  = sorted(buckets.keys())

    selected = []
    seen_fps = set()
    exhausted = set()

    while len(selected) < k and len(exhausted) < len(bucket_keys):
        for ec in bucket_keys:
            if ec in exhausted:
                continue
            while True:
                try:
                    sg = next(bucket_iters[ec])
                except StopIteration:
                    exhausted.add(ec)
                    break
                fp = structural_fingerprint(sg)
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    selected.append(sg)
                    break                
            if len(selected) >= k:
                break

    
    if len(selected) < k:
        sel_ids = {id(s) for s in selected}
        for sg in scored_sgs:
            if id(sg) not in sel_ids:
                selected.append(sg)
                sel_ids.add(id(sg))
            if len(selected) == k:
                break

    return selected[:k]



# Main

def main():
    if len(sys.argv) != 4:
        print("Usage: python identify_subgraphs.py <gaston_bin> <db_graphs> <out.pkl>")
        sys.exit(1)

    gaston_bin, db_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # ---- load & clean -------------------------------------------------
    graphs = parse_graph_dataset(db_path)
    print(f"Loaded {len(graphs)} graphs")

    graphs = remove_duplicate_edges(graphs)
    print(f"After duplicate-edge removal: {len(graphs)} graphs")

    total = len(graphs)

    # write gaston input 
    tmp_input  = "/tmp/_gaston_input.dat"
    write_gaston_format(graphs, tmp_input)

    
    if total < 5000:
        support_pcts = [70]
    else:
        support_pcts = [35]

    print(f"Support levels (%): {support_pcts}")

    all_subgraphs = []
    for pct in support_pcts:
        minsup   = max(1, total * pct // 100)
        out_file = f"/tmp/_gaston_out_{pct}"
        print(f"  Gaston @ {pct}%  (minsup={minsup}) …", end=" ", flush=True)

        if not run_gaston(gaston_bin, tmp_input, minsup, out_file):
            print("FAILED - skipping")
            continue

        batch = parse_gaston_output(out_file)
        print(f"{len(batch)} subgraphs mined")
        all_subgraphs.extend(batch)

    print(f"Total mined across all levels: {len(all_subgraphs)}")

    if not all_subgraphs:
        print("ERROR: No subgraphs mined at any support level!")
        sys.exit(1)

    
    all_subgraphs = [sg for sg in all_subgraphs if len(sg["edges"]) >= 1]
    print(f"After edge filter:{len(all_subgraphs)}")

    all_subgraphs = [sg for sg in all_subgraphs
                     if len(sg["edges"]) <= 8 and len(sg["nodes"]) <= 10]
    print(f"After size cap filter:{len(all_subgraphs)}")

    # Drop near-universal subgraphs 
    all_subgraphs = [sg for sg in all_subgraphs
                     if sg["support"] < 0.90 * total]
    print(f"After frequency cap filter:    {len(all_subgraphs)}")

    canon = {}
    for sg in all_subgraphs:
        key = canonical_key(sg)
        if key not in canon:
            canon[key] = sg
        else:
            old_dist = abs(canon[key]["support"] / total - 0.5)
            new_dist = abs(sg["support"]         / total - 0.5)
            if new_dist < old_dist:
                canon[key] = sg

    all_subgraphs = list(canon.values())
    print(f"After cross-level dedup:       {len(all_subgraphs)}")

    # ---- score & sort --------------------------------------------------
    for sg in all_subgraphs:
        sg["disc_score"] = discriminative_score(sg, total)

    all_subgraphs.sort(key=lambda x: x["disc_score"], reverse=True)

    # ---- select exactly 50 with diversity ------------------------------
    K = 50
    final = greedy_diverse_selection(all_subgraphs, K)

    with open(out_path, "wb") as f:
        pickle.dump(final, f)

    print(f"\nSaved {len(final)} discriminative subgraphs → {out_path}")
    print(f"Score range:  {final[-1]['disc_score']:.4f} … {final[0]['disc_score']:.4f}")
    print(f"Edge-count distribution: {Counter(len(sg['edges']) for sg in final)}")


if __name__ == "__main__":
    main()
