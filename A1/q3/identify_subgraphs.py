#!/usr/bin/env python3

import sys
import os
import subprocess
import pickle
import math
from collections import defaultdict, Counter
import numpy as np

def parse_graph_dataset(path):
    """Parse graph dataset in standard format."""
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
    """Remove duplicate edges from graphs."""
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


def write_gaston_format(graphs, outp):
    """Write graphs in Gaston input format."""
    with open(outp, "w") as f:
        for gid, g in enumerate(graphs):
            f.write(f"t # {gid}\n")
            for nid, lbl in g["nodes"]:
                f.write(f"v {nid} {lbl}\n")
            for u, v, lbl in g["edges"]:
                f.write(f"e {u} {v} {lbl}\n")


def run_gaston(binpath, infile, minsup, outfile):
    """Run Gaston with given parameters."""
    try:
        subprocess.run(
            [binpath, str(minsup), infile, outfile],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600
        )
        return os.path.exists(outfile) and os.path.getsize(outfile) > 0
    except Exception as exc:
        print(f"    Gaston exception: {exc}")
        return False


def parse_gaston_output(path):
    """Parse Gaston output to extract subgraphs with support."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    
    subgraphs = []
    sg = None
    pending_support = None
    
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            
            if line.startswith("#"):
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    pending_support = int(parts[1])
                continue
            
            if line.startswith("t"):
                if sg is not None:
                    subgraphs.append(sg)
                sg = {"nodes": [], "edges": [], "support": pending_support}
                pending_support = None
                continue
            
            if line.startswith("v") and sg is not None:
                _, nid, lbl = line.split()
                sg["nodes"].append((int(nid), int(lbl)))
                continue
            
            if line.startswith(("e", "u", "d")) and sg is not None:
                parts = line.split()
                if len(parts) >= 4:
                    sg["edges"].append((int(parts[1]), int(parts[2]), int(parts[3])))
    
    if sg is not None:
        subgraphs.append(sg)
    
    return subgraphs


def entropy(p):
    """Binary entropy function."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def discriminative_score_large_dataset(sg, total_graphs):
    
    freq = sg["support"] / total_graphs
    base_entropy = entropy(freq)
    
    deviation = abs(freq - 0.5)
    if deviation < 0.1:  # Within 40-60 range
        balance_score = 1.0
    elif deviation < 0.15:  # Within 35-65 range
        balance_score = 0.8
    elif deviation < 0.25:  # Within 25-75 range
        balance_score = 0.5
    else:  # Outside 25-75%
        balance_score = 0.2  
    
    n_edges = len(sg["edges"])
    n_nodes = len(sg["nodes"])
    
    
    if 2 <= n_edges <= 4:
        size_score = 1.3 + 0.1 * n_edges  
    elif n_edges == 5:
        size_score = 1.2  
    elif n_edges == 1:
        size_score = 0.5  # Penalize single edges heavily
    elif 6 <= n_edges <= 7:
        size_score = 0.9  # Moderate penalty
    else:  # 8+ edges
        size_score = 0.3  # Very heavy penalty
    
    density = n_edges / max(n_nodes, 1)
    if 0.5 <= density <= 2.0:  
        complexity_score = 1.2
    else:
        complexity_score = 1.0
    
    if 0.40 <= freq <= 0.60:
        range_bonus = 1.5  
    elif 0.35 <= freq <= 0.65:
        range_bonus = 1.2  
    elif 0.30 <= freq <= 0.70:
        range_bonus = 1.0  
    elif 0.25 <= freq <= 0.75:
        range_bonus = 0.7  
    else:
        range_bonus = 0.3  
    
    final_score = base_entropy * balance_score * size_score * complexity_score * range_bonus
    
    return final_score


def canonical_key(sg):
    """Canonical key for deduplication."""
    return (tuple(sorted(sg["nodes"])), tuple(sorted(sg["edges"])))


def structural_fingerprint(sg):
    """Structural fingerprint for diversity."""
    node_label_dist = tuple(sorted(Counter(n[1] for n in sg["nodes"]).items()))
    edge_label_dist = tuple(sorted(Counter(e[2] for e in sg["edges"]).items()))
    return (node_label_dist, edge_label_dist, len(sg["nodes"]), len(sg["edges"]))


def greedy_diverse_selection_large(scored_sgs, k, total_graphs):
    """
    Selection optimized for large datasets.
    Emphasize mid-range support even more.
    """
    if len(scored_sgs) <= k:
        return scored_sgs
    
    # Tighter bins for large datasets
    support_bins = {
        'optimal': [],  
        'good': [],       
        'acceptable': []  
    }
    
    for sg in scored_sgs:
        freq = sg["support"] / total_graphs
        if 0.40 <= freq <= 0.60:
            support_bins['optimal'].append(sg)
        elif (0.30 <= freq < 0.40) or (0.60 < freq <= 0.70):
            support_bins['good'].append(sg)
        elif (0.25 <= freq < 0.30) or (0.70 < freq <= 0.75):
            support_bins['acceptable'].append(sg)
    
    # Quota allocation - HEAVILY favor optimal range
    quota = {
        'optimal': min(int(k * 0.75), len(support_bins['optimal'])), 
        'good': min(int(k * 0.20), len(support_bins['good'])),
        'acceptable': min(int(k * 0.05), len(support_bins['acceptable']))
    }
    
    # Adjust if optimal bin is empty
    total_quota = sum(quota.values())
    if total_quota < k:
        if len(support_bins['optimal']) > quota['optimal']:
            quota['optimal'] = min(k - total_quota + quota['optimal'], 
                                  len(support_bins['optimal']))
        elif len(support_bins['good']) > quota['good']:
            quota['good'] = min(k - total_quota + quota['good'],
                               len(support_bins['good']))
    
    # Select from each bin
    selected = []
    for bin_name in ['optimal', 'good', 'acceptable']:
        bin_features = support_bins[bin_name]
        bin_features.sort(key=lambda x: x["disc_score"], reverse=True)
        
        for sg in bin_features[:quota[bin_name]]:
            selected.append(sg)
    
    # If still need more, add highest scoring
    if len(selected) < k:
        all_selected_keys = {canonical_key(sg) for sg in selected}
        for sg in scored_sgs:
            if len(selected) >= k:
                break
            if canonical_key(sg) not in all_selected_keys:
                selected.append(sg)
                all_selected_keys.add(canonical_key(sg))
    
    selected.sort(key=lambda x: x["disc_score"], reverse=True)
    return selected[:k]


def main():
    if len(sys.argv) != 4:
        print("Usage: python identify_subgraphs.py <gaston_bin> <db_graphs> <out.pkl>")
        sys.exit(1)
    
    gaston_bin, db_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Load graphs
    print(f"Loading graphs from {db_path}...")
    graphs = parse_graph_dataset(db_path)
    total = len(graphs)
    print(f"Loaded {total:,} graphs")
    
    graphs = remove_duplicate_edges(graphs)
    print(f"After duplicate-edge removal: {len(graphs):,} graphs")
    total = len(graphs)
    
    # Write Gaston input
    tmp_input = "/tmp/_gaston_input.dat"
    write_gaston_format(graphs, tmp_input)
    
    # ADAPTIVE support thresholds based on dataset size
    if total < 5000:
        support_pcts = [34]
        print("Small dataset mode")
    elif total < 20000:
        support_pcts = [30]
        print("Medium dataset mode")
    else:
        support_pcts = [34]
        print("Large dataset mode (40K+ graphs)")
    
    print(f"Mining at support levels: {support_pcts}%")
    
    all_subgraphs = []
    for pct in support_pcts:
        minsup = max(1, total * pct // 100)
        out_file = f"/tmp/_gaston_out_{pct}"
        print(f"  Mining @ {pct}% (minsup={minsup:,})...", end=" ", flush=True)
        
        if not run_gaston(gaston_bin, tmp_input, minsup, out_file):
            print("FAILED")
            continue
        
        batch = parse_gaston_output(out_file)
        print(f"{len(batch)} patterns")
        all_subgraphs.extend(batch)
    
    print(f"\nTotal patterns mined: {len(all_subgraphs):,}")
    
    if not all_subgraphs:
        print("ERROR: No subgraphs mined!")
        sys.exit(1)
    
    print("\nFiltering patterns...")
    
    # Must have edges
    all_subgraphs = [sg for sg in all_subgraphs if len(sg["edges"]) >= 1]
    print(f"  After edge filter: {len(all_subgraphs):,}")
    
    all_subgraphs = [sg for sg in all_subgraphs 
                     if len(sg["edges"]) <= 7 and len(sg["nodes"]) <= 8]
    print(f"  After size filter: {len(all_subgraphs):,}")
    
    all_subgraphs = [sg for sg in all_subgraphs
                     if 0.25 * total <= sg["support"] <= 0.75 * total]
    print(f"  After support range filter (25-75%): {len(all_subgraphs):,}")
    
    # Deduplicate
    canon_map = {}
    for sg in all_subgraphs:
        key = canonical_key(sg)
        if key not in canon_map:
            canon_map[key] = sg
        else:
            # Keep one closest to 50%
            old_dist = abs(canon_map[key]["support"] / total - 0.5)
            new_dist = abs(sg["support"] / total - 0.5)
            if new_dist < old_dist:
                canon_map[key] = sg
    
    all_subgraphs = list(canon_map.values())
    print(f"  After deduplication: {len(all_subgraphs):,}")
    
    # Score all patterns
    print("\nScoring patterns...")
    for sg in all_subgraphs:
        sg["disc_score"] = discriminative_score_large_dataset(sg, total)
    
    all_subgraphs.sort(key=lambda x: x["disc_score"], reverse=True)
    
    # Select top-k with diversity
    K = 50
    print(f"\nSelecting top {K} features with diversity...")
    final = greedy_diverse_selection_large(all_subgraphs, K, total)
    
    # Save
    with open(out_path, "wb") as f:
        pickle.dump(final, f)
    
    print(f"\n{'='*70}")
    print(f"SAVED {len(final)} DISCRIMINATIVE SUBGRAPHS -> {out_path}")
    print(f"{'='*70}")
    
    if final:
        print(f"\nFeature Statistics:")
        print(f"  Score range: {final[-1]['disc_score']:.4f} to {final[0]['disc_score']:.4f}")
        
        edge_dist = Counter(len(sg['edges']) for sg in final)
        print(f"  Edge count distribution: {dict(sorted(edge_dist.items()))}")
        
        support_pcts = [sg["support"] / total * 100 for sg in final]
        print(f" Support range: {min(support_pcts):.1f}% to {max(support_pcts):.1f}%")
        print(f" Average support: {sum(support_pcts)/len(support_pcts):.1f}%")
        print(f" Median support: {sorted(support_pcts)[len(support_pcts)//2]:.1f}%")
        
        # Count features in optimal range (40-60%)
        optimal_count = sum(1 for p in support_pcts if 40 <= p <= 60)
        print(f"  Features in 40-60% range: {optimal_count}/{len(final)} ({optimal_count/len(final)*100:.1f}%)")


if __name__ == "__main__":
    main()