#!/usr/bin/env python3

import sys
import numpy as np
import time

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_candidates.py <db_feat.npy> <q_feat.npy> <out.dat>")
        sys.exit(1)

    db_feat = np.load(sys.argv[1])     
    q_feat  = np.load(sys.argv[2])      
    out_path = sys.argv[3]

    N_db, K  = db_feat.shape
    N_q      = q_feat.shape[0]

    print(f"DB graphs:    {N_db}")
    print(f"Query graphs: {N_q}")
    print(f"Features (k): {K}")

    start = time.time()
    CHUNK = max(1, (2_000_000_000 // (N_db * K)))   
    lines = []         

    for q_start in range(0, N_q, CHUNK):
        q_end   = min(q_start + CHUNK, N_q)
        q_chunk = q_feat[q_start:q_end]                          

        dom = np.all(q_chunk[:, None, :] <= db_feat[None, :, :], axis=2)   

        for local_i in range(q_end - q_start):
            qi = q_start + local_i
            candidates = np.where(dom[local_i])[0]   

            lines.append(f"q # {qi}")
            if len(candidates) > 0:
                lines.append("c # " + " ".join(map(str, candidates.tolist())))
            else:
                lines.append("c #")

        if (q_end % 50 == 0) or q_end == N_q:
            print(f"  Processed {q_end}/{N_q} queries ...", flush=True)

    # ---------------------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------------------
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    elapsed = time.time() - start

    # ---------------------------------------------------------------------------
    # Quick stats
    # ---------------------------------------------------------------------------
    candidate_counts = []
    for line in lines:
        if line.startswith("c #"):
            parts = line.split()
            candidate_counts.append(len(parts) - 2)  

    avg_c = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
    print(f"\n Done in {elapsed:.1f}s")
    print(f" Avg candidates per query: {avg_c:.1f}")
    print(f" Min: {min(candidate_counts)}   Max: {max(candidate_counts)}")
    print(f" Saved : {out_path}")


if __name__ == "__main__":
    main()
