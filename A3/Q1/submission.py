from __future__ import annotations

import time
import numpy as np


def solve(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int, K: int, time_budget: float) -> np.ndarray:
    """
    base_vectors: (N, d) float32
    query_vectors: (Q, d) float32
    k: number of nearest neighbors per query
    K: number of top representative base indices to return
    time_budget: seconds

    Strategy:
    1. Build a FAISS index on base_vectors (IVF with HNSW coarse quantizer for speed+recall balance).
    2. For each query, retrieve k approximate nearest neighbors.
    3. Accumulate hit counts across all queries.
    4. Sort by (-count, index) to get the top-K most representative items, breaking ties by lower index.
    """
    import faiss

    t_start = time.perf_counter()

    N, d = base_vectors.shape
    Q = query_vectors.shape[0]

    # Ensure contiguous float32
    base = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Index selection heuristic:
    #   - Small N (< 20k):  Flat (exact) — always fast enough
    #   - Medium N:         IVF + Flat with sensible nlist
    #   - Large N:          IVF + PQ or HNSW-backed IVF for speed
    # ------------------------------------------------------------------ #

    time_remaining = time_budget - (time.perf_counter() - t_start)

    def _build_flat():
        idx = faiss.IndexFlatL2(d)
        idx.add(base)
        return idx, None

    def _build_ivf(nlist, use_hnsw_coarse=False):
        if use_hnsw_coarse:
            quantizer = faiss.IndexHNSWFlat(d, 32)
            quantizer.hnsw.efConstruction = 80
        else:
            quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        idx.train(base)
        idx.add(base)
        return idx, nlist

    def _build_ivfpq(nlist, M, nbits=8):
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
        idx.train(base)
        idx.add(base)
        return idx, nlist

    # Choose index type based on N
    if N < 20_000:
        index, nlist = _build_flat()
        nprobe = None
    elif N < 200_000:
        nlist = min(int(4 * np.sqrt(N)), N // 10, 4096)
        nlist = max(nlist, 64)
        index, nlist = _build_ivf(nlist, use_hnsw_coarse=False)
        nprobe = min(nlist, max(16, nlist // 8))
        index.nprobe = nprobe
    else:
        nlist = min(int(4 * np.sqrt(N)), 8192)
        nlist = max(nlist, 256)

        # PQ: M must divide d
        M_candidates = [m for m in [64, 48, 32, 16, 8] if d % m == 0]
        if not M_candidates:
            # fallback: find largest divisor of d ≤ 64
            M_candidates = [m for m in range(min(64, d), 0, -1) if d % m == 0]
        M = M_candidates[0] if M_candidates else min(d, 8)

        # If d is small or M is hard to find, just use IVFFlat
        build_pq = (d >= 32) and (M >= 8) and (N > 500_000)

        if build_pq:
            try:
                index, nlist = _build_ivfpq(nlist, M, nbits=8)
                nprobe = min(nlist, max(32, nlist // 4))
                index.nprobe = nprobe
            except Exception:
                index, nlist = _build_ivf(nlist, use_hnsw_coarse=True)
                nprobe = min(nlist, max(32, nlist // 4))
                index.nprobe = nprobe
        else:
            index, nlist = _build_ivf(nlist, use_hnsw_coarse=(N > 300_000))
            nprobe = min(nlist, max(32, nlist // 4))
            index.nprobe = nprobe

    build_time = time.perf_counter() - t_start

    # ------------------------------------------------------------------ #
    # Adaptive nprobe: if we have lots of remaining time, increase recall
    # ------------------------------------------------------------------ #
    time_for_search = time_budget - build_time - 0.5 
    if nlist is not None and time_for_search > 0:
        # Estimate time per query at current nprobe via a small timing probe
        probe_q = min(Q, max(1, min(50, Q // 20)))
        t0 = time.perf_counter()
        _ = index.search(query[:probe_q], k)
        probe_elapsed = time.perf_counter() - t0
        if probe_elapsed > 0:
            time_per_query = probe_elapsed / probe_q
            affordable_nprobe = int(nprobe * (time_for_search / max(time_per_query * Q, 1e-9)))
            affordable_nprobe = min(nlist, max(nprobe, affordable_nprobe))
            index.nprobe = affordable_nprobe

    # ------------------------------------------------------------------ #
    # Search: batch all queries at once (FAISS is most efficient this way)
    # ------------------------------------------------------------------ #
    _, I = index.search(query, k)  # I: (Q, k)

    # ------------------------------------------------------------------ #
    # Count representativeness
    # ------------------------------------------------------------------ #
    counts = np.zeros(N, dtype=np.int64)
    valid_mask = I >= 0  # FAISS may return -1 for padding
    valid_indices = I[valid_mask].ravel()
    np.add.at(counts, valid_indices, 1)

    order = np.arange(N, dtype=np.int64)
    order = order[np.argsort(-counts[order], kind="stable")]

    top_K = order[:K]

    # Pad with unused indices if somehow K > N 
    if top_K.shape[0] < K:
        used = set(top_K.tolist())
        extras = [i for i in range(N) if i not in used]
        top_K = np.concatenate([top_K, np.array(extras[:K - top_K.shape[0]], dtype=np.int64)])

    return top_K.astype(np.int64)