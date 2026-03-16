#!/usr/bin/env python3
import sys
import random
import time
from collections import defaultdict, deque

RANDOM_SEED = 3000
random.seed(RANDOM_SEED)

TIME_LIMIT = 3600  # 1 hour in seconds
start_time  = time.time()


def load_graph(path):

    adj = defaultdict(list)
    nodes = set()

    with open(path) as f:
        for line in f:
            u, v, p = line.strip().split()

            u = int(u)
            v = int(v)
            p = float(p)

            adj[u].append((v, p))
            nodes.add(u)
            nodes.add(v)

    return adj, nodes



def load_seeds(path):

    seeds = []

    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s:
                    seeds.append(int(s))
    except:
        pass

    return seeds


def default_seed(nodes):

    if not nodes:
        return []

    return [min(nodes)]


def compute_h_hop(adj, seeds, h):

    if h == -1:
        return None

    visited = set(seeds)
    q = deque([(s, 0) for s in seeds])

    while q:

        u, d = q.popleft()

        if d == h:
            continue

        for v, _ in adj.get(u, []):

            if v not in visited:
                visited.add(v)
                q.append((v, d + 1))

    return visited


def restrict_graph(adj, allowed):

    if allowed is None:
        return adj

    new_adj = defaultdict(list)

    for u in adj:

        if u not in allowed:
            continue

        for v, p in adj[u]:
            if v in allowed:
                new_adj[u].append((v, p))

    return new_adj


def sample_graph(adj):

    g = defaultdict(list)

    for u in adj:

        for v, p in adj[u]:
            if random.random() <= p:
                g[u].append(v)

    return g


# ------------------------------------------------
# BFS TREE
# ------------------------------------------------
def bfs_tree(graph, seeds):

    parent = {}
    order = []

    q = deque()

    for s in seeds:
        parent[s] = None
        q.append(s)

    while q:

        u = q.popleft()
        order.append(u)

        for v in graph.get(u, []):

            if v not in parent:
                parent[v] = u
                q.append(v)

    return parent, order


def subtree_sizes(parent, order):

    children = defaultdict(list)

    for v, p in parent.items():
        if p is not None:
            children[p].append(v)

    size = {v: 1 for v in parent}

    for u in reversed(order):

        for c in children[u]:
            size[u] += size[c]

    return size


def estimate_reduction(adj, seeds, theta):

    reduction = defaultdict(float)

    for _ in range(theta):

        g = sample_graph(adj)

        parent, order = bfs_tree(g, seeds)

        if not parent:
            continue

        subtree = subtree_sizes(parent, order)

        for v in parent:

            u = parent[v]

            if u is None:
                continue

            reduction[(u, v)] += subtree[v]

    for e in reduction:
        reduction[e] /= theta

    return reduction



def write_output(path, edges):

    with open(path, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")


def advanced_greedy(adj, seeds, k, theta, output_path):

    blocked = []
    removed = set()

    for i in range(k):

        # ---- time check ----
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed

        if remaining <= 0:
            print(f"[TIMEOUT] Stopped after blocking {len(blocked)}/{k} edges "
                  f"({elapsed:.1f}s elapsed).", file=sys.stderr)
            break

        # Adaptive theta: if time is tight, reduce samples
        adaptive_theta = theta
        if remaining < 300:          # fewer than 5 min left
            adaptive_theta = max(1, theta // 5)
        elif remaining < 600:        # fewer than 10 min left
            adaptive_theta = max(1, theta // 2)

        reduction = estimate_reduction(adj, seeds, adaptive_theta)

        best_edge = None
        best_score = -1

        for (u, v), score in reduction.items():

            if (u, v) in removed:
                continue

            if score > best_score:
                best_score = score
                best_edge = (u, v)

        if best_edge is None:
            print(f"[INFO] No more edges to block after {len(blocked)} edges.",
                  file=sys.stderr)
            break

        blocked.append(best_edge)
        removed.add(best_edge)

        u, v = best_edge
        adj[u] = [(x, p) for x, p in adj[u] if x != v]

        # ---- partial save after every blocked edge ----
        write_output(output_path, blocked)

        elapsed = time.time() - start_time
        print(f"[{i+1}/{k}] Blocked ({u},{v})  score={best_score:.4f}  "
              f"elapsed={elapsed:.1f}s  saved to {output_path}",
              file=sys.stderr)

    return blocked


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():

    graph_path  = sys.argv[1]
    seed_path   = sys.argv[2]
    output_path = sys.argv[3]
    k           = int(sys.argv[4])
    n_samples   = int(sys.argv[5])
    hops        = int(sys.argv[6])

    adj, nodes = load_graph(graph_path)

    seeds = load_seeds(seed_path)

    if not seeds:
        seeds = default_seed(nodes)

    allowed = compute_h_hop(adj, seeds, hops)
    adj     = restrict_graph(adj, allowed)

    # Write empty output immediately so the file always exists
    write_output(output_path, [])

    blocked = advanced_greedy(adj, seeds, k, n_samples, output_path)

    # Final write (redundant but safe)
    write_output(output_path, blocked)

    total = time.time() - start_time
    print(f"[DONE] Blocked {len(blocked)} edges in {total:.1f}s.", file=sys.stderr)


if __name__ == "__main__":
    main()
