import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import json
import sys


def load_dataset_api(student_id="aib253002", dataset_num=1):

    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id={student_id}&dataset_num={dataset_num}"

    try:
        with urllib.request.urlopen(url) as response:
            raw_data = response.read().decode("utf-8")
            data = json.loads(raw_data)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    X = np.array(data["X"])
    return X



def kmeans(X, k, max_iter=100):

    n, d = X.shape

    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):

        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = []

        for i in range(k):
            cluster_points = X[labels == i]

            if len(cluster_points) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(cluster_points.mean(axis=0))

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    obj = 0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            obj += np.sum((cluster_points - centroids[i]) ** 2)

    return obj



def compute_objectives(X):

    ks = list(range(1, 16))
    objectives = []

    for k in ks:

        best_obj = float("inf")

        for _ in range(10):
            obj = kmeans(X, k)
            best_obj = min(best_obj, obj)

        objectives.append(best_obj)

    return ks, objectives


def find_elbow(ks, objectives):

    x = np.array(ks, dtype=float)
    y = np.array(objectives, dtype=float)

    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec /= np.linalg.norm(line_vec)

    dists = []

    for i in range(len(x)):
        pt_vec = np.array([x[i] - x[0], y[i] - y[0]])
        proj = np.dot(pt_vec, line_vec) * line_vec
        dist = np.linalg.norm(pt_vec - proj)
        dists.append(dist)

    return ks[np.argmax(dists)]


def plot_curve(ks, objectives, optimal_k):

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    ax.fill_between(ks, objectives, alpha=0.08, color='#378ADD')
    ax.plot(ks, objectives, color='#378ADD', linewidth=2.5)

    ax.scatter(ks, objectives, color='#378ADD', s=55,
               edgecolors='white', linewidths=1.5)

    optimal_obj = objectives[optimal_k - 1]
    ax.scatter([optimal_k], [optimal_obj], color='#D85A30', s=90,
               edgecolors='white', linewidths=1.5)

    ax.axvline(x=optimal_k, color='#D85A30', linestyle='--', alpha=0.4)

    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('objective value')
    ax.set_title('K-means objective vs k')

    ax.set_xticks(ks)

    plt.tight_layout()
    plt.savefig("plot.png", dpi=150)
    plt.close()



def plot_two_datasets(results):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#FAFAFA')

    for idx, (ks, objectives, optimal_k, title) in enumerate(results):

        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')

        ax.fill_between(ks, objectives, alpha=0.08, color='#378ADD')
        ax.plot(ks, objectives, color='#378ADD', linewidth=2.5)

        ax.scatter(ks, objectives, color='#378ADD', s=55,
                   edgecolors='white', linewidths=1.5)

        optimal_obj = objectives[optimal_k - 1]
        ax.scatter([optimal_k], [optimal_obj], color='#D85A30', s=90,
                   edgecolors='white', linewidths=1.5)

        ax.axvline(x=optimal_k, color='#D85A30', linestyle='--', alpha=0.4)

        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel("objective")
        ax.set_xticks(ks)

    plt.tight_layout()
    plt.savefig("plot.png", dpi=150)
    plt.close()



def main():

    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("python3 Q1.py <dataset_num>")
        print("python3 Q1.py <path>.npy")
        sys.exit(1)

    arg = sys.argv[1]


    if arg.isdigit():

        results = []

        for dataset_num in [1, 2]:

            X = load_dataset_api(dataset_num=dataset_num)

            ks, objectives = compute_objectives(X)
            optimal_k = find_elbow(ks, objectives)

            results.append((ks, objectives, optimal_k, f"Dataset {dataset_num}"))

            print(f"Dataset {dataset_num} optimal k:", optimal_k)

        plot_two_datasets(results)


    elif arg.endswith(".npy"):

        X = np.load(arg)

        ks, objectives = compute_objectives(X)
        optimal_k = find_elbow(ks, objectives)

        plot_curve(ks, objectives, optimal_k)

        print(optimal_k)

    else:
        print(f"Unrecognised argument: {arg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()