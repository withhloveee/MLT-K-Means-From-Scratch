import numpy as np

def compute_means(X, labels, K):
    means = []

    for k in range(K):
        cluster_points = X[labels == k]

        k_mean = np.mean(cluster_points, axis=0)

        means.append(k_mean)

    return np.array(means)

def assign_clusters(X, means):
    labels = []

    for point in X:
        distances = np.linalg.norm(means - point, axis=1)

        nearest_cluster = np.argmin(distances)

        labels.append(nearest_cluster)

    return np.array(labels)