import matplotlib.pyplot as plt

def plot_clusters(X, labels, centroids=None):
    """
    X          : numpy array of shape (n_samples, 2)
    labels     : cluster assignments
    centroids  : optional centroid array
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=50
    )

    if centroids is not None:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='X',
            s=200
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means Clustering")
    plt.grid(True)

    plt.show()