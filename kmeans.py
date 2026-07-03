import pandas as pd
import numpy as np
from graph import plot_clusters

df = pd.read_csv("data_1000.csv")
X = df.to_numpy()

sample_size = X.shape[0]
K = 2
z0 = np.random.randint(0,K,size=sample_size)

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

z = z0
iteration = 0

while True:
    iteration += 1
    
    print(f"\nIteration {iteration}")
    
    means = compute_means(X, z, K)
    plot_clusters(X, z, means)
    
    new_z = assign_clusters(X, means)
    
    # Checking convergence
    print("Changes:", np.sum(z != new_z))

    if np.array_equal(z, new_z):
        break

    z = new_z


# assume, we already know the K value.
# step1: initialize the z(0)
# step2: find mean using the z(0)
# step3: assign the new z using the new mean call it z(1)
# repeat until the z(t) = z(t+1)