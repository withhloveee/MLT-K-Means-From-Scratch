import pandas as pd
import numpy as np

from graph import plot_clusters
from kmeans.clustering import compute_means,assign_clusters
from kmeans.initialization import initialize_labels

df = pd.read_csv("two_clusters_150_rows.csv")
# X is {data matrix}
X = df.to_numpy()

# assume, we already know the K value.
# step1: initialize the z(0)
# step2: find mean using the z(0)
# step3: assign the new z using the new mean call it z(1)
# repeat until the z(t) = z(t+1)

K = 2
z = initialize_labels(X,K)

iteration = 0
while True:
    iteration += 1
    
    means = compute_means(X, z, K)
    new_z = assign_clusters(X, means)
    
    plot_clusters(X, z, means)
    
    print(f"\nIteration {iteration}")
    print("Changes:", np.sum(z != new_z))

    if np.array_equal(z, new_z):
        break
    else:
        z = new_z