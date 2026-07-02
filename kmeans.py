import pandas as pd
import numpy as np
from graph import plot_clusters

df = pd.read_csv("data.csv")
X = df.to_numpy()

sample_size = X.shape[0]
K = 2
z0 = np.random.randint(0,K,size=sample_size)

means = []

for k in range(K):
    cluster_points = X[z0 == k]
    
    k_mean = np.mean(cluster_points,axis=0)
    
    means.append(k_mean)
    
    print(f"\nCluster {k}")
    print(cluster_points)
    print("Mean:", k_mean)

means = np.array(means)
print("zo:",z0)

plot_clusters(X, z0, means)
    

# assume, we already know the K value.
# step1: initialize the z(0)
# step2: find mean using the z(0)
# step3: assign the new z using the new mean call it z(1)
# repeat until the z(t) = z(t+1)