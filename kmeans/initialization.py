import numpy as np

def initialize_labels(X, K):
    sample_size = X.shape[0]
    return np.random.randint(0, K, size=sample_size)