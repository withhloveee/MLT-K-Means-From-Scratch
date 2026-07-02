import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
X = df.to_numpy()

print(X)

# assume, we already know the K value.
# step1: initialize the z(0)
# step2: find mean using the z(0)
# step3: assign the new z using the new mean call it z(1)
# repeat until the z(t) = z(t+1)