import numpy as np
import pandas as pd

n = 1000

x = np.random.randint(0, 1000, size=n)
y = np.random.randint(0, 1000, size=n)

df = pd.DataFrame({
    "x": x,
    "y": y
})

df.to_csv("data_1000.csv", index=False)

print("CSV created with 1000 points.")