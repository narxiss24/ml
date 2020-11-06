import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Get data

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')

# %% Select Setosa and Versicolor

y = df.iloc[0:100, 4].values  # Get as an array
y = np.where(y == 'Iris-setosa', -1, 1)  # label Setosa as 1 and Versicolor as -1
