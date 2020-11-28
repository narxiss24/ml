import pandas as pd

# import numpy as np
# import statsmodels.api as sm
# from scipy.special import expit

# pd.set_option('display.max_rows', None)

# Get data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
              'thal', 'hd']

# Drop rows that has '?' as missing values
df = df[~(df == '?').any(axis=1)]

df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg'], drop_first=True, prefix_sep='')
