import pandas as pd
#import numpy as np
#import statsmodels.api as sm
#from statsmodels.api import add_constant
#from scipy.special import expit

pd.set_option('display.max_rows', None)

# Get data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')

df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']

print(df.trestbps)