import pandas as pd
#import numpy as np
#import statsmodels.api as sm
#from scipy.special import expit

#pd.set_option('display.max_rows', None)

# Get data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')

df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']

# Drop rows that has '?' as missing values
df = df[~(df == '?').any(axis=1)]

column_list = ['sex', 'cp', 'fbs']

for i in column_list:
    dummies = pd.get_dummies(df[i], drop_first=True).rename(columns=lambda x: i + str(int(x)))
    
    df.drop(i, axis=1, inplace=True)
    
    df = df.merge(dummies, left_index=True, right_index=True)
    

print(df)
