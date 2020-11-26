import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import add_constant

# Get data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')

df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']

# For hd, set 0 = healthy, 1 = heart disease
df['hd'] = df['hd'].apply(lambda x: 1 if x<1 else 0)

# Set chest pains as dummy variables and rename cols
cp = pd.get_dummies(df['cp'])

cp.columns = ['cp1', 'cp2', 'cp3', 'cp4']

# Set cp4 (No chest pain) as baseline
cp.drop('cp4', axis=1 , inplace=True)

# Set cp as the independent variable (X) and add the constant term
X = add_constant(cp)

# Set hd as dependent variable (Y)
y = df['hd']

# Load statsmodels

regressor_logistic = sm.Logit(endog=y, exog=X).fit()

print(regressor_logistic.summary())

print(
"""
cp1 = typical angina
cp2 = atypical angina
cp3 = non-anginal pain
cp4 = asymptomatic
""")

# Get the odds from coefficients
coef_logit = regressor_logistic.params

coef_constant = coef_logit[0]

print(
"""
=====
Odds
=====
""")

print(np.exp(coef_logit))

print(
"""
Patients with typical angina has 5.77 times higher odds of developing
heart disease than those that are asymptomatic
"""
)

print(
"""
=============
Probabilities
=============
""")

def logit_to_probs(coef_constant, coef_logit):
    return np.exp(coef_constant+coef_logit)/(1+np.exp(coef_constant + coef_logit))

print(logit_to_probs(coef_constant, coef_logit))

print(
"""
Patients with typical angina has 68% higher probability of developing
heart disease than those that are asymptomatic
"""
)
