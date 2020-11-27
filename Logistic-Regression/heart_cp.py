import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import add_constant
from scipy.special import expit

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

print(
"""
=====
Odds
=====
""")

print(np.exp(coef_logit))

print(
"""
Patients with typical angina has 5.77 times higher odds of having
heart disease than those that are asymptomatic

Patients with atypical angina has 2.13 times higher
odds of having heart disease than those with typical angina
(odds of atypical vs asymptomatic/odds of typical
vs asymptomatic = 12.26/5.76)
"""
)

print(
"""
=============
Probabilities
=============
""")

coef_constant = coef_logit[0]

print(expit(coef_constant + coef_logit))

print(
"""
Patients with typical angina has 68% higher probability of having
heart disease than those that are asymptomatic ('higher probability'
because when we look back at the odds, it was >1, if it was <1,
it would then be 'lower probability')
"""
)
