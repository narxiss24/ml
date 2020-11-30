import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import add_constant
from scipy.special import expit

# Get data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
              'thal', 'hd']

# For hd, set 0 = healthy, 1 = heart disease
df['hd'] = df['hd'].apply(lambda x: 1 if x > 0 else x)

# Set chest pains as dummy variables and rename cols
cp = pd.get_dummies(df['cp'])

cp.columns = ['cp1', 'cp2', 'cp3', 'cp4']

# Set cp4 (No chest pain) as baseline
cp.drop('cp4', axis=1, inplace=True)

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
The odds of having heart disease for patients with typical 
angina is 0.17 times the odds of those that are asymptomatic,
holding other independent variables fixed 

The odds of having heart disease is 83% lower [(1-0.17)*100]
for patients with typical angina than those that are asymptomatic

The odds of having heart disease for patients with typical
angina is 2.13 times the odds of those with atypical angina 
(odds of cp1/odds of cp2 = 0.17/0.08)

The odds of having heart disease is 113% higher [(2.13-1)*100]
for patients with typical angina than those with atypical angina
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
Patients who are asymptomatic has 89% probability of having 
heart disease (doesn't make sense, maybe asymptomatic
during a follow up visit)

Patients with typical angina has 32% probability of having 
heart disease

Patients with atypical angina has 18% probability of having 
heart disease

Patients with non-anginal pain has 21% probability of having 
heart disease
    """
)
