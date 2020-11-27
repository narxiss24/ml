import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import expit

df = pd.read_csv('https://raw.githubusercontent.com/narxiss24/datasets/master/titanic_train.csv',
                 usecols=['Pclass', 'Survived'])

pclass = pd.get_dummies(df['Pclass'], drop_first=True)

pclass.columns = ['2nd', '3rd']

X = sm.add_constant(pclass)

Y = df['Survived']

mdl = sm.Logit(endog=Y, exog=X, missing='drop').fit()

print(mdl.summary())

print(
    """
    ====
    Odds
    ====
    """
)

print(np.exp(mdl.params))

print(
    """
The odds for second class passengers to survive is 0.52 times the odds of those that 
are in the first class, holding other independent variables fixed 

The odds for second class passengers to survive is 48% less [(1-0.52)*100] than those 
in the first class

The odds for second class passengers to survive is 2.89 times the odds of those that are
in the third class (odds of 2nd/odds of 3rd = 0.52/0.18)
    """
)

print(
    """
=============
Probabilities
=============
    """
)

coef = mdl.params[0]

print(expit(coef + mdl.params))

print(
    """
Passengers in the first class has a 74% probability of surviving

Passengers in the second class has a 47% probability of surviving

Passengers in the third class has a 24% probability of surviving
    """
)
