import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.special import expit

d = {'m_gene': [0,1,0,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0],
'cancer': [0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,1,1]
}

df = pd.DataFrame(data=d)

print(pd.crosstab(df.cancer, df.m_gene))

model = sm.Logit(df.cancer, sm.add_constant(df.m_gene)).fit()

print(model.summary())
print(np.exp(model.params))
print(expit(model.params))

print(np.log(11.25))