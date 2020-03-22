import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.api import add_constant

#%%

url = 'C:/Users/narxi/ml/01 - Linear-Regression/USA_Housing.csv'
df = pd.read_csv(url)
df

#%%

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

#%%

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

#%%

lm = LinearRegression()
lm.fit(X_train, y_train)

#%%

lm.intercept_

#%%

lm.coef_

#%%

pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])

#%%

predictions = lm.predict(X_test)
predictions

#%%

plt.scatter(y_test, predictions)

#%%

np.sqrt(metrics.mean_squared_error(y_test, predictions))

#%%

metrics.explained_variance_score(y_test, predictions)
lm.score(X, y)

#%%

sns.distplot((y_test - predictions), bins=50)

#%% Add a constant term (not automatically added in statsmodels) 
X = add_constant(X)

#%%

regressor_OLS = sm.OLS(endog=y.astype(float), exog=X.astype(float)).fit()

regressor_OLS.summary()