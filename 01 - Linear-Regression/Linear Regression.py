import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

url = 'https://raw.githubusercontent.com/narxiss24/machine-learning-practice/master/01%20-%20Linear-regression/USA_Housing.csv'

df = pd.read_csv(url)

df

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)

lm = LinearRegression()

lm.fit(X_train, y_train)

lm.intercept_

lm.coef_

pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])

predictions = lm.predict(X_test)

predictions

plt.scatter(y_test, predictions)

np.sqrt(metrics.mean_squared_error(y_test, predictions))

metrics.explained_variance_score(y_test, predictions)
lm.score(X, y)

sns.distplot((y_test - predictions), bins=50)

X = np.append(arr=np.ones((5000, 1)).astype(float), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y.astype(float), exog=X_opt.astype(float)).fit()

regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5]]

regressor_OLS = sm.OLS(endog=y.astype(float), exog=X_opt.astype(float)).fit()

regressor_OLS.summary()
