import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# %% Reading CSV file
train = pd.read_csv('https://raw.githubusercontent.com/narxiss24/datasets/master/titanic_train.csv')

# %% Heatmap to visualize outliers
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.show()

# %% Visualizing categorical variables
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train)

plt.show()


# %%
sns.countplot(x='SibSp', data=train)

# %% Visualizing numerical variables
sns.distplot(train['Age'].dropna(), bins=30)
train['Age'].dropna().hist(bins=30)

plt.show()
# %%
train['Fare'].hist(bins=40, figsize=(10, 4))


# %% Using boxplot
# plt.figure(figsize=(10,7))
# sns.boxplot(x='Pclass',y='Age',data=train)

# %% Creating a function to impute age based on average age in each class

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# %% Applying impute function to age column
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# %% Dropping useless column
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

# %%
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# %%
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
pclass = pd.get_dummies(train['Pclass'], drop_first=True)

# %%
train = pd.concat([train, sex, embark, pclass], axis=1)

# %%
train.drop(['Sex', 'PassengerId', 'Embarked', 'Name', 'Ticket', 'Pclass'], axis=1, inplace=True)

# %%
train.columns = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Male', 'Queen Mary', 'Southampton', 'Second class',
                 'Third class']

# %%
X = train.drop('Survived', axis=1)
y = train['Survived']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%
logmodel = LogisticRegression()

# %%
logmodel.fit(X_train, y_train)

# %% Print predictions
prediction = (logmodel.predict(X_test))

# %% Print classification report
print(classification_report(y_test, prediction))

# %%
print(confusion_matrix(y_test, prediction))

# %% Getting a single row (passenger)
row = X_test.iloc[1].values.reshape(1, -1)

# %% Predicting survivabiliy of a single passenger
print(logmodel.predict(row))

# %% Getting prediction score of a single passenger
score = logmodel.predict_proba(row)
print(np.around(np.amax(score, axis=1), decimals=2))

# %% Using statsmodels
import statsmodels.api as sm
from statsmodels.api import add_constant

# %%
X = add_constant(X)

# %% Using statsmodels

regressor_logistic = sm.Logit(endog=y.astype(float), exog=X.astype(float)).fit()

regressor_logistic.summary()

# %% Dropping non-significant predictors
# X = X.drop(['Parch', 'Fare', 'Queen Mary', 'Southampton'], axis=1)

# %% Using statsmodels

regressor_logistic = sm.Logit(endog=y.astype(float), exog=X.astype(float)).fit()

regressor_logistic.summary()

# %% Getting odd ratios
np.exp(regressor_logistic.params)

# %%

# The odds of surviving in 3rd class is 0.07 times the odds of surviving in the 1st class, holding other independent variables fixed 
# OR
# The odds of surviving is 90% lower ((0.10-1)*100) for 3rd class passengers than for 1st class passengers, holding other independent variables fixed

# The odds of surviving in 2nd class is 0.36 times the odds of surviving in the 1st class, holding other independent variables fixed 
# OR
# The odds of surviving is 64% lower ((0.36-1)*100) for 2nd class passengers than for 1st class passengers, holding other independent variables fixed
