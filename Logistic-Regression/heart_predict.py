import heart_all
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

df = heart_all.get_data()

y = df['hd1']

df.drop('hd1', axis=1, inplace=True)

X = sm.add_constant(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = sm.Logit(y_train, X_train).fit()

yhat = model.predict(X_test)
prediction = list(map(round, yhat))

print('Actual values:', list(y_test.values))
print('Predictions:', prediction)

print(confusion_matrix(y_test.values, prediction))
print(accuracy_score(y_test.values, prediction))

plt.plot(yhat, y_test.values, 'bo')

plt.show()
