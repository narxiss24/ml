from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np


def train():
    iris = load_iris()
    X = iris['data'][:-1, :]
    y = iris['target'][:-1]

    model = LogisticRegression()
    model.fit(X, y)

    return iris, model


def predict():
    iris, model = train()

    pred = model.predict(iris['data'][-1, :].reshape(1, -1))
    pred_proba = model.predict_proba(iris['data'][-1, :].reshape(1, -1))

    print(pred, np.around((pred_proba), decimals=2))


if __name__ == '__main__':
    predict()
