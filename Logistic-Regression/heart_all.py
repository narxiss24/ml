import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit


# pd.set_option('display.max_rows', None)

def get_data():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')

    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                  'ca', 'thal', 'hd']

    df['hd'] = df['hd'].apply(lambda x: 1 if x > 0 else x)

    # Drop rows that has '?' as missing values
    df = df[~(df == '?').any(axis=1)]

    df = df[['hd', 'sex', 'oldpeak', 'thalach', 'ca', 'exang']]
    
    df = pd.get_dummies(df, columns=['hd', 'sex', 'exang', 'ca'], drop_first=True, prefix_sep='')

    return df


def load_model(df):
    y = df['hd1']

    df.drop('hd1', axis=1, inplace=True)

    X = sm.add_constant(df)

    model = sm.Logit(endog=y, exog=X).fit()

    return model


def main():
    df = get_data()

    model = load_model(df)

    print(model.summary())

    print(
        """
oldpeak: ST depression induced by exercise relative to rest 
thalach: maximum heart rate achieved
sex: (1 = male; 0 = female)
ca: number of major vessels (0-3) colored by flourosopy
exang: exercise induced angina (1 = yes; 0 = no)    
        """
    )

    print(
        """
====
Odds
====
        """
    )

    print(np.exp(model.params))

    print(
        """
=============
Probabilities
=============
        """
    )

    print(expit(model.params[0] + model.params))

    return df


if __name__ == '__main__':
    main()
