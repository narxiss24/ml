import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://caremeasurement:p2mCH.$V]eTJCu!3@staging.cgj8ilbmhg0t.us-east-1.rds.amazonaws.com/classification')

df = pd.read_sql_query('SELECT * FROM `supply_unique_match`', engine)
supply_name = pd.read_sql_query('SELECT id, name FROM `supply_supplyname`', engine)

#%%
df.columns

#%%
df.drop(['drg',
         'hospital_supply_name',
         'manufacturer_number_cleaned',
         'versionModelNumberCleaned',
         'unique_match'
         ], axis=1, inplace=True)
df.dropna(inplace=True)

#%%
gmdnPTName = pd.get_dummies(df['gmdnPTName'],drop_first=True)

#%%
train = pd.concat([df[['cost','supply_name_id']],gmdnPTName],axis=1)

#%%
y = train['supply_name_id']

#%%
X = train.drop('supply_name_id', axis=1)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#%%
logmodel = LogisticRegression()

#%%
logmodel.fit(X_train,y_train)

#%% Print predictions
prediction = (logmodel.predict(X_test))

#%% Print classification report
print(classification_report(y_test, prediction))