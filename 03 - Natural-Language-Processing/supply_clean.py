import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from function import text_process
from sqlalchemy import create_engine

# %%

engine = create_engine('CM_URL')

df = pd.read_sql_query('SELECT * FROM `supply_unique_full_name`', engine)

# %%


def drg_to_string(drg):
    drg_string = 'drg{}'.format(drg)
    if drg_string == 'drgNone':
        return ''
    else:
        return drg_string


def cost_to_string(cost):
    if pd.notna(cost):
        if cost <= 99:
            return '<=99'
        elif 100 <= cost <= 149:
            return '100-149'
        elif 150 <= cost <= 299:
            return '150-299'
        elif 300 <= cost <= 499:
            return '300-499'
        elif 500 <= cost <= 999:
            return '500-999'
        elif 1000 <= cost <= 1499:
            return '1000-1499'
        elif 1500 <= cost <= 2999:
            return '1500-2999'
        elif 3000 <= cost <= 3999:
            return '3000-3999'
        elif 4000 <= cost <= 4999:
            return '4000-4999'
        else:
            return '>=5000'
    else:
        return ''


def gmdnPTName_to_string(gmdnPTName):
    if gmdnPTName == None:
        return ''
    else:
        return gmdnPTName

# %%


df['drg'] = df['drg'].apply(drg_to_string)
df['cost'] = df['cost'].apply(cost_to_string)
df['gmdnPTName'] = df['gmdnPTName'].apply(gmdnPTName_to_string)

# %%

df['X'] = df['hospital_supply_name'] + ' ' + \
    df['gmdnPTName'] + ' ' + df['cost'] + ' ' + df['drg']

# %%

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SGDClassifier(loss="log",
                                 alpha=0.00001, penalty="l2", max_iter=5, tol=None)),
])

# %%
classifier = pipeline.fit(df['X'], df['name'])
joblib.dump(classifier, 'supply_model.pkl')
