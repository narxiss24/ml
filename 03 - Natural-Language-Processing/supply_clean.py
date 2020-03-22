import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from function import text_process
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://caremeasurement:p2mCH.$V]eTJCu!3@staging.cgj8ilbmhg0t.us-east-1.rds.amazonaws.com/classification')

df = pd.read_sql_query('SELECT * FROM `supply_unique_match_name`', engine)
print(df.head())

#%%
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SGDClassifier(loss="log", alpha=0.00001, penalty="l2", max_iter=5, tol=None)),
])

# msg_train, msg_test, label_train, label_test = \
# train_test_split(df['gmdnPTName'], df['supply_name'], test_size=0.2)

# pipeline.fit(df['gmdnPTName'],df['supply_name'])
# predictions = pipeline.predict(msg_test)

# print(classification_report(predictions,label_test))

#%%
classifier = pipeline.fit(df['gmdnPTName'],df['supply_name'])
joblib.dump(classifier, 'supply_model.pkl')