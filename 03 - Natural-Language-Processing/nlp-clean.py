import string
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

messages = pd.read_csv('C:/Users/narxi/py-crash-course/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SGDClassifier()),
])

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))

#%%
classifier = pipeline.fit(msg_train,label_train)
joblib.dump(classifier, 'spam_model.pkl')