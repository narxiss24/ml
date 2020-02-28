#%% Import nltk

#import nltk
#nltk.download_shell()

#%% List the first ten messages

messages = [line.rstrip() for line in open('C:/Users/narxi/py-crash-course/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection')]
print(len(messages))

messages[50]

for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print('/n')

#%% Create pandas table    
import pandas as pd
messages = pd.read_csv('C:/Users/narxi/py-crash-course/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])
print(messages.head())

#%%
print(messages.describe())

#%%
print(messages.groupby('label').describe())

#%%
messages['length'] = messages['message'].apply(len)
print(messages.head())

#%%
messages['length'].plot.hist(bins=150)

#%%
print(messages['length'].describe())

#%%
print(messages[messages['length'] == 910]['message'].iloc[0])

#%%
messages.hist(column='length', by='label', bins=60, figsize=(12,4))

#%%
import string

mess = 'Sample message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]
print(nopunc)

#%%
from nltk.corpus import stopwords
stopwords.words('english')

#%%
nopunc = ''.join(nopunc)
print(nopunc)

#%%
print(nopunc.split())

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print(clean_mess)

#%% Creating the function
def text_process(mess):
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#%%

print(messages['message'].head(5).apply(text_process))

#%%
from sklearn.feature_extraction.text import CountVectorizer

#%%

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

#%%
print(len(bow_transformer.vocabulary_))

#%%
mess10 = messages['message'][9]
print(mess10)

#%%
bow10 = bow_transformer.transform([mess10])
print(bow10)

#%%
print(bow10.shape)

bow_transformer.get_feature_names()[9554]

#%%
messages_bow = bow_transformer.transform(messages['message'])
print(messages_bow.nnz)

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

#%%
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))

#%%
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf10 = tfidf_transformer.transform(bow10)
print(tfidf10)

#%%

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

#%%

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

#%%

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

#%%

print('predicted:', spam_detect_model.predict(tfidf10)[0])
print('expected:', messages.label[9])

#%%
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

#%%
from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))

#%%
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

#%%
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

#%%

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

#%%

print(classification_report(predictions,label_test))

#%%
import joblib

classifier = pipeline.fit(msg_train,label_train)
joblib.dump(classifier, 'spam_model.pkl')

#%%
mdl = joblib.load('spam_model.pkl')

#%%
predictions = mdl.predict(msg_test)

#%%
print(predictions)

#%%










