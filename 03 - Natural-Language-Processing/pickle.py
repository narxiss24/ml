import string
import joblib
from nltk.corpus import stopwords

import pandas as pd

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

mdl = joblib.load('spam_model.pkl')

#messages = pd.read_csv('C:/Users/narxi/py-crash-course/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])

list = ['Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs'] 
ser = pd.Series(list) 

predictions = mdl.predict(ser)

df = pd.DataFrame(predictions, columns=['1'])
df = df.replace({'spam': 'Spam', 'ham': 'Ham'})
 