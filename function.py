import string
from nltk.corpus import stopwords
import inflect

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def get_digits(cost):
    count = 0
    initial = str(cost)[0]   

    while (cost > 0):
        cost = cost//10
        count = count + 1
        
    return '{}digits initial{}'.format(inflect.engine().number_to_words(count),inflect.engine().number_to_words(initial))
