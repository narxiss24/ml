import os
import string
from sqlalchemy import create_engine
from dotenv import load_dotenv
from nltk.corpus import stopwords

load_dotenv()


def sql_engine(stage):
    engine = create_engine(
        'mysql+pymysql://{}:{}@{}.cgj8ilbmhg0t.us-east-1.rds.amazonaws.com/{}'.format(os.getenv('USER'),
                                                                                      os.getenv('PASSWORD'), stage,
                                                                                      'classification'))
    return engine


def text_process(msg):
    nopunc = [char for char in msg if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
