import numpy as np
import pandas as pd
import string
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#taken from common words @ http://www.textfixer.com/resources/common-english-words.txt
stopwords_ = "a,able,about,across,after,all,almost,also,am,among,an,and,any,\
are,as,at,be,because,been,but,by,can,could,dear,did,do,does,either,\
else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,\
how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,\
me,might,most,must,my,neither,no,of,off,often,on,only,or,other,our,\
own,rather,said,say,says,she,should,since,so,some,than,that,the,their,\
them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,\
what,when,where,which,while,who,whom,why,will,with,would,yet,you,your".split(',')



class BoostModeler():

    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', GradientBoostingClassifier(learning_rate = 0.05, 
                max_depth = 5, max_features = 'sqrt', min_samples_leaf = 1,
                min_samples_split = 10, n_estimators = 100, subsample = 0.5))
            ])
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_binary(self, X, threshold = .5):
        binary_preds = []
        for value in self.predict_proba(X):
            if value[0] <= threshold:
                binary_preds.append(True)
            else:
                binary_preds.append(False)
        return binary_preds

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()

class BayesModeler():

    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
            ])
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_binary(self, X, threshold = .5):
        binary_preds = []
        for value in self.predict_proba(X):
            if value[0] <= threshold:
                binary_preds.append(True)
            else:
                binary_preds.append(False)
        return binary_preds

    def predict(self, X):
        return self.model.predict(X)