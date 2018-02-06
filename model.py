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
    '''Creates the Gradient Boosted model generally used by
    this project. Creates a pipeline that applies a count vectorizer,
    term frequency, indirect document frequency transformer and then creates
    a gradient boosted model (parameters grid searched)'''

    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', GradientBoostingClassifier(learning_rate = 0.05, 
                max_depth = 5, min_samples_leaf = 1,
                min_samples_split = 10, n_estimators = 100, subsample = 0.5))
            ])
    
    def fit(self, X, y):
        '''Fits the model.
        Inputs: X, a Series containing strings and y, a set of target values
        Outputs: None 
        '''
        self.model.fit(X, y)

    def predict_proba(self, X):
        '''Runs predict_proba on fitted the model
        Input: X, a pandas Series of strings
        Output: a series of arrays of predicted probabilities
        '''
        return self.model.predict_proba(X)

    def predict_binary(self, X, threshold = .5):
        '''Makes predictions based on two classes, allows for
        manual selection of threshold
        Input: X, a pandas Series of strings and threshold, float
        Output: Hard classifier predictions
        '''
        binary_preds = []
        #runs predict probas and then compares to threshold
        for value in self.predict_proba(X):
            if value[0] <= threshold:
                binary_preds.append(True)
            else:
                binary_preds.append(False)
        return binary_preds

    def predict(self, X):
        '''Runs predict on fitted the model
        Input: X, a pandas Series of strings
        Output: a series of hard classifier predictions
        '''
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()

class BayesModeler():
    '''Creates the Naive Bayes model used to maximize precision.
    Creates a pipeline that applies a count vectorizer, term frequency, 
    indirect document frequency transformer and then creates
    a naive bayes model'''

    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
            ])
    
    def fit(self, X, y):
        self.model.fit(X, y)
        '''Fits the model.
        Inputs: X, a Series containing strings and y, a set of target values
        Outputs: None 
        '''

    def predict_proba(self, X):
        '''Runs predict_proba on fitted the model
        Input: X, a pandas Series of strings
        Output: a series of arrays of predicted probabilities
        '''
        return self.model.predict_proba(X)

    def predict_binary(self, X, threshold = .5):
        '''Makes predictions based on two classes, allows for
        manual selection of threshold
        Input: X, a pandas Series of strings and threshold, float
        Output: Hard classifier predictions
        '''
        binary_preds = []
        #runs predict probas and then compares to threshold
        for value in self.predict_proba(X):
            if value[0] <= threshold:
                binary_preds.append(True)
            else:
                binary_preds.append(False)
        return binary_preds

    def predict(self, X):
        '''Runs predict on fitted the model
        Input: X, a pandas Series of strings
        Output: a series of hard classifier predictions
        '''
        return self.model.predict(X)