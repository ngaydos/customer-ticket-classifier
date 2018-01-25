import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


df = pd.read_csv('data/realcapstonedata.csv')

class Transformer():

    def __init__(self):
        pass

    def transform_train(self, df, txt_columns, target_column):
        '''Takes a dataframe of the training data and returns 
        a cleaned dataset model training or testing
        input = pandas_df including porting y-value, list of strings, string
        output = 2 pandas datasets, one containing y_value, 
        one containing all relevant x data.
        '''
        bool_check = False
        no_nas = df[df[target_column].notna()]
        for i in range(len(txt_columns)):
            if bool_check:
                X += ' ' + no_nas[txt_columns[i]].str.lower()
            else:
                X = no_nas[txt_columns[i]].copy().str.lower()
                bool_check = True
        X = X.fillna('empty')
        y = no_nas[target_column]
        return X, y

    def transform_test(self, df, txt_columns):
        '''Takes in a dataframe of actual data and returns the a transformed dataframe
        for running in the model. '''
        bool_check = False
        for i in range(len(txt_columns)):
            if bool_check:
                X += " " + df[txt_columns[i]].str.lower()
            else:
                X = df[txt_columns[i]].copy().str.lower()
                bool_check = True
        X = X.fillna('empty')
        return X

class Modeler():

    def __init__(self, model= GradientBoostingClassifier()):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', model)
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