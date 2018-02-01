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

class Transformer():

    def __init__(self):
        pass

    def transform_train(self, df, txt_columns, target_column, stopwords = set(stopwords_)):
        '''Takes a dataframe of the training data and returns 
        a cleaned dataset model training or testing
        input = pandas_df including porting y-value, list of strings, string
        output = 2 pandas datasets, one containing y_value, 
        one containing all relevant x data.
        '''
        bool_check = False
        #remove all values where the target value does not exist
        no_nas = df[df[target_column].notna()]
        #loops over all the text columns specified and adds a lower
        #case version to a pandas series
        for i in range(len(txt_columns)):
            if bool_check:
                X += ' ' + no_nas[txt_columns[i]].str.lower()
            else:
                X = no_nas[txt_columns[i]].copy().str.lower()
                bool_check = True
        #fills any empty fields with a string that says "empty"
        X = X.fillna('empty')
        #removes punctuation and stopwords
        X_final = pd.Series()
        for text in X:
            text_clean = text
            for char in set(string.punctuation):
                #replace each item of punctuation with nothing
                text_clean = text_clean.replace(char, '')
            word_list = text_clean.split()
            for word in text_clean.split():
                if word in stopwords:
                    word_list.remove(word)
            X_final = X_final.append(pd.Series(" ".join(word_list)), ignore_index = True)
        y = no_nas[target_column]
        return X_final, y

    def transform_test(self, df, txt_columns, stopwords = set(stopwords_)):
        '''Takes in a dataframe of actual data and returns the a transformed dataframe
        for running in the model. '''
        stemmer = PorterStemmer()
        bool_check = False
        for i in range(len(txt_columns)):
            if bool_check:
                X += " " + df[txt_columns[i]].str.lower()
            else:
                X = df[txt_columns[i]].copy().str.lower()
                bool_check = True
        X = X.fillna('empty')
        X_final = pd.Series()
        for text in X:
            text_clean = text
            for char in set(string.punctuation):
                text_clean = text_clean.replace(char, '')
            word_list = text_clean.split()
            for word in text_clean.split():
                if word in stopwords:
                    word_list.remove(word)
            X_final = X_final.append(pd.Series(" ".join(word_list)), ignore_index = True)
        return X_final