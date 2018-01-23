import numpy as np
import pandas as pd

df = pd.read_csv('data/realcapstonedata.csv')

class Transformer():

    def __init__(self):
        pass

    def transform_train(df, txt_columns, target_column):
        '''Takes a dataframe of the training data and returns 
        a cleaned dataset for training or testing
        input = pandas_df including porting y-value, list of strings, string
        output = 2 pandas datasets, one containing y_value, 
        one containing all relevant x data.
        '''
        bool_check = False
        no_nas = df[df[target_column].notna()]
        for i in range(len(txt_columns)):
            if bool_check:
                X += ' ' + no_nas[txt_columns[i]]
            else:
                X = no_nas[txt_columns[i]]
                boolcheck = True
        X = X.fillna('empty')
        y = no_nas[target_column]
        return X, y

    def transform_test(df, txt_columns):
        bool_check = False
        for i in range(len(txt_columns)):
            if bool_check:
                X += " " + df[txt_columns[i]]
            else:
                X = df[txt_columns[i]]
                boolcheck = True
        return X