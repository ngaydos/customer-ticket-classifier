import numpy as np
import pandas as pd

df = pd.read_csv('data/realcapstonedata.csv')

class Transformer():

    def __init__(self):
        pass

    def transform_train(df):
        '''Takes a dataframe of the training data and returns 
        a cleaned dataset for training or testing
        input = pandas_df including porting y-value
        output = 2 pandas datasets, one containing y_value, 
        one containing all relevant x data.
        '''
        non_zeroes = list(df.count() != 0)
        non_zero_columns = df.columns[non_zeroes]
        dropped_df = df[non_zero_columns]
        X = dropped_df.drop('Assigned Group')
        #need to add additional cleaning code here

        y = dropped_df['Assigned Group'] == "Numbering"