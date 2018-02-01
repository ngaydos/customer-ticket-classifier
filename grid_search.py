from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer import Transformer

gradient_grid = {'learning_rate': [0.005, 0.01, 0.05], 
                          'max_depth': [5, 10, 15],
                          'min_samples_split':[10, 40, 80, 160],
                          'min_samples_leaf': [1, 10, 25, 50],
                          'subsample': [0.8, 1.0],
                          'max_features': ['sqrt'], 
                          'n_estimators': [100],
                          'random_state': [1]}

def grid_search(X_train, y_train, model=GradientBoostingClassifier(), params =gradient_grid):
    searcher = GridSearchCV(model, params, scoring = 'roc_auc')
    searcher.fit(X_train, y_train)
    best_params = searcher.best_params_
    best_score = searcher.best_score_
    return best_params


