# -*- coding: utf-8 -*-
"""
Created on Sat May 29 22:00:11 2021

@author: rkara
"""


class CardinalityTruncater:
 
    #constructor
    def __init__(self, treshold = 0.001):
        self.treshold = treshold
        self.frequents = {}
        
    def fit(self, X):
        for feature in X:
            unique_ratios = X[feature].value_counts(normalize=True) 
            self.frequents[feature] = unique_ratios[unique_ratios>self.treshold].index
        
        return self
        
    def transform(self, X_df):
        # Perform the transformation to new categorical data.
        X = X_df.copy()
        for feature in X:
            X[feature][~(X[feature].isin(self.frequents[feature])) & ~X[feature].isna()]= "other"  
            
        return X    