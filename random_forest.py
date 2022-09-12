# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 20:29:19 2022

@author: hannah siegel
"""



#random forest

import numpy as np
from decision_tree import DecisionTree
from collections import Counter


#global function
def bootstrap_sample(X,y):
    n_samples = X.shape[0] #first dimension is the number of samples and the second is the number of features
    idxs = np.random.choice(n_samples, size = n_samples, replace = True)
    return X[idxs],y[idxs]

def _most_common_label(y):
    counter = Counter(y) #calcualte the num of occurances related to the y
    most_common = counter.most_common(1)[0][0]
    return most_common #will determine the most common label

class RandomForest:
    
    def __init__(self, n_trees = 100, min_samples_split = 2, max_depth = 100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = [] #empty array of trees to store the trees we're going to create
        
    def fit(self, X, y): #training
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
            max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [_most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
