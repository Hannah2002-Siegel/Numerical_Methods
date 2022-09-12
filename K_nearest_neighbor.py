# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 21:43:33 2022

@author: hannah siegel
"""

#Resources: Python Engineer, What is K nearet neighbor? - IBM,
    
    
#KNN practice - just researched about KNN and am going to be trying it out. 
#don't have anything to test against, but I can cross that bridge when I get 
#there. This is going to be an implementation of theory and mathematical 
#verification, rather than practical verification.

import numpy as np
from collections import Counter

#global function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
    

#k --> number of nearest neighbors we want to consider
#
class KNN:
    
    def __init__(self, k=3):
        self.k = k
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    #predict new samples
    def predict(self, X):
        predicted_labels = [self._predict(x)for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x): #helper method
        #compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        
        #get k nearest samples, labels
        k_idxs = np.argsort(distances)[:self.k]#will sort distances and return the indicies
        k_nearest_labels = [self.y_train[i] for i in k_idxs]
        
        
        #majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        
