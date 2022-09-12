# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:44:13 2022

@author: hannah siegel
"""
#example script for the linear regression.py

import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn import datasets
import matplotlib.pyplot as plt


X,y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 1234)


# fig = plt.figure(figsize = (8,6))
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
# plt.show()

# print(X_train.shape) #nD array or size 80,1
# print(y_train.shape) #1D row vector of size 80

#import linear_regression.py for testing
from linear_regression import LinearRegression

regressor = LinearRegression(lr = 0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)

#plotting
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize = (8,6))
m1 = plt.scatter(X_train, y_train, color = cmap(0,9), s = 10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s = 10)
plt.plot(X, y_pred_line, color = 'black', linewidth = 2, label = "Prediction")
plt.show()







