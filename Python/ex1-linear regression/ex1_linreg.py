#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:48:30 2020

Ex1: Andrew Ng's Machine Learning Course
Functions for Linear Regression

@author: dwhite7
"""

" cost function of linear regression "
import numpy as np 

def computeCost(X,y,theta):
    " compute cost function via means squared error (MSE)"
    m = len(y)
    hx = X.dot(theta)
    rss = np.sum((hx-y)**2)
    mse = 1/2/m * rss
    
    return mse


def gradientDescent(X,y,theta,alpha,num_iter):
    " gradient descent of linear regression"
    m = len(y)
    J = np.zeros(num_iter)
    for i in range(num_iter):
        hx = X.dot(theta)
        theta = theta - (alpha/m * (hx-y).dot(X))
        J[i] = computeCost(X,y,theta)
        
    return theta
    

def featureNormalize(X):
    "scale by mean substaction divided by std"
    mu = np.mean(X)
    sd = np.std(X)
    X = (X-mu)/sd
    
    return X


def normalEquation(X,y):
    " theta = inv(X'X)*X'y "
    [n,m] = np.shape(X)
    theta = np.zeros(m)
    theta = np.linalg.pinv((X.T).dot(X)).dot((X.T).dot(y))
    
    return theta
    
    
