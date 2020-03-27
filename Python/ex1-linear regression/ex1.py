""" Example 1 Of Andrew Ng's Machine Learning Course
author: David S. White
date: 2020-02-26

Script for Loading and Executing Functions 
"""

import numpy as np


" load the first Data Set "
data1 = np.loadtxt('ex1data1.txt',dtype = str)
n = len(data1)
x = np.zeros(n)
y = np.zeros(n)
for l in range(n):
    currentLine = data1[l]
    loc = currentLine.find(',')
    x[l] = currentLine[0:loc]
    y[l] = currentLine[loc+1:]


    