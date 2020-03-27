""" Example 1 Of Andrew Ng's Machine Learning Course
author: David S. White
date: 2020-02-26

Script for Loading and Executing Functions 
"""

import numpy as np
import matplotlib.pyplot as plt
import ex1_linreg as ex1

" Load the Data "
data = np.loadtxt('ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size

"Add parameter 1 to x -> X"
X = np.stack([np.ones(m), X], axis=1)

"init theta w/ zeros. Should return value of " 
theta = np.array((0,0))
J = ex1.computeCost(X,y,theta)
print(J)

"now with better values of theta"
theta = np.array((-1,2))
J = ex1.computeCost(X,y,theta)
print(J)

"now w/ gradient descent"
alpha = 0.01
num_iter = 5000
theta = [-5,2]
theta = ex1.gradientDescent(X, y, theta, alpha, num_iter)
print(theta)

" now with normal equation "
theta = ex1.normalEquation(X, y)
print(theta)


" Plot data w/ linear regression fit" 
plt.figure()
plt.scatter(X[:,1],y)
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit vs Population")


" Feature Normalization"
x = ex1.featureNormalize(X[:,1])

