#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:01:28 2019

@author: kashishahuja
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualizing Linear Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X , lin_reg.predict(X) , color = 'blue')
plt.title('Truth or Bluff(Linear)')
plt.xlabel('Postion label')
plt.ylabel('Salary')
plt.show

#Visualizing Polynomial Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)) , color = 'blue')
plt.title('Truth or Bluff(Polynomial)')
plt.xlabel('Postion label')
plt.ylabel('Salary')
plt.show

#Trying Polynomial with 3(cube) degree
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualizing Linear Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X , lin_reg.predict(X) , color = 'blue')
plt.title('Truth or Bluff(Linear)')
plt.xlabel('Postion label')
plt.ylabel('Salary')
plt.show

#Visualizing Polynomial Regression Model
plt.scatter(X,Y,color = 'red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)) , color = 'blue')
plt.title('Truth or Bluff(Polynomial)')
plt.xlabel('Postion label')
plt.ylabel('Salary')
plt.show

#Predicting a result using Linear model
lin_reg.predict([[6.5]])

#Predicting a result using Polynomial model
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))