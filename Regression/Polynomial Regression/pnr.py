# Data Preprocessing Template

# Importing the libraries
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(X, y)

# Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree=4)
x_poly = poly_regr.fit_transform(X)
lin_regr_2 = LinearRegression()
lin_regr_2.fit(x_poly, y)
# pdb.set_trace()

# Visualizing the Linear Regression 
# plt.scatter(X, y, color='red')
# plt.plot(X, lin_regr.predict(X), color='blue')


# Visualizing the Polynomial Regression
# plt.scatter(X, y, color='red')
# plt.plot(X, lin_regr_2.predict(poly_regr.fit_transform(X)))
# plt.title("Truth or Bluff (Polynomial Regression)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# Predicting Salary with Linear Regression
print(lin_regr.predict(6.5))

# Predicting Salary with Polynomial Regression
print(lin_regr_2.predict(poly_regr.fit_transform(6.5)))