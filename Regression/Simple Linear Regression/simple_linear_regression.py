import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Feature scaling
'''
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
'''

# Simple Linear Regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting Test Results
y_pred = regressor.predict(x_test)

# Visualization
plt.scatter(x_train, y_train, color='red')
plt.scatter(x_test, y_test, color='green')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary WRT Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()