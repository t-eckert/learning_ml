import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

# Encode categorical data
lenc_x = LabelEncoder()
x[:, 3] = lenc_x.fit_transform(x[:, 3])
ohenc_x = OneHotEncoder(categorical_features=[3])
x = ohenc_x.fit_transform(x).toarray()

# Removes the first variable of x - avoids dummy var trap
x = x[:,1:]

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
'''
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
'''

# Fitting MLR to training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediction
y_pred = regressor.predict(x_test)

# Backward Elimination
# Handling the constant
x = np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())
x_opt = x[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())
x_opt = x[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())