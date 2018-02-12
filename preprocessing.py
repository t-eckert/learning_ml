import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Impute missing values by mean
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
imp = imp.fit(x[:,1:3])

x[:, 1:3] = imp.transform(x[:, 1:3])

# Encode categorical data
lenc_x = LabelEncoder()
x[:,0] = lenc_x.fit_transform(x[:,0])
ohenc_x = OneHotEncoder(categorical_features=[0])
x = ohenc_x.fit_transform(x).toarray()

lenc_y = LabelEncoder()
y = lenc_y.fit_transform(y)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)