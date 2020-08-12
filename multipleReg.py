import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])] , remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_pred = mlr.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

print(mlr.predict([[1, 0, 0, 160000, 130000, 300000]]))

## Getting the final linear regression equation with the 
print(mlr.coef_)
print(mlr.intercept_)