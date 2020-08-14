import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

lr = LinearRegression()
lr.fit(x,y)

pf = PolynomialFeatures(degree = 4)
x_poly = pf.fit_transform(x)

lr_2 = LinearRegression()
lr_2.fit(x_poly , y)

plt.scatter(x,y, color='red')
plt.plot(x,lr.predict(x), color='blue')
plt.title('Truth ot Bluff (Linear_Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='red')
plt.plot(x_grid,lr_2.predict(pf.fit_transform(x_grid)), color='blue')
plt.title('Truth ot Bluff (Polynomial_Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(lr.predict([[6.5]]))
print(lr_2.predict(pf.fit_transform([[6.5]])))