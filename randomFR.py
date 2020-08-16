import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

rfr = RandomForestRegressor(n_estimators=10 , random_state=0)
rfr. fit(x,y)

rfr.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color='red')
plt.plot(x_grid,rfr.predict(x_grid), color='blue')
plt.title('Truth ot Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x,y, color='red')
plt.plot(x,rfr.predict(x), color='blue')
plt.title('Truth ot Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()