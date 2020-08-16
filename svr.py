import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y), 1)

sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

svr = SVR(kernel='rbf')
svr.fit(x,y)
sc_y.inverse_transform(svr.predict(sc_x.transform([[6.5]])))

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(svr.predict(x)), color='blue')
plt.title('Truth ot Bluff (Suport Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color='red')
plt.plot(x_grid,sc_y.inverse_transform(svr.predict(sc_x.transform(x_grid))), color='blue')
plt.title('Truth ot Bluff (Suport Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()