import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state=1)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,lr.predict(x_train), color='blue')
plt.title('salary vs exprience (Training Set)')
plt.xlabel('Years of Exprience')
plt.ylabel('Salary of Exprience')
plt.show()

plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,lr.predict(x_train), color='blue')
plt.title('salary vs exprience (Test Set)')
plt.xlabel('Years of Exprience')
plt.ylabel('Salary of Exprience')
plt.show()

# **Making a single prediction (for example the salary of an employee with 12 years of experience)**
print(lr.predict([[12]]))

# ## Getting the final linear regression equation with the values of the coefficients
print(lr.coef_)
print(lr.intercept_)