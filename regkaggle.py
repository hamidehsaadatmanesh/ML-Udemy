# https://www.kaggle.com/foxtreme/linear-regression-project

import pandas as pd
import numpy as np
import matplotlib as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

import os
print(os.listdir("../input/"))

customers = pd.read_csv('../input/Ecommerce Customers')

# Check the head of customers, and check out its info() and describe() methods
customers.head()
customers.describe()
customers.info()

# Use seaborn to create a jointplot to compare the Time on Website / Time on App and Yearly Amount Spent columns
sns.jointplot(x='Time on Website',y ='Yearly Amount Spent', data = customers)
sns.jointplot(x='Time on App',y ='Yearly Amount Spent', data = customers)

# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(x='Time on App',y ='Length of Membership', data = customers, kind='hex')

# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.
# (Don't worry about the the colors)
sns.pairplot(customers)

# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
sns.set(color_codes=True)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=customers)

# Set a variable X equal to the numerical features of the customers 
# and a variable y equal to the "Yearly Amount Spent" column
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y= customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train )

# Print out the coefficients of the model
print(lm.coef_)

predictions = lm.predict(X_test)

plt.pyplot.scatter(y_test, predictions)
plt.pyplot.ylabel('Predicted')
plt.pyplot.xlabel('Y test')

# Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, predictions)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, predictions)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))

# Plot a histogram of the residuals and make sure it looks normally distributed
# Use either seaborn distplot, or just plt.hist()
sns.distplot((y_test-predictions))

pd.DataFrame(lm.coef_ , X.columns, columns=['Coeffecient'])