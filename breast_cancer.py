import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('breast_cancer.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print("Accuracy: [,.2f] %",format(accuracies.mean()*100))
print("Standard Dev: [,.2f] %",format(accuracies.std()*100))