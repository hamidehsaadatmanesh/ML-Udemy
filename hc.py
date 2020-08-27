import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

y_pred = hc.fit_predict(x)
print(y_pred)

plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100,c='red',label='Cluster1')
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c='green',label='Cluster3')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()