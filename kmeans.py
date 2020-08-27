import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=42)
y_pred = kmeans.fit_predict(x)
print(y_pred)

plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100,c='red',label='Cluster1')
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c='green',label='Cluster3')
plt.scatter(x[y_pred == 3,0],x[y_pred == 3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(x[y_pred == 4,0],x[y_pred == 4,1],s=100,c='magenta',label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='yellow',label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()