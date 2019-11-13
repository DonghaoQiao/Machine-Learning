from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# load data
iris = datasets.load_iris()
x = iris.data
x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
y = iris.target
fig = plt.figure(figsize=(15,5))

# original distribution
plt.subplot(131)
plt.scatter([xi[2] for xi in x],[xi[3] for xi in x],c=y)
plt.title('Origin')

# K-means cluster
plt.subplot(132)
y_pred = KMeans(n_clusters=3).fit_predict(x)
plt.scatter([xi[2] for xi in x],[xi[3] for xi in x],c=y_pred)
plt.title('K-means')

# DBSCAN cluster
plt.subplot(133)
y_pred = DBSCAN(eps = 0.1, min_samples = 5).fit_predict(x)
plt.scatter([xi[2] for xi in x],[xi[3] for xi in x],c=y_pred)
plt.title('DBSCAN')

plt.show()
