import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import numpy as np

data = np.loadtxt('spiral.txt', skiprows=1)

X = data[:, :2]
true_labels = data[:, 2]

plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
plt.title("True Cluster")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
rand_index_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
print('Rand Index for Kmeans Clustering:', rand_index_kmeans)

single_link = AgglomerativeClustering(n_clusters=3, linkage='single')
single_link_labels = single_link.fit_predict(X)
rand_index_single_link = adjusted_rand_score(true_labels, single_link_labels)
print('Rand Index for Single-link Hierarchical Clustering:', rand_index_single_link)

complete_link = AgglomerativeClustering(n_clusters=3, linkage='complete')
complete_link_labels = complete_link.fit_predict(X)
rand_index_complete_link = adjusted_rand_score(true_labels, complete_link_labels)

print("Rand Index for Complete-link Hierarchical Clustering:", rand_index_complete_link)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")