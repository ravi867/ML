# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs  # You can use your dataset instead

# Here, we'll generate a synthetic dataset for demonstration purposes
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Visualize the data (optional)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data Distribution")
plt.show()

# Standardize the data (important for K-means)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a K-means clustering model
kmeans = KMeans(n_clusters=4, random_state=42)  # we can change the number of clusters

# Fit the model to the data
kmeans.fit(X_std)

# Get cluster assignments for each data point
labels = kmeans.labels_

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Visualize the clustered data
plt.scatter(X_std[:, 0], X_std[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.title("K-means Clustering")
plt.legend()
plt.show()
