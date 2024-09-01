## kmeans.py

def p(str):
    print(str, '\n')


# K-Means Clustering
# Load libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# Set random seed
np.random.seed(42)

# Randomly generate 150 data points with 4 centers
points, labels = make_blobs(
    n_samples=150,   # Number of generated samples
    centers=4,       # Number of centers
    n_features=2,    # Number of features
    random_state=42  # Random seed
)

# Print the first 10 coordinates of the randomly generated points
# print(points.shape, '\n', points[:10])
# print(labels.shape, '\n', labels[:10])

# DataFrame creation
points_df = pd.DataFrame(points, columns=['X', 'Y'])
p(points_df)

# Set up and plot the graph
figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
axes.scatter(points_df['X'], points_df['Y'], label='Random Data')
# axes.grid()
# axes.legend()
# plt.show()


# K-means cluster creation

# Import kmeans library
from sklearn.cluster import KMeans

# Create clusters
k_cluster = KMeans(n_clusters=4) # 4 clusters -> This is essentially the only input we can provide.

# Fit the data into the cluster
k_cluster.fit(points)

# p(k_cluster.labels_) # Labels
# p(np.shape(k_cluster.labels_)) # 150 points
# p(np.unique(k_cluster.labels_)) # Unique values of labels

# Color dictionary
color_di = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'black'
}

# Plot the graph
# plt.figure(figsize=(10, 6))
# for cluster in range(4):
#     cluster_sub = points[k_cluster.labels_ == cluster]
#     plt.scatter(
#         cluster_sub[:, 0],                  # First feature
#         cluster_sub[:, 1],                  # Second feature
#         c = color_di[cluster],              # Color by cluster
#         label = f"Cluster {cluster}"        # Label by cluster
#     )
# plt.grid(True)
# plt.legend()
# plt.show()


## K-Means Circular Cluster Creation

# Libraries
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles

# n_samples: Number of samples
# factor: Scale factor between the inner and outer circle
# noise: The closer to 0, the less noise; the higher, the more noise
circle_point, circle_labels = make_circles(n_samples=150, factor=0.5, noise=.01)

# Set graph size
plt.figure(figsize=(10, 6))

# Create model
circles_kmeans = KMeans(n_clusters=2)

# Train model
circles_kmeans.fit(circle_point)

# Color dictionary
color_di = {0: 'blue', 1: 'red'}

# Scatter plot
for i in range(2):
    cluster_sub = circle_point[circles_kmeans.labels_ == i]
    plt.scatter(
        cluster_sub[:, 0],
        cluster_sub[:, 1],
        c = color_di[i],
        label = f"cluster_{i}"
    )
plt.grid(True)
plt.legend()
plt.show()

# Scatter plot by cluster with colors applied
# for cluster in range(3):
#     cluster_sub = X[diag_kmeans.labels_ == cluster]
#     axes.scatter(cluster_sub[:, 0], cluster_sub[:, 1], c=color_di[cluster],
#                label=f"cluster {cluster}")
# plt.legend()
# plt.show()

X, y = make_blobs(n_samples=200, random_state=163)
# p(X)
# p(y)

# Transformation matrix
transformation = [[0.6, -0.6], [-0.3, 0.8]]

# dot(): Perform matrix multiplication between the array and matrix
# and apply linear transformation to all data points in the X array
diag_points = np.dot(X, transformation)
#p(diag_points)

# Graph size
#figure = plt.figure(figsize=(10, 6))

# Add subplot
#axes = figure.add_subplot(111)

# Create cluster
#diag_kmeans = KMeans(n_clusters=3)

# Train model
#diag_kmeans.fit(X)

# Color dictionary
#color_di = {0: "red", 1: "blue", 2: "green"}

# Scatter plot by cluster with colors applied
#for cluster in range(3):
#    cluster_sub = X[diag_kmeans.labels_ == cluster]
#    axes.scatter(cluster_sub[:, 0], cluster_sub[:, 1], c=color_di[cluster],
#                 label=f"cluster {cluster}")
#plt.legend()
#plt.show()



## DBSCAN: Density-Based Clustering Algorithm

# Load library
from sklearn.cluster import DBSCAN

figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
color_di = {1: 'red', 1: 'blue', 2: 'green', 3: 'black', 4: 'orange'}

epsilon = 0.5 # Radius
minPts = 3 # Minimum number of adjacent points

# Create DBSCAN
# eps: Radius of the cluster
# min_samples: Minimum number of adjacent points to form a cluster
# metric: Parameter specifying the distance metric, default is Euclidean distance
# algorithm: Parameter selecting the clustering algorithm, default is auto
diag_dbscan = DBSCAN(eps=epsilon, min_samples=minPts) # Create

# Train model
diag_dbscan.fit(diag_points)

# Number of clusters, DBSCAN's cluster numbers include negative values, so we add 1
n_cluster = max(diag_dbscan.labels_) + 1
# p(n_cluster)

# Cluster numbers of each data point
# -1: noise, 0~4: cluster number
p(diag_dbscan.labels_)

# Scatter plot
figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
color_di = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'orange'}
for i in range(n_cluster):
    cluster_sub = diag_points[diag_dbscan.labels_ == i]
    plt.scatter(
        cluster_sub[:, 0],
        cluster_sub[:, 1],
        c = color_di[i]
    )
axes.grid(True)
plt.show()
