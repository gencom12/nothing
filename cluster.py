import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
dataset = pd.read_csv('Live.csv')

# Remove columns with 100% null values
dataset.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

# Drop columns with unique values
unique_vars = [col for col in dataset.columns if dataset[col].nunique() == dataset.shape[0]]
dataset.drop(columns=unique_vars, inplace=True)

# Handle missing values
dataset.dropna(inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
dataset['status_type'] = label_encoder.fit_transform(dataset['status_type'])

# Feature scaling
scaler = StandardScaler()
numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Find optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(dataset[numerical_features])
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow method, select the optimal number of clusters
optimal_clusters = 5  # Assuming optimal clusters to be 5 based on the elbow method

# Perform k-means clustering
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(dataset[numerical_features])
dataset['cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 8))
for i in range(optimal_clusters):
    plt.scatter(dataset[dataset['cluster'] == i]['num_reactions'], dataset[dataset['cluster'] == i]['num_comments'], s=50, label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Number of Reactions (Scaled)')
plt.ylabel('Number of Comments (Scaled)')
plt.legend()
plt.show()

# Identify majority status_type in each cluster
cluster_majority_status = dataset.groupby('cluster')['status_type'].agg(lambda x: x.value_counts().index[0]).reset_index()
print("Majority status_type for each cluster:")
print(cluster_majority_status)
