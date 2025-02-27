import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from chromadb_class import ReviewVectorDB

# Connect to ChromaDB
chroma_client = ReviewVectorDB(persist_directory="./chroma_db")
collection = chroma_client.client.get_collection(name="app_reviews")  # Update with your collection name

# Retrieve all stored embeddings and metadata
results = collection.get(include=["documents", "embeddings", "metadatas"])

# Extract embeddings and corresponding reviews
embeddings = np.array(results["embeddings"])  # Shape: (num_reviews, embedding_dim)
reviews = [document for document in results["documents"]]

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=0.4, min_samples=5, metric="cosine")  # Adjust parameters as needed
clusters = dbscan.fit_predict(embeddings)

# Reduce dimensions for visualization (PCA)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=clusters, palette="tab10")
plt.title("Review Clusters (DBSCAN) from ChromaDB")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig('clusters_dbscan.png')
plt.show()

# Show sample reviews per cluster
unique_clusters = set(clusters)
for cluster in unique_clusters:
    if cluster == -1:
        print("\nNoise / Outliers:")
    else:
        print(f"\nCluster {cluster}:")

    sample_reviews = np.array(reviews)[clusters == cluster]
    print(sample_reviews[:5])  # Show 5 sample reviews
