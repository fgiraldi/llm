import numpy as np
from sklearn.cluster import KMeans
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


# Determine optimal clusters using Elbow method (optional)
def find_optimal_clusters(embeddings, max_k=10):
    inertia = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)

    plt.plot(range(2, max_k), inertia, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.show()


# Uncomment to run Elbow Method
# find_optimal_clusters(embeddings)

# Apply K-Means Clustering
optimal_k = 4  # Choose based on the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# Reduce dimensions for visualization (PCA)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=clusters, palette="tab10")
plt.title("Review Clusters from ChromaDB")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig('clusters_kmeans.png')
plt.show()

# Show sample reviews per cluster
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    sample_reviews = np.array(reviews)[clusters == cluster]
    print(sample_reviews[:5])  # Show 5 sample reviews from this cluster
