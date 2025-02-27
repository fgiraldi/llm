import json
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


start_time = time.time()

# Load JSON data
with open("json_files/records1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract 'review' texts and 'embeddings'
reviews = [item["review"] for item in data]
embeddings = [item["embeddings"] for item in data]  # Shape: (num_reviews, embedding_dim)

with open("json_files/records2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract 'review' texts and 'embeddings'
reviews.extend([item["review"] for item in data])
embeddings.extend([item["embeddings"] for item in data])  # Shape: (num_reviews, embedding_dim)

with open("json_files/records3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract 'review' texts and 'embeddings'
reviews.extend([item["review"] for item in data])
embeddings.extend([item["embeddings"] for item in data])  # Shape: (num_reviews, embedding_dim)

with open("json_files/records4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract 'review' texts and 'embeddings'
reviews.extend([item["review"] for item in data])
embeddings.extend([item["embeddings"] for item in data])  # Shape: (num_reviews, embedding_dim)

with open("json_files/records5.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract 'review' texts and 'embeddings'
reviews.extend([item["review"] for item in data])
embeddings.extend([item["embeddings"] for item in data])  # Shape: (num_reviews, embedding_dim)

with open("json_files/records6.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract 'review' texts and 'embeddings'
reviews.extend([item["review"] for item in data])
embeddings.extend([item["embeddings"] for item in data])  # Shape: (num_reviews, embedding_dim)
data = None
embeddings = np.array(embeddings)  # Shape: (num_reviews, embedding_dim)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to load {len(embeddings)} items from JSON files")


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
start_time = time.time()
optimal_k = 4  # Choose based on the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# Create DataFrame with reviews and their assigned cluster
df = pd.DataFrame({"review": reviews, "cluster": clusters})

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
# plt.show()

# Show sample reviews per cluster
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    sample_reviews = np.array(reviews)[clusters == cluster]
    print(sample_reviews[:5])  # Show 5 sample reviews from this cluster


# Function to extract keywords from each cluster
def extract_keywords_per_cluster(df, n_keywords=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    vectorizer.fit_transform(df["review"])  # Convert reviews to TF-IDF vectors
    feature_names = np.array(vectorizer.get_feature_names_out())  # Get words

    cluster_keywords = {}

    for cluster in sorted(df["cluster"].unique()):
        cluster_reviews = df[df["cluster"] == cluster]["review"]
        tfidf_cluster = vectorizer.transform(cluster_reviews)  # Apply TF-IDF to cluster reviews

        # **Fix: Sum TF-IDF scores instead of averaging**
        sum_tfidf = np.array(tfidf_cluster.sum(axis=0)).flatten()

        # Get top N keywords
        top_keywords_idx = np.argsort(sum_tfidf)[::-1][:n_keywords]  # Sort in descending order
        top_keywords = feature_names[top_keywords_idx]

        cluster_keywords[cluster] = top_keywords.tolist()

    return cluster_keywords


# Extract and print keywords
cluster_keywords = extract_keywords_per_cluster(df, n_keywords=10)

for cluster, keywords in cluster_keywords.items():
    print(f"\nCluster {cluster} Keywords: {', '.join(keywords)}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to process {len(embeddings)} items")
