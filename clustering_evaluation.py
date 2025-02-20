import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from chromadb_class import ReviewVectorDB


# Connect to ChromaDB
start_time = time.time()
chroma_client = ReviewVectorDB(persist_directory="./chroma_db")
collection = chroma_client.client.get_collection(name="app_reviews")  # Update with your collection name

# Retrieve all stored embeddings and metadata
results = collection.get(include=["documents", "embeddings"])
# Extract embeddings and corresponding reviews
embeddings = np.array(results["embeddings"])  # Shape: (num_reviews, embedding_dim)
reviews = [document for document in results["documents"]]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to load {len(embeddings)} items from a ChromaDB")


def evaluate_clustering(embeddings, k_range=(2, 10)):
    best_k = None
    best_score = -1
    scores = {}

    for k in range(*k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Silhouette score (higher is better)
        sil_score = silhouette_score(embeddings, labels)
        scores[k] = sil_score

        print(f"K={k} -> Silhouette Score: {sil_score:.4f}")

        if sil_score > best_score:
            best_score = sil_score
            best_k = k

    return best_k, scores


best_k, scores = evaluate_clustering(embeddings)
print(f"Optimal number of clusters: {best_k}")


def davies_bouldin_test(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    db_score = davies_bouldin_score(embeddings, labels)
    print(f"Davies-Bouldin Score for K={k}: {db_score:.4f}")
    return db_score


db_score = davies_bouldin_test(embeddings, best_k)
print(f"davies_bouldin_test: {db_score}")
