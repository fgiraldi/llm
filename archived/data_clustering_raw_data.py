from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from collections import Counter
from typing import TypedDict
import pandas as pd
import spacy


class ReviewChromaDocMetadata(TypedDict):
    rating: str
    date: str
    category: str


class ReviewChromaDoc:
    text: str
    metadata: ReviewChromaDocMetadata

    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


def analyze_review_trends(chroma_documents: list[ReviewChromaDoc]):
    """
    Analyze trends in app reviews using clustering and topic modeling.

    Parameters:
    chroma_documents: List of documents from Chroma DB containing review text and metadata

    Returns:
    dict: Analysis results including clusters, topics, and temporal trends
    """
    # Extract text and metadata
    reviews = []
    metadata = []

    for doc in chroma_documents:
        reviews.append(doc.text)
        metadata.append({
            'rating': float(doc.metadata['rating']),  # Ensure rating is float
            'date': doc.metadata['date'],
            'category': doc.metadata['category']
        })

    # Load spaCy for text preprocessing
    nlp = spacy.load('en_core_web_sm')

    def preprocess_text(text):
        doc = nlp(text)
        # Keep only nouns, adjectives, and verbs; remove stop words
        tokens = [
            token.lemma_.lower() for token in doc
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'])
            and not token.is_stop
            and token.lemma_.isalpha()
        ]
        return ' '.join(tokens)

    # Preprocess reviews
    processed_reviews = [preprocess_text(review) for review in reviews]

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(processed_reviews)

    # Perform K-means clustering
    n_clusters = 5  # Adjust based on your needs
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Calculate cluster sizes
    cluster_sizes = Counter(cluster_labels)
    cluster_percentages = {
        f'Cluster_{k}': {
            'count': v,
            'percentage': (v / len(cluster_labels) * 100)
        }
        for k, v in cluster_sizes.items()
    }

    # Perform topic modeling using NMF
    n_topics = 5  # Adjust based on your needs
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit_transform(tfidf_matrix)

    # Get top terms for each cluster
    cluster_terms = {}
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-10:][::-1]  # Get top 10 terms
        top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_indices]
        cluster_terms[f'Cluster_{i}'] = {
            'terms': top_terms,
            'size': cluster_sizes[i],
            'percentage': cluster_percentages[f'Cluster_{i}']['percentage']
        }

    # Get top terms for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_terms = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[:-10-1:-1]
        top_terms = [feature_names[i] for i in top_indices]
        topic_terms[f'Topic_{topic_idx}'] = top_terms

    # Analyze temporal trends
    df = pd.DataFrame(metadata)
    df['cluster'] = cluster_labels.astype(int)  # Ensure cluster is int
    df['date'] = pd.to_datetime(df['date'])

    # Monthly trend analysis
    monthly_trends = df.set_index('date').resample('ME').agg({
        'rating': 'mean',
        'cluster': lambda x: int(x.mode().iloc[0]) if not x.empty else -1  # Get most common cluster as int
    }).reset_index()

    # Category analysis
    category_clusters = df.groupby(['category', 'cluster']).size().unstack(fill_value=0)

    results = {
        'cluster_terms': cluster_terms,
        'topic_terms': topic_terms,
        'monthly_trends': monthly_trends.to_dict('records'),
        'category_clusters': category_clusters.to_dict(),
        'cluster_labels': cluster_labels.tolist(),
        'cluster_sizes': cluster_percentages
    }

    return results


def visualize_trends(analysis_results):
    """
    Create visualizations for the analysis results.
    Returns a dictionary of plots and summaries.
    """
    # Convert monthly trends to DataFrame
    monthly_df = pd.DataFrame(analysis_results['monthly_trends'])

    summaries = {
        'clusters': {},
        'temporal': {},
        'categories': {}
    }

    # Summarize cluster information
    for cluster, terms in analysis_results['cluster_terms'].items():
        summaries['clusters'][cluster] = {
            'top_terms': terms,
            'size': analysis_results['cluster_labels'].count(
                int(cluster.split('_')[1])
            )
        }

    # Summarize temporal trends
    summaries['temporal'] = {
        'avg_rating_trend': monthly_df['rating'].tolist(),
        'dominant_clusters': monthly_df['cluster'].value_counts().to_dict()
    }

    return summaries
