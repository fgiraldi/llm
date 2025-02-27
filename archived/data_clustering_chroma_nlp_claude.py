import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from textblob import TextBlob
import spacy
from collections import Counter
from chromadb_class import ReviewVectorDB
import matplotlib.pyplot as plt
import seaborn as sns


# Connect to ChromaDB
start_time = time.time()
chroma_client = ReviewVectorDB(persist_directory="./chroma_db")
collection = chroma_client.client.get_collection(name="app_reviews")  # Update with your collection name

# Retrieve all stored embeddings and metadata
results = collection.get(include=["documents", "embeddings", "metadatas"])
reviews = [document for document in results["documents"]]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to load {len(reviews)} items from a ChromaDB")


def analyze_app_issues(reviews_df, num_topics=5, min_topic_freq=0.02):
    """
    Analyze app reviews to identify specific issues and problems reported by users.

    Parameters:
    reviews_df: DataFrame with 'review_text' column
    num_topics: Number of topics to extract
    min_topic_freq: Minimum frequency threshold for topic significance

    Returns:
    Dict containing analysis results
    """
    # Define issue-related keywords
    issue_keywords = [
        'error', 'bug', 'crash', 'problem', 'issue', 'fail', 'broken', 'stuck',
        'slow', "doesn't work", 'not working', 'glitch', 'freezes', 'frozen',
        'terrible', 'poor', 'bad', 'difficult', 'unusable', 'annoying',
        'impossible', 'waste', 'disappointed', 'frustrating', 'awful'
    ]

    # Preprocess reviews and identify problem-related ones
    def preprocess_text(text):
        text = str(text).lower().strip()
        return text

    def contains_issue(text):
        text = str(text).lower()
        return any(keyword in text for keyword in issue_keywords)

    def get_sentiment(text):
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity

    reviews_df['processed_text'] = reviews_df['review_text'].apply(preprocess_text)
    reviews_df['sentiment'] = reviews_df['review_text'].apply(get_sentiment)
    reviews_df['has_issue'] = reviews_df['processed_text'].apply(contains_issue)

    # Filter for negative reviews or reviews containing issue keywords
    problem_reviews = reviews_df[
        (reviews_df['sentiment'] < 0) | (reviews_df['has_issue'])
    ].copy()

    if len(problem_reviews) == 0:
        return {"error": "No issues found in the reviews"}

    # Topic Modeling on problem reviews
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)  # Include bigrams for better context
    )
    doc_term_matrix = tfidf.fit_transform(problem_reviews['processed_text'])

    nmf_model = NMF(n_components=min(num_topics, len(problem_reviews)), random_state=42)
    topic_matrix = nmf_model.fit_transform(doc_term_matrix)

    # Get top words for each topic with their context
    feature_names = tfidf.get_feature_names_out()

    def get_top_words_per_topic(model, feature_names, n_words=10):
        topics = []
        word_scores = []
        example_reviews = []

        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            scores = [topic[i] for i in top_words_idx]

            # Get example reviews for this topic
            topic_strength = topic_matrix[:, topic_idx]
            top_review_idx = topic_strength.argsort()[-3:][::-1]
            examples = problem_reviews.iloc[top_review_idx]['review_text'].tolist()

            topics.append(top_words)
            word_scores.append(scores)
            example_reviews.append(examples)

        return topics, word_scores, example_reviews

    # Calculate issue severity
    def calculate_severity(topic_idx):
        topic_reviews = problem_reviews[topic_matrix[:, topic_idx] > min_topic_freq]
        severity = {
            'count': len(topic_reviews),
            'avg_sentiment': topic_reviews['sentiment'].mean(),
            'frequency': len(topic_reviews) / len(reviews_df)
        }
        return severity

    topics, word_scores, example_reviews = get_top_words_per_topic(nmf_model, feature_names)

    # Calculate severity for each topic
    topic_severity = [calculate_severity(i) for i in range(len(topics))]

    # Aggregate results
    analysis_results = {
        'total_reviews': len(reviews_df),
        'problem_reviews_count': len(problem_reviews),
        'topics': topics,
        'word_scores': word_scores,
        'example_reviews': example_reviews,
        'topic_severity': topic_severity
    }

    return analysis_results


def visualize_issues(analysis_results):
    """
    Create visualizations focused on app issues
    """
    if "error" in analysis_results:
        print(analysis_results["error"])
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Issue Topics and Their Severity
    num_topics = len(analysis_results['topics'])

    # Create topic labels with severity metrics
    topic_labels = []
    for i in range(num_topics):
        severity = analysis_results['topic_severity'][i]
        label = f"Issue {i+1}\n{severity['count']} reviews\n{severity['frequency']:.1%} of all reviews"
        topic_labels.append(label)

    # Plot top words for each issue topic
    plt.figure(figsize=(15, num_topics * 3))
    for idx, (topic_words, word_scores) in enumerate(zip(analysis_results['topics'], analysis_results['word_scores'])):
        plt.subplot(num_topics, 1, idx + 1)
        y_pos = np.arange(len(topic_words))
        plt.barh(y_pos, word_scores, align='center')
        plt.yticks(y_pos, topic_words)
        plt.xlabel('Term Importance')
        plt.title(f'Issue Topic {idx + 1} Key Terms\n' +
                  f"Example: {analysis_results['example_reviews'][idx][0][:100]}...")
    plt.tight_layout()
    plt.show()

    # 2. Issue Severity Overview
    plt.figure(figsize=(10, 6))
    severities = [s['count'] for s in analysis_results['topic_severity']]
    sentiments = [s['avg_sentiment'] for s in analysis_results['topic_severity']]

    # Create size-sentiment scatter plot
    plt.scatter(severities, sentiments, s=[s*100 for s in severities], alpha=0.6)
    for i, (x, y) in enumerate(zip(severities, sentiments)):
        plt.annotate(f'Issue {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Number of Complaints')
    plt.ylabel('Average Sentiment (more negative = more severe)')
    plt.title('Issue Severity Analysis')
    plt.grid(True)
    plt.show()

    # 3. Overall Issue Statistics
    plt.figure(figsize=(8, 4))
    plt.bar(['All Reviews', 'Problem Reviews'],
            [analysis_results['total_reviews'], analysis_results['problem_reviews_count']])
    plt.title('Proportion of Reviews Reporting Issues')
    plt.ylabel('Number of Reviews')
    plt.show()


# Example usage
"""
reviews_df = pd.DataFrame({
    'review_text': ['Your app keeps crashing when I try to upload photos',
                   'Great app but the search feature is slow',
                   'Love the new UI update!']
})

results = analyze_app_reviews(reviews_df)
visualize_results(results)
"""
reviews_df = pd.DataFrame({"review_text": reviews})
results = analyze_app_issues(reviews_df)
visualize_issues(results)
