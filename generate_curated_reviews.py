import config
from datetime import datetime
import json
import pandas as pd
import re
from langchain_openai import OpenAIEmbeddings

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from review_class import Review


file_path = "dashboard_export_2025_01_14.csv"
output_file = "reviews.json"
api_key = config.api_key
openai_embedding_fn = OpenAIEmbeddings(openai_api_key=api_key)


def parse_date(date_str: str) -> str:
    """
    Convert date string to ISO format string.

    Args:
        date_str: Date in format "Month Day Year Hour Min" (e.g., "1/14/2025 8:08")

    Returns:
        ISO formatted date string (YYYY-MM-DD)
    """
    try:
        # Parse the date string
        date_obj = datetime.strptime(date_str, "%m/%d/%Y %H:%M")
        # Convert to ISO format string
        return date_obj.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return None


def is_review_negative(vader_label: str, cardiff_label: str, nlptown_label: str) -> bool:
    """
    Determine if a review is negative based on the sentiment labels.

    Args:
        vader_label: Sentiment label from VADER
        cardiff_label: Sentiment label from Cardiff NLP
        nlptown_label: Sentiment label from NLPTown

    Returns:
        True if the review is negative for at least one transformer, False otherwise
    """

    return any(label == "negative" for label in [vader_label, cardiff_label, nlptown_label])


data = pd.read_csv(file_path)
# Drop rows where 'Review' field is empty or has less than 7 characters
data = data.dropna(subset=['Review'])
data = data[data['Review'].str.len() > 7]
# Filter rows that do not match 'Very easy'
data = data[~data['Ease'].str.match('Very easy')]
data['review_clean_text'] = data['Review'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
print(f"Processing {len(data)} items")

vader_sentiment = SentimentIntensityAnalyzer()
data['vader_score'] = data['review_clean_text'].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])
# Create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']
data['vader_label'] = pd.cut(data['vader_score'], bins, labels=names)

# Calculate sentiment labels using transformer models
transformer_pipeline_cardiff = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
transformer_pipeline_nlptown = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
data['cardiff_label'] = data['review_clean_text'].apply(lambda x: transformer_pipeline_cardiff(x)[0]['label'])
data['nlptown_label'] = data['review_clean_text'].apply(lambda x: 'negative' if transformer_pipeline_nlptown(x)[0]['label'] in ['1 star', '2 stars'] else 'positive')

data['is_negative'] = data.apply(lambda x: is_review_negative(x['vader_label'], x['cardiff_label'], x['nlptown_label']), axis=1)
print(data[['Review', 'vader_label', 'cardiff_label', 'nlptown_label', 'is_negative']].head(25))
print(f"Number of negative reviews: {data['is_negative'].sum()}")

# Keep only negative reviews
data = data[data['is_negative']]
print(f"Number of negative reviews: {len(data)}")

# Export data to csv
data.to_csv("negative_reviews.csv", index=False)

# Convert DataFrame to list of serializable dictionaries
serializable_data = [
    Review(
        review["Review"],
        parse_date(review["Response Date"]),
        review["Category"]
    ).to_dict(format_date=False)
    for index, review in data.iterrows()
]

embeddings = openai_embedding_fn.embed_documents([review["review"] for review in serializable_data])
serializable_data = list(map(lambda x, y: {**x, "embedding": y}, serializable_data, embeddings))

try:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
