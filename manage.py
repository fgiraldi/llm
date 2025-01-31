import click
import json
from datetime import date

from chromadb_class import ReviewVectorDB
from data_clustering_raw_data import analyze_review_trends, ReviewChromaDoc
from data_visualization import (
    create_trend_visualizations, save_visualizations, print_cluster_summary, visualize_cluster_sizes
)

CURRENT_DATE_ISO = date.today().isoformat()
CURRENT_DATE_YEAR = CURRENT_DATE_ISO[0:4]
CURRENT_DATE_MONTH = CURRENT_DATE_ISO[5:7]


# Create a Click group to group commands
@click.group()
def cli():
    """Management script for various commands."""
    pass


@cli.command()
def process_reviews_into_vector_db():
    # Initialize the vector database
    vector_db = ReviewVectorDB(persist_directory="./chroma_db")

    # Load reviews from JSON file
    with open('reviews.json', 'r') as f:
        reviews = json.load(f)

    # Insert reviews
    vector_db.upsert_reviews(reviews)

    # Print collection stats
    print("Collection stats:", vector_db.get_stats())


@cli.command()
def detect_clusters():
    vector_db = ReviewVectorDB(persist_directory="./chroma_db")

    # Filter docs by one category
    chroma_docs = vector_db.collection.get(
        where={
            "$or": [
                {"category": "user experience"},
                {"category": "response time"},
                {"category": "help desk"},
                {"category": "integrations"},
            ]
        },
        include=['metadatas', 'documents']
    )
    # Create a list of usable data to analize
    chroma_docs = [
        ReviewChromaDoc(text, metadata)
        for text, metadata in zip(chroma_docs['documents'], chroma_docs['metadatas'])
    ]
    print(f"Processing {len(chroma_docs)} docs...")
    results = analyze_review_trends(chroma_docs)
    # visualization = visualize_trends(results)

    # Example: Print top terms for each cluster
    for cluster, terms in results['cluster_terms'].items():
        print(f"\n{cluster}:")
        print(", ".join(terms))

    # Create and display visualizations
    figures = create_trend_visualizations(results)

    # Save them as interactive HTML files:
    save_visualizations(figures)

    print_cluster_summary(results)
    visualize_cluster_sizes(results)


@cli.command()
@click.option("--query", default="Very good app", help="Reviews text to look for similar ones")
def query_similar(query):
    vector_db = ReviewVectorDB(persist_directory="./chroma_db")

    # Example query
    results = vector_db.query_similar(
        query,
        top_k=3
    )
    print(results)

    # Print results
    for i, (doc, metadata, distance, embedding) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
        results['embeddings'][0]
    )):
        print(f"\nMatch {i+1}")
        print(f"Distance: {distance}")
        print(f"Review: {doc}")
        print(f"Metadata: {metadata}")
        print(f"Embeddings: {embedding}")


# @cli.command()
# @click.option("--year", default=CURRENT_DATE_YEAR, help="Year")
# @click.option("--month", default=CURRENT_DATE_MONTH, help="Month")
# def query_by_month(year, month):
#     vector_db = ReviewVectorDB(persist_directory="./chroma_db")

#     results = vector_db.get_reviews_by_month(
#         year=year,
#         month=month
#     )

#     print(results)

#     # Print results
#     print(f"\nDate range query results year {year} month {month}:")
#     for i, (doc, metadata, distance) in enumerate(zip(
#         results['documents'][0],
#         results['metadatas'][0],
#         results['distances'][0]
#     )):
#         print(f"\nMatch {i+1}")
#         print(f"Distance: {distance}")
#         print(f"Review: {doc}")
#         print(f"Date: {metadata['date']}")
#         print(f"Year: {metadata['year']}")
#         print(f"Month: {metadata['month']}")
#         print(f"Rating: {metadata['rating']}")


if __name__ == "__main__":
    cli()
