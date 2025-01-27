import click
import json
from chromadb_class import ReviewVectorDB


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
def query_similar():
    vector_db = ReviewVectorDB(persist_directory="./chroma_db")

    # Example query
    results = vector_db.query_similar(
        "The app keeps crashing when I try to save my progress",
        top_k=3
    )

    # Print results
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\nMatch {i+1}")
        print(f"Distance: {distance}")
        print(f"Review: {doc}")
        print(f"Metadata: {metadata}")


if __name__ == "__main__":
    cli()
