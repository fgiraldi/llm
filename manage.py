import click
import json
from datetime import date
from chromadb_class import ReviewVectorDB


CURRENT_DATE_ISO = date.today().isoformat()
CURRENT_DATE_YEAR = CURRENT_DATE_ISO[0:4]
CURRENT_DATE_MONTH = CURRENT_DATE_ISO[5:7]
print(CURRENT_DATE_ISO)
print(CURRENT_DATE_YEAR)
print(CURRENT_DATE_MONTH)

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
@click.option("--query", default="Very good app", help="Reviews text to look for similar ones")
def query_similar(query):
    vector_db = ReviewVectorDB(persist_directory="./chroma_db")

    # Example query
    results = vector_db.query_similar(
        query,
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


@cli.command()
@click.option("--year", default=CURRENT_DATE_YEAR, help="Year")
@click.option("--month", default=CURRENT_DATE_MONTH, help="Month")
def query_by_month(year, month):
    vector_db = ReviewVectorDB(persist_directory="./chroma_db")

    results = vector_db.get_reviews_by_month(
        year=year,
        month=month
    )

    # Print results
    print(f"\nDate range query results year {year} month {month}:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\nMatch {i+1}")
        print(f"Distance: {distance}")
        print(f"Review: {doc}")
        print(f"Date: {metadata['date']}")
        print(f"Rating: {metadata['rating']}")


if __name__ == "__main__":
    cli()
