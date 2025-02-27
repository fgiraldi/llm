from datetime import datetime
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import uuid


class ReviewVectorDB:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the Chroma vector database.

        Args:
            persist_directory: Directory where Chroma will store its data
        """
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="app_reviews",
            embedding_function=self.embedding_function,
            metadata={"description": "App user reviews"}
        )

    def _parse_date(self, date_str: str) -> str:
        """
        Convert date string to ISO format string.

        Args:
            date_str: Date in format "Month Day Year" (e.g., "January 15 2024")

        Returns:
            ISO formatted date string (YYYY-MM-DD)
        """
        try:
            # Parse the date string
            date_obj = datetime.strptime(date_str, "%B %d %Y")
            # Convert to ISO format string
            return date_obj.strftime("%Y-%m-%d")
        except ValueError as e:
            print(f"Warning: Could not parse date '{date_str}': {e}")
            return None

    def process_reviews(self, reviews: List[Dict]) -> tuple[List[str], List[str], List[Dict]]:
        """
        Process reviews and prepare them for insertion.

        Args:
            reviews: List of review dictionaries

        Returns:
            Tuple of (ids, texts, metadata) for insertion into Chroma
        """
        ids = []
        texts = []
        metadatas = []

        for review in tqdm(reviews, desc="Processing reviews"):
            # Generate a unique ID
            review_id = str(uuid.uuid4())

            # Extract text content (adjust key names based on your JSON structure)
            text = review.get('review', '')
            # Parse the date string to ISO format
            date_str = review.get('date')
            iso_date = date_str if date_str else '2025-01-01'

            # Create metadata
            metadata = {
                'rating': str(review.get('star', '')),  # Chroma requires metadata values to be strings
                'date': iso_date,  # Store as ISO format string
                'year': str(datetime.strptime(iso_date, "%Y-%m-%d").year) if iso_date else None,
                'month': str(datetime.strptime(iso_date, "%Y-%m-%d").month) if iso_date else None,
                'category': review.get('category', ''),
                # Add any other metadata fields you want to store
            }

            ids.append(review_id)
            texts.append(text)
            metadatas.append(metadata)

        return ids, texts, metadatas

    def upsert_reviews(self, reviews: List[Dict], batch_size: int = 100):
        """
        Insert reviews into Chroma in batches.

        Args:
            reviews: List of review dictionaries
            batch_size: Number of reviews to insert at once
        """
        # Process all reviews first
        ids, texts, metadatas = self.process_reviews(reviews)

        # Insert in batches
        for i in tqdm(range(0, len(ids), batch_size), desc="Uploading to Chroma"):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadatas[i:i + batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadata
            )

    def query_similar(self, query_text: str, top_k: int = 5) -> Dict:
        """
        Query the vector database for similar reviews.

        Args:
            query_text: Text to find similar reviews for
            top_k: Number of similar reviews to return

        Returns:
            Dictionary containing similar reviews with scores
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )

        return results

    def get_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary containing collection statistics
        """
        return {
            "total_reviews": self.collection.count(),
            "name": self.collection.name,
            "metadata": self.collection.metadata
        }

    def get_reviews_by_month(self, year: str, month: str) -> Dict:
        """
        Get all reviews for a specific month and year.

        Args:
            year: Year as string YYYY
            month: Month as string ('01'-'12')

        Returns:
            Dictionary containing reviews for the specified month
        """
        where = {
            "$and": [
                {"year": year},
                {"month": month}
            ]
        }

        return self.collection.query(
            query_texts=[""],  # Empty query to get all matching documents
            where=where,
            include=['documents', 'metadatas', 'distances'],
            n_results=1000  # Adjust based on your needs
        )
