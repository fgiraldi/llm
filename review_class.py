import random
from datetime import datetime
from typing import List


# Example custom class
class Review:
    _categories = ['user experience', 'response time', 'help desk', 'accessibility', 'recommendations', 'integrations']
    embedding: List[float] = None

    def __init__(self, review, date, category, embedding=None):
        self.category = random.choice(self._categories)
        self.review = review
        self.category = category
        self.date = date  # Assume this is a datetime object
        self.embedding = embedding if embedding else []

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

    def to_dict(self, format_date: bool = True):
        """Convert the object to a serializable dictionary."""
        return {
            "category": self.category,
            "review": self.review,
            "embedding": self.embedding if self.embedding else []
        }
