import random
from datetime import datetime


# Example custom class
class Review:
    _categories = ['user experience', 'response time', 'help desk', 'accessibility', 'recommendations', 'integrations']

    def __init__(self, review, date, star):
        self.category = random.choice(self._categories)
        self.review = review
        self.date = date  # Assume this is a datetime object
        self.star = star

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

    def to_dict(self):
        """Convert the object to a serializable dictionary."""
        return {
            "category": self.category,
            "review": self.review,
            "date": self._parse_date(self.date),
            "star": self.star
        }
