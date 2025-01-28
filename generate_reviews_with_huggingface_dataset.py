import json
import random
from datetime import datetime

from datasets import load_dataset

categories = ['user experience', 'response time', 'help desk', 'accessibility', 'recommendations', 'integrations']


# Example custom class
class Review:
    def __init__(self, review, date, star):
        self.category = random.choice(categories)
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


dataset = load_dataset("Sharathhebbar24/app_reviews_modded")
print(len(dataset["test"]))
output_file = "reviews.json"
# Convert iterable to a list of serializable dictionaries
serializable_data = [
    Review(
        review["review"],
        review["date"],
        review["star"]
    ).to_dict()
    for review in dataset["test"]
]

try:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(serializable_data[0:10], file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
