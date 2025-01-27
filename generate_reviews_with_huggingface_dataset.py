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

    def to_dict(self):
        """Convert the object to a serializable dictionary."""
        return {
            "category": self.category,
            "review": self.review,
            "date": self.date.isoformat() if isinstance(self.date, datetime) else self.date,
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
