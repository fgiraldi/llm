import json
from datetime import datetime

from datasets import load_dataset


# Example custom class
class Review:
    def __init__(self, package_name, review, date, star, products):
        self.package_name = package_name
        self.review = review
        self.date = date  # Assume this is a datetime object
        self.star = star
        self.products = products

    def to_dict(self):
        """Convert the object to a serializable dictionary."""
        return {
            "package_name": self.package_name,
            "review": self.review,
            "date": self.date.isoformat() if isinstance(self.date, datetime) else self.date,
            "star": self.star,
            "products": self.products,
        }


dataset = load_dataset("Sharathhebbar24/app_reviews_modded")
print(len(dataset["test"]))
output_file = "reviews.json"
# Convert iterable to a list of serializable dictionaries
serializable_data = [
    Review(
        review["package_name"],
        review["review"],
        review["date"],
        review["star"],
        review["products"]
    ).to_dict()
    for review in dataset["test"]
]

try:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(serializable_data[0:1000], file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
