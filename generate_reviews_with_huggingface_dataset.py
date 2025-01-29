import json
from datasets import load_dataset

from review_class import Review


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
        json.dump(serializable_data[0:10000], file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
