import json

from review_class import Review


file_path = "positive10k.txt"
output_file = "reviews.json"

with open(file_path, "r", encoding="utf-8") as file:
    # Convert iterable to a list of serializable dictionaries
    positive_data = [
        Review(
            review.strip(),
            "",
            ""
        ).to_dict(format_date=False)
        for review in file
        if review != "" and len(review) > 7
    ]

file_path = "negative10k.txt"

with open(file_path, "r", encoding="utf-8") as file:
    # Convert iterable to a list of serializable dictionaries
    negative_data = [
        Review(
            review.strip(),
            "",
            ""
        ).to_dict(format_date=False)
        for review in file
        if review != "" and len(review) > 7
    ]

serializable_data = []
serializable_data.extend(positive_data)
serializable_data.extend(negative_data)
print(f"Processing {len(serializable_data)} items")

try:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
