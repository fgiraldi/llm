import csv
import json

from review_class import Review


file_path = "dummy_data_v3.csv"
output_file = "reviews.json"

with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    # Convert iterable to a list of serializable dictionaries
    serializable_data = [
        Review(
            review["Why?"],
            review["Response date"],
            ""
        ).to_dict(format_date=False)
        for review in reader
        if review["Why?"] != ""
    ]

print(f"Processing {len(serializable_data)} items")

try:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
