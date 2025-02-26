import config
import csv
from datetime import datetime
import json
from langchain_openai import OpenAIEmbeddings

from review_class import Review


file_path = "dashboard_export_2025_01_14.csv"
output_file = "reviews.json"
api_key = config.api_key
openai_embedding_fn = OpenAIEmbeddings(openai_api_key=api_key)


def parse_date(date_str: str) -> str:
    """
    Convert date string to ISO format string.

    Args:
        date_str: Date in format "Month Day Year Hour Min" (e.g., "1/14/2025 8:08")

    Returns:
        ISO formatted date string (YYYY-MM-DD)
    """
    try:
        # Parse the date string
        date_obj = datetime.strptime(date_str, "%m/%d/%Y %H:%M")
        # Convert to ISO format string
        return date_obj.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return None


with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    # Convert iterable to a list of serializable dictionaries
    serializable_data = [
        Review(
            review["Review"],
            parse_date(review["Response Date"]),
            "",
            review["Ease"]
        ).to_dict(format_date=False)
        for review in reader
        if review["Review"] != "" and len(review["Review"]) > 7
    ]

print(f"Processing {len(serializable_data)} items")

embeddings = openai_embedding_fn.embed_documents([review["review"] for review in serializable_data])
serializable_data = list(map(lambda x, y: {**x, "embedding": y}, serializable_data, embeddings))

try:
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
