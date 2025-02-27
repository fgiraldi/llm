import config
from openai import OpenAI

client = OpenAI(api_key=config.api_key)


prompt = """
Generate 10 user reviews for a fictional e-commerce product:
1. Include positive, neutral, and negative reviews.
2. Vary the tone, style, and length (short, medium, long).
3. Use realistic language as if written by actual users.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1500,
    temperature=0.8)

print(response.choices[0].message.content)
