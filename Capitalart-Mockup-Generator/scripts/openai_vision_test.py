import os
from dotenv import load_dotenv
from openai import OpenAI

# === FULL SETUP ===

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "../JSON-Files/.env")
load_dotenv(dotenv_path)

# Pull the API key properly
api_key = os.getenv("OPENAI_API_KEY")

# Debug: Show if API Key is loaded (optional)
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not loaded from .env! Check your path and file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Set your uploaded image URL (must be publicly accessible)
test_image_url = "https://ezygallery.com/artworks/test-image-01.jpg"

# Call OpenAI 4o Vision
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please carefully describe this artwork, focusing on visual features, composition, and emotional impact."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": test_image_url
                    }
                }
            ]
        }
    ],
    max_tokens=800
)

# Output result
print("\n🖼️ Image Analysis Result:\n")
print(response.choices[0].message.content)
