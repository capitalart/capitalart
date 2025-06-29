import os
import json
from google.cloud import vision

# === SETUP ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "JSON-Files/vision-ai-service-account.json"

# === PATHS ===
input_folder = "Input/Artworks/4x5"
output_folder = "Output/Analysis/4x5"
os.makedirs(output_folder, exist_ok=True)

# === VISION CLIENT ===
client = vision.ImageAnnotatorClient()

# === PROCESS IMAGES ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.replace(".jpg", "_analysis.json").replace(".jpeg", "_analysis.json").replace(".png", "_analysis.json"))

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    labels_data = []
    for label in response.label_annotations:
        labels_data.append({
            "description": label.description,
            "score": round(label.score, 2)
        })

    with open(output_path, "w") as f:
        json.dump(labels_data, f, indent=2)

    print(f"✅ Saved analysis for {filename}")

print("\n🎯 All artworks analyzed and saved!")
