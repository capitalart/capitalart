# Mockup Generator Code Snapshot

**Generated:** Tuesday, 13 May 2025, 11:14:34 PM ACST (+0930)

---

## 📄 FILE: `scripts/generate-5x4-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "5x4"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate-2x3-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "2x3"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/fix_srgb_profiles.py`
```
import os
from PIL import Image

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MOCKUPS_DIR = os.path.join(BASE_DIR, 'Input', 'Mockups')

def strip_broken_icc_profiles():
    print("🧼 Stripping broken sRGB profiles from PNGs...")
    for filename in os.listdir(MOCKUPS_DIR):
        if filename.lower().endswith('.png'):
            path = os.path.join(MOCKUPS_DIR, filename)
            try:
                img = Image.open(path)
                img = img.convert("RGBA")  # ensure consistent mode
                img.save(path, format="PNG", icc_profile=None)  # strip any broken profiles
                print(f"✅ Cleaned: {filename}")
            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")
    print("🏁 All done, no more sRGB warnings 🎉")

if __name__ == "__main__":
    strip_broken_icc_profiles()
```
---

## 📄 FILE: `scripts/generate-3x4-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "3x4"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate-16x9-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "16x9"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate-4x3-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "4x3"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/openai_vision_test.py`
```
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
```
---

## 📄 FILE: `scripts/test_env.py`
```
# test_env.py
import cv2
import pytz
from datetime import datetime

print("✅ OpenCV version:", cv2.__version__)
print("🕒 Adelaide time:", datetime.now(pytz.timezone("Australia/Adelaide")).strftime('%Y-%m-%d %I:%M %p %Z'))
```
---

## 📄 FILE: `scripts/generate-9x16-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "9x16"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate-1x1-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "1x1"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate-4x5-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "4x5"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate-3x2-composites.py`
```
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "3x2"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{index:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {coord_path}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{base_name}__{os.path.splitext(mockup_file)[0]}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")
```
---

## 📄 FILE: `scripts/generate_all_coordinates.py`
```
#!/usr/bin/env python3
# =============================================================================
# 🧠 Script: generate_all_coordinates.py
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts/
# 🎯 Purpose:
#     Scans all PNG mockup images inside Input/Mockups/[aspect-ratio] folders.
#     Detects transparent artwork zones and outputs a JSON file with 4 corner
#     coordinates into Input/Coordinates/[aspect-ratio] folders.
# ▶️ Run with:
#     python3 scripts/generate_all_coordinates.py
# =============================================================================

import os
import cv2
import json

# ----------------------------------------
# 📁 Folder Paths
# ----------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MOCKUP_DIR = os.path.join(BASE_DIR, 'Input', 'Mockups')
COORDINATE_DIR = os.path.join(BASE_DIR, 'Input', 'Coordinates')

# ----------------------------------------
# 🔧 Ensure output folders exist
# ----------------------------------------
def ensure_folder(path):
    """Ensure a folder exists; create if missing."""
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
# 📐 Corner Sorting
# ----------------------------------------
def sort_corners(pts):
    """
    Sorts 4 corner points to a consistent order:
    top-left, top-right, bottom-left, bottom-right
    """
    pts = sorted(pts, key=lambda p: (p["y"], p["x"]))  # Primary sort by Y, secondary by X
    top = sorted(pts[:2], key=lambda p: p["x"])
    bottom = sorted(pts[2:], key=lambda p: p["x"])
    return [*top, *bottom]

# ----------------------------------------
# 🔍 Transparent Region Detector
# ----------------------------------------
def detect_corner_points(image):
    """
    Detects 4 corner points of a transparent region in a PNG mockup.
    Returns a list of 4 dict points or None if detection fails.
    """
    if image is None or image.shape[2] != 4:
        return None

    # Use alpha channel to find transparent regions
    alpha = image[:, :, 3]
    _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    thresh_inv = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get largest contour and approximate shape
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        return None

    corners = [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in approx]
    return sort_corners(corners)

# ----------------------------------------
# 🚀 Coordinate Generation Runner
# ----------------------------------------
def generate_all_coordinates():
    """
    Loops through all subfolders in Input/Mockups/,
    detects artwork areas in .png files, and outputs JSON
    coordinate templates to Input/Coordinates/[aspect-ratio]/
    """
    print(f"\n📁 Scanning mockup source: {MOCKUP_DIR}\n")

    if not os.path.exists(MOCKUP_DIR):
        print(f"❌ Error: Mockup directory not found: {MOCKUP_DIR}")
        return

    for folder in sorted(os.listdir(MOCKUP_DIR)):
        mockup_folder = os.path.join(MOCKUP_DIR, folder)
        if not os.path.isdir(mockup_folder):
            continue

        print(f"🔍 Processing folder: {folder}")
        output_folder = os.path.join(COORDINATE_DIR, folder)
        ensure_folder(output_folder)

        for filename in os.listdir(mockup_folder):
            if not filename.lower().endswith('.png'):
                continue

            mockup_path = os.path.join(mockup_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.png', '.json'))

            try:
                image = cv2.imread(mockup_path, cv2.IMREAD_UNCHANGED)
                corners = detect_corner_points(image)

                if corners:
                    data = {
                        "template": filename,
                        "corners": corners
                    }
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"✅ Saved: {output_path}")
                else:
                    print(f"⚠️ Skipped (no valid corners): {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

    print("\n🏁 All coordinate templates generated.\n")

# ----------------------------------------
# 🔧 Script Entrypoint
# ----------------------------------------
if __name__ == "__main__":
    generate_all_coordinates()
```
---

## 📄 FILE: `scripts/generate_composites.py`
```
#!/usr/bin/env python3
# ============================================================================
# 🧠 Script Name: generate_composites.py
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts
# 🛠️ Purpose: Composite signed artwork into transparent mockup templates
# 🔗 Dependencies: OpenCV, NumPy, utils.py with `load_corner_data`, `perspective_transform`
# 🕒 Last Modified: May 9, 2025
# ▶️ Run with:
#     python3 scripts/generate_composites.py
# ============================================================================

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from itertools import chain
from utils import load_corner_data, perspective_transform

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("composite_log.txt"),
        logging.StreamHandler()
    ]
)

# === Directory Setup ===
BASE_DIR = Path(__file__).resolve().parent.parent
ARTWORKS_DIR = BASE_DIR / "Input" / "Artworks"
MOCKUPS_DIR = BASE_DIR / "Input" / "Mockups"
COORDINATES_DIR = BASE_DIR / "Input" / "Coordinates"
COMPOSITES_DIR = BASE_DIR / "Output" / "Composites"
DEBUG_DIR = BASE_DIR / "Output" / "Debug_Warped"

# === Ensure Output Folders Exist ===
COMPOSITES_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# === Recognized Image Extensions ===
valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# === Composite Counter ===
total_composites = 0

# ============================================================================
# 🔁 MAIN PROCESSING LOOP — by Aspect Ratio Folder
# ============================================================================
for aspect_folder in sorted(ARTWORKS_DIR.iterdir()):
    if not aspect_folder.is_dir():
        continue

    aspect_name = aspect_folder.name
    artwork_files = sorted(chain.from_iterable(aspect_folder.glob(ext) for ext in valid_extensions))

    if not artwork_files:
        logging.info(f"⏭️ No artwork found in: {aspect_name}")
        continue

    mockup_folder = MOCKUPS_DIR / aspect_name
    coordinate_folder = COORDINATES_DIR / aspect_name

    if not mockup_folder.exists() or not coordinate_folder.exists():
        logging.warning(f"⚠️ Missing folders for {aspect_name}: skipping")
        continue

    # === Load Corner Coordinates for Mockups ===
    try:
        corner_data = load_corner_data(coordinate_folder)
        # Normalize keys to lowercase to ensure flexible matching
        corner_data_normalized = {k.lower(): v for k, v in corner_data.items()}
    except Exception as e:
        logging.exception(f"❌ Failed to load coordinate data for {aspect_name}: {e}")
        continue

    mockup_files = sorted(mockup_folder.glob("*[Mm]ockup-*.png"))
    if not mockup_files:
        logging.warning(f"⚠️ No mockups found in: {mockup_folder}")
        continue

    logging.info(f"📐 Processing aspect ratio: {aspect_name}")
    logging.info(f"   🎨 Artworks: {len(artwork_files)} | 🖼️ Mockups: {len(mockup_files)}")

    for artwork_path in artwork_files:
        artwork_name = artwork_path.stem[:70]
        logging.info(f"\n🎨 Artwork: {artwork_name}")

        output_subfolder = COMPOSITES_DIR / artwork_name
        output_subfolder.mkdir(parents=True, exist_ok=True)

        artwork_img = cv2.imread(str(artwork_path), cv2.IMREAD_UNCHANGED)
        if artwork_img is None:
            logging.error(f"❌ Could not read artwork: {artwork_path.name}")
            continue

        for mockup_path in mockup_files:
            mockup_filename = mockup_path.name
            lookup_key = mockup_filename.lower()

            if lookup_key not in corner_data_normalized:
                logging.warning(f"⚠️ Missing coordinate data for: {mockup_filename}")
                continue

            corners = corner_data_normalized[lookup_key]
            if not isinstance(corners, list) or len(corners) != 4:
                logging.error(f"❌ Invalid corner data for {mockup_filename}")
                continue

            mockup_img = cv2.imread(str(mockup_path), cv2.IMREAD_UNCHANGED)
            if mockup_img is None:
                logging.error(f"❌ Failed to read mockup: {mockup_filename}")
                continue

            try:
                warped = perspective_transform(artwork_img, corners, mockup_img.shape[:2])
            except Exception as e:
                logging.exception(f"❌ Warp failed for {artwork_path.name} -> {mockup_filename}: {e}")
                continue

            try:
                if mockup_img.shape[2] == 4:
                    alpha_mask = mockup_img[:, :, 3:] / 255.0
                    composite_rgb = (warped[:, :, :3] * (1 - alpha_mask) +
                                     mockup_img[:, :, :3] * alpha_mask).astype(np.uint8)
                else:
                    composite_rgb = cv2.addWeighted(warped[:, :, :3], 1, mockup_img[:, :, :3], 1, 0)

                mockup_number = mockup_path.stem.replace("mockup-layer-", "").zfill(2)
                output_file = output_subfolder / f"{artwork_name}-mockup-{mockup_number}.jpg"

                success = cv2.imwrite(str(output_file), composite_rgb)
                if success:
                    logging.info(f"   💾 Saved: {output_file}")
                    total_composites += 1
                else:
                    logging.error(f"❌ Failed to save: {output_file}")

            except Exception as e:
                logging.exception(f"❌ Composite or save error: {e}")

# ============================================================================
# ✅ SUMMARY
# ============================================================================
logging.info(f"\n✅ Done! Total composites generated: {total_composites}")
```
---

## 📄 FILE: `scripts/utils.py`
```
#!/usr/bin/env python3
# =============================================================================
# 🧠 Module: utils.py
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts
# ▶️ Imported by:
#     generate_composites.py
# 🎯 Purpose:
#     1. Load JSON coordinate data for mockups.
#     2. Apply a perspective warp to map artwork into mockup templates.
# 🔗 Dependencies: OpenCV, NumPy, JSON, pathlib
# 🕒 Last Updated: May 9, 2025
# =============================================================================

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_corner_data(templates_dir: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Loads all JSON files from the given coordinates folder.

    Args:
        templates_dir (str): Path to the folder containing JSON corner files.

    Returns:
        dict: A mapping from lowercase template filename (e.g., 'mockup-01.png')
              to its 4-corner data for warping.
    """
    corner_data = {}
    templates_path = Path(templates_dir)

    for json_file in templates_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                template_name = data.get("template", "").lower().strip()
                corners = data.get("corners")

                if template_name and corners and len(corners) == 4:
                    corner_data[template_name] = corners
        except Exception as e:
            print(f"❌ Failed to read {json_file.name}: {e}")

    return corner_data


def perspective_transform(
    artwork_img: np.ndarray,
    dst_points: List[Dict[str, int]],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Applies a 4-point perspective transformation to map an artwork into a mockup zone.

    Args:
        artwork_img (np.ndarray): The original artwork image.
        dst_points (list): List of 4 destination corner dicts (with "x" and "y").
        output_shape (tuple): Shape (height, width) of the mockup image.

    Returns:
        np.ndarray: The warped artwork image sized to fit inside the mockup.
    """
    h, w = artwork_img.shape[:2]

    # Source points are the corners of the artwork image
    src_points = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype="float32"
    )

    # Destination points are the mockup's 4 corners
    dst_points_array = np.array(
        [[pt["x"], pt["y"]] for pt in dst_points], dtype="float32"
    )

    # Create the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points_array)

    # Warp the artwork using the matrix and match mockup's resolution
    warped = cv2.warpPerspective(
        artwork_img,
        matrix,
        (output_shape[1], output_shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return warped
```
---

## 📄 FILE: `scripts/google_vision_basic_analyzer.py`
```
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
```
---

## 📄 FILE: `scripts/generate_folder_structure.py`
```
#!/usr/bin/env python3
# =========================================================
# 🧠 Script: generate_folder_structure.py
# 📍 Local Mockup Generator Tool
# 📅 Timestamped with Australia/Adelaide timezone
# ▶️ Run with:
#     python3 scripts/generate_folder_structure.py
# =========================================================

import os
from datetime import datetime
from pathlib import Path
import pytz

BASE_DIR = Path("/Users/robin/Documents/01-ezygallery-MockupWorkShop")
IGNORE_FOLDERS = {"venv", "__pycache__", ".git", ".idea", ".vscode", "node_modules"}
IGNORE_FILES = {".DS_Store", "Thumbs.db"}
IGNORE_EXTS = {".pyc", ".log", ".tmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".psd"}

timestamp = datetime.now(pytz.timezone("Australia/Adelaide")).strftime("%a-%d-%b-%Y_%I-%M%p")
output_file = BASE_DIR / f"folder-structure-no-images_{timestamp}.txt"

def generate_tree(path, prefix=""):
    lines = []
    entries = sorted(os.listdir(path))
    for idx, name in enumerate(entries):
        full_path = path / name
        connector = "└── " if idx == len(entries) - 1 else "├── "
        if full_path.is_dir() and name not in IGNORE_FOLDERS:
            lines.append(f"{prefix}{connector}{name}/")
            sub_prefix = "    " if idx == len(entries) - 1 else "│   "
            lines.extend(generate_tree(full_path, prefix + sub_prefix))
        elif full_path.is_file():
            if name in IGNORE_FILES or full_path.suffix.lower() in IGNORE_EXTS:
                continue
            lines.append(f"{prefix}{connector}{name}")
    return lines

if __name__ == "__main__":
    print("📂 Generating folder structure (excluding image files)...")
    lines = [f"{BASE_DIR.name}/"] + generate_tree(BASE_DIR)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Folder structure saved to: {output_file}")```
---

## 📄 FILE: `scripts/write_all_composite_generators.py`
```
aspect_ratios = [
    "1x1", "2x3", "3x2", "3x4",
    "4x3", "4x5", "5x4", "9x16", "16x9"
]

template = '''import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "{aspect}"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{{aspect_ratio}}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{{aspect_ratio}}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{{aspect_ratio}}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{{aspect_ratio}}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    for index, artwork_file in enumerate(artwork_files, start=1):
        base_name, ext = os.path.splitext(artwork_file)
        set_name = f"set-{{index:02d}}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, base_name + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, base_name + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {{mockup_file}}")
                continue

            with open(coord_path, "r") as f:
                coords_data = json.load(f)

            if "corners" not in coords_data:
                print(f"⚠️ Invalid or missing 'corners' in {{coord_path}}")
                continue

            raw_corners = coords_data["corners"]
            dst_coords = [
                [raw_corners[0]["x"], raw_corners[0]["y"]],
                [raw_corners[1]["x"], raw_corners[1]["y"]],
                [raw_corners[3]["x"], raw_corners[3]["y"]],
                [raw_corners[2]["x"], raw_corners[2]["y"]]
            ]

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            output_filename = f"{{base_name}}__{{os.path.splitext(mockup_file)[0]}}__composite.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {{output_filename}}")

    print(f"\\n🎯 All {{aspect_ratio}} composite sets completed.")
'''

for aspect in aspect_ratios:
    filename = f"generate-{aspect}-composites.py"
    with open(filename, "w") as f:
        f.write(template.format(aspect=aspect))
    print(f"✅ Created script: {filename}")
```
---

## 📄 FILE: `scripts/backup_mockup_structure.sh`
```
#!/bin/bash
# ======================================================
# 🧠 Local Backup Script: Mockup Generator (Structure Only)
# 📍 Saves project structure excluding large image files
# 🕒 Timestamp set using Australia/Adelaide timezone
# ▶️ Run with:
#     bash scripts/backup_mockup_structure.sh
# ======================================================

set -e

echo "🔄 Starting Mockup Generator STRUCTURE-ONLY backup..."
BACKUP_BASE_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop/backups"
PROJECT_SOURCE_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop"
LOG_DIR="$BACKUP_BASE_DIR/logs"

mkdir -p "$BACKUP_BASE_DIR"
mkdir -p "$LOG_DIR"

# Set time/date stamp (Adelaide time)
TIMESTAMP=$(TZ="Australia/Adelaide" date +"%a-%d-%b-%Y_%I-%M%p") # e.g. Wed-08-May-2025_06-24PM

BACKUP_FILENAME="mockup_structure_backup_${TIMESTAMP}.tar.gz"
BACKUP_PATH="$BACKUP_BASE_DIR/$BACKUP_FILENAME"
LOG_FILE="$LOG_DIR/backup_log.txt"

echo "----------------------------------------------------" | tee -a "$LOG_FILE"
echo "Backup started at $TIMESTAMP" | tee -a "$LOG_FILE"

cd /Users/robin/Documents || exit 1

tar -czf "$BACKUP_PATH" \
    --exclude=backups \
    --exclude=*.jpg \
    --exclude=*.jpeg \
    --exclude=*.png \
    --exclude=*.webp \
    --exclude=*.tif \
    --exclude=*.tiff \
    --exclude=*.psd \
    --exclude=venv \
    --exclude=__pycache__ \
    --exclude=.DS_Store \
    --exclude=.Spotlight-V100 \
    --exclude=.TemporaryItems \
    --exclude=.Trashes \
    --exclude=.DocumentRevisions-V100 \
    --exclude=.fseventsd \
    --exclude=.VolumeIcon.icns \
    --exclude=.AppleDouble \
    --exclude=.apdisk \
    "01-ezygallery-MockupWorkShop"

echo "✅ STRUCTURE backup created at $BACKUP_PATH" | tee -a "$LOG_FILE"
echo "----------------------------------------------------"```
---

## 📄 FILE: `scripts/backup_mockup_generator.sh`
```
#!/bin/bash
# ======================================================
# 🧠 Local Backup Script: Mockup Generator (Mac Dev)
# 📍 Saves project archive excluding venv/macOS metadata
# 🕒 Timestamp set using Australia/Adelaide timezone
# ▶️ Run with:
#     bash backup_mockup_generator.sh
# ======================================================

set -e

echo "🔄 Starting Mockup Generator backup..."
BACKUP_BASE_DIR="/Users/robin/mockup-generator-backups"
PROJECT_SOURCE_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop"
LOG_DIR="/Users/robin/mockup-generator-backups/logs"

mkdir -p "$BACKUP_BASE_DIR"
mkdir -p "$LOG_DIR"

# Set time/date stamp (Adelaide time)
TIMESTAMP=$(TZ="Australia/Adelaide" date +"%Y-%m-%d_%H%M%S_%Z") # More sortable timestamp
READABLE_TIMESTAMP=$(TZ="Australia/Adelaide" date +"%a-%d-%b-%Y_%I-%M%p")
BACKUP_PATH="$BACKUP_BASE_DIR/$BACKUP_FILENAME"
LOG_FILE="$LOG_DIR/backup_log.txt"

echo "----------------------------------------------------" | tee -a "$LOG_FILE"
echo "Backup started at $(TZ="Australia/Adelaide" date)" | tee -a "$LOG_FILE"

cd /Users/robin/Documents || exit 1

tar -czf "$BACKUP_PATH" \
    --exclude=venv \
    --exclude=__pycache__ \
    --exclude=.DS_Store \
    --exclude=.Spotlight-V100 \
    --exclude=.TemporaryItems \
    --exclude=.Trashes \
    --exclude=.DocumentRevisions-V100 \
    --exclude=.fseventsd \
    --exclude=.VolumeIcon.icns \
    --exclude=.AppleDouble \
    --exclude=.apdisk \
    "01-ezygallery-MockupWorkShop"

echo "✅ Backup created at $BACKUP_PATH" | tee -a "$LOG_FILE"
echo "----------------------------------------------------"```
---

## 📄 FILE: `scripts/gather_mockup_code_to_text.sh`
```
#!/bin/bash
# =========================================================================
# 🧠 Script Name: gather_mockup_code_to_text.sh
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts
# ▶️ Run with:
#     bash scripts/gather_mockup_code_to_text.sh
# =========================================================================

# --- Configuration ---
SNAPSHOT_BASENAME="mockup_code_snapshot"
OUTPUT_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop/backups"
ADELAIDE_TIMEZONE="Australia/Adelaide"

FILENAME_TIMESTAMP=$(TZ="$ADELAIDE_TIMEZONE" date +"%a-%d-%b-%Y_%I.%M%p_%Z")
OUTPUT_FILE="${OUTPUT_DIR}/${SNAPSHOT_BASENAME}_${FILENAME_TIMESTAMP}.md"
GENERATED_TIMESTAMP=$(TZ="$ADELAIDE_TIMEZONE" date +"%A, %d %B %Y, %I:%M:%S %p %Z (%z)")

INCLUDE_EXTENSIONS=("*.py" "*.sh" "*.txt" "*.md" "*.json")
EXCLUDE_PATHS=(
  "*/venv/*"
  "*/__pycache__/*"
  "*/.git/*"
  "*/.vscode/*"
  "*/.DS_Store"
  "*/backups/*"
  "*/__MACOSX/*"
  "*/Artworks/*"
  "*/Upscaled-Art/*"
  "*/Optimised-Art/*"
  "*/Signed-Art/*"
)

PROJECT_DIRECTORIES=(
  "/Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts"
  "/Users/robin/Documents/01-ezygallery-MockupWorkShop/Coordinates"
  "/Users/robin/Documents/01-ezygallery-MockupWorkShop/Output"
)

echo "📦 Gathering mockup generator source files into Markdown snapshot..."
mkdir -p "$OUTPUT_DIR"
echo -e "# Mockup Generator Code Snapshot\n\n**Generated:** ${GENERATED_TIMESTAMP}\n\n---" > "$OUTPUT_FILE"

# Process and append all files
for base_path in "${PROJECT_DIRECTORIES[@]}"; do
  if [ -d "$base_path" ]; then
    for ext in "${INCLUDE_EXTENSIONS[@]}"; do
      while IFS= read -r file; do
        # Check against excluded paths
        skip=false
        for exclude in "${EXCLUDE_PATHS[@]}"; do
          if [[ "$file" == $exclude ]]; then
            skip=true; break
          fi
        done
        if [ "$skip" = false ]; then
          relative_path="${file#/Users/robin/Documents/01-ezygallery-MockupWorkShop/}"
          {
            echo ""
            echo "## 📄 FILE: \`$relative_path\`"
            echo '```'
            cat "$file"
            echo '```'
            echo "---"
          } >> "$OUTPUT_FILE"
        fi
      done < <(find "$base_path" -type f -name "$ext" 2>/dev/null)
    done
  fi
done

echo "✅ Markdown snapshot saved to: $OUTPUT_FILE"
```
---
