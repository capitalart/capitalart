# 🧠 CapitalArt Snapshot — REPORTS-26-JUN-2025-02-47PM


---
## 📄 mockup_selector_ui.py

```py
# ============================== [ mockup_selector_ui.py ] ==============================
# Mockup Selection Interface for CapitalArt
# --------------------------------------------------------------------------------------
# Flask-based UI that:
# - Dynamically reads mockup categories from 4x5-categorised folder
# - Randomly selects one mockup from each category
# - Allows per-slot regenerate or swap category
# - Displays all choices in a visual grid for user approval
# ======================================================================================

import os
import random
from pathlib import Path
from typing import List, Dict

from flask import Flask, render_template, request, redirect, url_for, session

# ============================== [ 1. CONFIGURATION ] ==============================

# Root of all categorised mockups (update if aspect ratio changes)
BASE_DIR = Path("Capitalart-Mockup-Generator/Input/Mockups/4x5-categorised")

# Static web server config
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="/mockups")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mockup-secret-key")

# ============================== [ 2. DYNAMIC CATEGORY SCAN ] ==============================

def get_categories() -> List[str]:
    """Scan the base directory for valid mockup categories (exclude Uncategorised)."""
    return sorted([
        folder.name for folder in BASE_DIR.iterdir()
        if folder.is_dir() and folder.name.lower() != "uncategorised"
    ])

# ============================== [ 3. RANDOM IMAGE PICKING ] ==============================

def random_image(category: str) -> str | None:
    """Return a random image filename from a category folder."""
    category_path = BASE_DIR / category
    if not category_path.exists():
        return None
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(category_path.glob(ext))
    return random.choice(images).name if images else None

# ============================== [ 4. SESSION INITIALISATION ] ==============================

def init_slots():
    """Initialise one random mockup per category and store in session."""
    categories = get_categories()
    session["slots"] = [{"category": cat, "image": random_image(cat)} for cat in categories]

# ============================== [ 5. VALID CATEGORY OPTIONS ] ==============================

def compute_remaining(slots: List[Dict]) -> List[List[str]]:
    """For each slot, compute swappable categories not currently in use."""
    used = [s["category"] for s in slots]
    all_categories = get_categories()
    return [
        [c for c in all_categories if c not in used or c == slot["category"]]
        for slot in slots
    ]

# ============================== [ 6. ROUTES ] ==============================

@app.route("/")
def index():
    if "slots" not in session:
        init_slots()
    slots = session["slots"]
    options = compute_remaining(slots)
    zipped = list(zip(slots, options))
    return render_template("mockup_selector.html", zipped=zipped)

@app.route("/regenerate", methods=["POST"])
def regenerate():
    slot_idx = int(request.form["slot"])
    slots = session.get("slots", [])
    if 0 <= slot_idx < len(slots):
        cat = slots[slot_idx]["category"]
        slots[slot_idx]["image"] = random_image(cat)
        session["slots"] = slots
    return redirect(url_for("index"))

@app.route("/swap", methods=["POST"])
def swap():
    slot_idx = int(request.form["slot"])
    new_cat = request.form["new_category"]
    slots = session.get("slots", [])
    if 0 <= slot_idx < len(slots):
        slots[slot_idx]["category"] = new_cat
        slots[slot_idx]["image"] = random_image(new_cat)
        session["slots"] = slots
    return redirect(url_for("index"))

@app.route("/proceed", methods=["POST"])
def proceed():
    # Composite generation step will be handled in the next script
    return "🚧 Composite generation coming soon!", 200

@app.route("/reset", methods=["POST"])
def reset():
    init_slots()
    return redirect(url_for("index"))

# ============================== [ 7. ENTRY POINT ] ==============================

if __name__ == "__main__":
    app.run(debug=True)

```

---
## 📄 mockup_categoriser.py

```py
# ============================== [ mockup_categoriser.py ] ==============================
# Bulk AI-based mockup categorisation script for CapitalArt Mockup Generator
# --------------------------------------------------------------------------------------
# Uses OpenAI GPT-4.1 (fallback to GPT-4o / Turbo) via .env-configured key
# Categorises mockups from 4x5 folder into detected or predefined categories
# Moves files into categorised folders under `4x5-categorised/`
# Logs results to mockup_categorisation_log.txt
# ======================================================================================

import os
import shutil
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# ============================== [ 1. CONFIG & CONSTANTS ] ==============================

load_dotenv(dotenv_path="/Users/robin/capitalart/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4.1")
FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4-turbo")

client = OpenAI(api_key=OPENAI_API_KEY)

MOCKUP_INPUT_FOLDER = "/Users/robin/capitalart/Capitalart-Mockup-Generator/Input/Mockups/4x5"
MOCKUP_OUTPUT_BASE = "/Users/robin/capitalart/Capitalart-Mockup-Generator/Input/Mockups/4x5-categorised"
LOG_FILE = "/Users/robin/capitalart/mockup_categorisation_log.txt"

# Dynamically detect valid category folders (ignoring 'Uncategorised')
def detect_valid_categories():
    if not os.path.exists(MOCKUP_OUTPUT_BASE):
        return []
    return [
        folder for folder in os.listdir(MOCKUP_OUTPUT_BASE)
        if os.path.isdir(os.path.join(MOCKUP_OUTPUT_BASE, folder)) and folder.lower() != "uncategorised"
    ]

# ============================== [ 2. HELPER FUNCTIONS ] ==============================

def create_category_folders(categories):
    for category in categories:
        folder_path = os.path.join(MOCKUP_OUTPUT_BASE, category)
        os.makedirs(folder_path, exist_ok=True)

def log_result(filename: str, category: str):
    with open(LOG_FILE, "a") as f:
        f.write(f"{filename} -> {category}\n")

def move_file_to_category(file_path: str, category: str):
    dest_folder = os.path.join(MOCKUP_OUTPUT_BASE, category)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.move(file_path, os.path.join(dest_folder, os.path.basename(file_path)))

def is_image(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ============================== [ 3. OPENAI ANALYSIS ] ==============================

def analyse_mockup(file_path: str, valid_categories: list) -> str:
    try:
        encoded_image = encode_image_to_base64(file_path)

        system_prompt = (
            "You are an expert AI assistant helping a professional digital artist organise mockup preview images. "
            "You will receive one image at a time, and your job is to classify it into one of the following categories:\n\n"
            f"{', '.join(valid_categories)}\n\n"
            "These images depict digital artworks displayed in styled rooms. Only respond with the *exact* category name. "
            "If unsure, choose the closest appropriate category based on furniture, lighting, layout or wall style. "
            "No explanations, just return the category string."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=20,
            temperature=0
        )

        category = response.choices[0].message.content.strip()
        if category not in valid_categories:
            raise ValueError(f"Returned category '{category}' is not valid.")
        return category

    except Exception as e:
        print(f"[ERROR] {os.path.basename(file_path)}: {e}")
        return "Uncategorised"

# ============================== [ 4. MAIN EXECUTION ] ==============================

def main():
    print("🔍 Starting mockup categorisation...")

    valid_categories = detect_valid_categories()
    if not valid_categories:
        print("⚠️ No valid category folders found. Please create them first.")
        return

    create_category_folders(valid_categories)

    images = [f for f in os.listdir(MOCKUP_INPUT_FOLDER) if is_image(f)]

    for image_name in images:
        image_path = os.path.join(MOCKUP_INPUT_FOLDER, image_name)
        print(f"→ Analysing {image_name}...")
        category = analyse_mockup(image_path, valid_categories)
        move_file_to_category(image_path, category)
        log_result(image_name, category)
        time.sleep(1.5)

    print("✅ All mockups categorised and moved successfully.")

# ============================== [ 5. ENTRY POINT ] ==============================

if __name__ == "__main__":
    main()

```

---
## 📄 main.py

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 🚀 CapitalArt App Entry Point
# 🔧 FILE: main.py
# 🧠 RULESET: Robbie's Rules™ — No fluff, just fire
# ===========================================

# --- [ 1a: Standard Imports | main-1a ] ---
import os
import sys

# --- [ 1b: Third-Party Imports | main-1b ] ---
from dotenv import load_dotenv

# --- [ 1c: Flask App Import | main-1c ] ---
from mockup_selector_ui import app  # 🔄 import the initialized Flask app

# ===========================================
# 2. 🌱 Environment Setup
# ===========================================

def init_environment():
    """Loads .env variables and validates critical keys."""
    load_dotenv()
    secret_key = os.getenv("FLASK_SECRET_KEY", "")
    if not secret_key or secret_key == "mockup-secret":
        print("⚠️  WARNING: Using default or missing FLASK_SECRET_KEY. Set a strong one in your .env.")
    app.secret_key = secret_key


# ===========================================
# 3. 🚀 Run the App
# ===========================================

def run_server():
    """Launch the Flask development server."""
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)


# ===========================================
# 🔚 Main Bootstrap
# ===========================================

if __name__ == "__main__":
    print("🎨 Launching CapitalArt Mockup Selector UI...")
    init_environment()
    run_server()

```

---
## 📄 generate_folder_tree.py

```py
import os

# ============================== [ CONFIGURATION ] ==============================

ROOT_DIR = "."  # Set to "." to run from project root
OUTPUT_FILE = "folder_structure.txt"
IGNORE_DIRS = {".git", "__pycache__", ".venv", ".idea", ".DS_Store", "node_modules", "env"}

# ============================== [ HELPER FUNCTION ] ==============================

def generate_tree(start_path: str, prefix: str = "") -> str:
    tree_str = ""
    entries = sorted(os.listdir(start_path))
    entries = [e for e in entries if e not in IGNORE_DIRS]

    for idx, entry in enumerate(entries):
        full_path = os.path.join(start_path, entry)
        connector = "└── " if idx == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry}\n"

        if os.path.isdir(full_path):
            extension = "    " if idx == len(entries) - 1 else "│   "
            tree_str += generate_tree(full_path, prefix + extension)
    return tree_str

# ============================== [ MAIN EXECUTION ] ==============================

if __name__ == "__main__":
    print(f"📂 Generating folder structure starting at: {os.path.abspath(ROOT_DIR)}")
    tree_output = f"{os.path.basename(os.path.abspath(ROOT_DIR))}\n"
    tree_output += generate_tree(ROOT_DIR)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(tree_output)

    print(f"✅ Folder structure written to: {OUTPUT_FILE}")

```

---
## 📄 requirements.txt

```txt
# === [ Core Functionality ] ===
openai>=1.0.0
python-dotenv>=1.0.0
requests>=2.30.0
Pillow>=9.5.0

# === [ UI for Mockup Selector ] ===
Flask>=2.3.0

# === [ Optional: Async support and fast file serving ] ===
httpx>=0.24.0

# === [ Optional: For fast CSV handling / tabular exports later ] ===
pandas>=2.0.0

# === [ Optional: Image preview thumbnails or resizing ops ] ===
opencv-python>=4.7.0.72

# === [ Optional: Fast and pretty logs for CLI tools ] ===
rich>=13.4.0
```

---
## 📄 templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CapitalArt Mockup Gallery</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
  <header class="gallery-header">
    <h1>🎨 CapitalArt Mockup Gallery</h1>
    <p>Browse artworks, explore categories, and preview AI-powered Etsy listings</p>
  </header>

  <main id="gallery">
    <!-- Artwork mockups dynamically injected here -->
  </main>

  <footer class="gallery-footer">
    <p>© Robin Custance • Proudly on Kaurna Country • GitHub Pages Powered</p>
  </footer>

  <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>

```

---
## 📄 templates/mockup_selector.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>CapitalArt Mockup Selector</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
  <h1>🖼️ Select Your Mockup Lineup</h1>
  <div class="grid">
    {% for slot, options in zipped %}
    <div class="item">
      {% if slot.image %}
      <img src="{{ url_for('static', filename=slot.category + '/' + slot.image) }}" alt="{{ slot.category }}" />
      {% else %}
      <p>No images for {{ slot.category }}</p>
      {% endif %}
      <strong>{{ slot.category }}</strong>
      <form method="post" action="{{ url_for('regenerate') }}">
        <input type="hidden" name="slot" value="{{ loop.index0 }}" />
        <button type="submit">🔄 Regenerate</button>
      </form>
      <form method="post" action="{{ url_for('swap') }}">
        <input type="hidden" name="slot" value="{{ loop.index0 }}" />
        <select name="new_category">
          {% for c in options %}
          <option value="{{ c }}" {% if c==slot.category %}selected{% endif %}>{{ c }}</option>
          {% endfor %}
        </select>
        <button type="submit">🔁 Swap</button>
      </form>
    </div>
    {% endfor %}
  </div>
  <form method="post" action="{{ url_for('proceed') }}">
    <button class="composite-btn" type="submit">✅ Proceed to Composite</button>
  </form>
</body>
</html>

```

---
## 📄 static/js/main.js

```js
// ==============================
// CapitalArt Mockup Gallery Script
// ==============================

document.addEventListener("DOMContentLoaded", () => {
  const gallery = document.getElementById("gallery");

  // Example data (replace with dynamic fetch later)
  const artworks = [
    {
      title: "Red Earth Songlines",
      image: "/mockups/Living Room/Red-Earth-Songlines-MU-01.jpg",
      description: "Digital dot painting in a styled living room."
    },
    {
      title: "Sunset Dreaming",
      image: "/mockups/Bedroom/Sunset-Dreaming-MU-01.jpg",
      description: "Aboriginal-inspired dot art mockup in bedroom setting."
    }
  ];

  artworks.forEach(art => {
    const div = document.createElement("div");
    div.className = "gallery-item";
    div.innerHTML = `
      <img src="${art.image}" alt="${art.title}">
      <h2>${art.title}</h2>
      <p>${art.description}</p>
    `;
    gallery.appendChild(div);
  });
});

```

---
## 📄 static/css/style.css

```css
/* ==============================
   CapitalArt Mockup Selector UI
   Style Sheet — Robbie Mode™
   ============================== */

body {
  font-family: system-ui, sans-serif;
  margin: 0;
  padding: 2em;
  background: #f9f9f9;
  color: #333;
}

h1 {
  text-align: center;
  margin-bottom: 1.5em;
  font-size: 2.2em;
}

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5em;
  justify-content: center;
}

.item {
  background: #fff;
  border: 1px solid #ddd;
  padding: 1em;
  border-radius: 8px;
  width: 240px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  text-align: center;
  transition: transform 0.2s ease;
}

.item:hover {
  transform: scale(1.02);
}

.item img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  margin-bottom: 0.75em;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

form {
  margin: 0.4em 0;
}

select,
button {
  font-size: 0.95em;
  padding: 0.4em;
  border: 1px solid #bbb;
  border-radius: 4px;
  margin-top: 0.3em;
  cursor: pointer;
}

button:hover {
  background-color: #eee;
}

.composite-btn {
  display: block;
  margin: 3em auto 0;
  padding: 0.75em 1.5em;
  font-size: 1em;
  background: #007acc;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s ease;
}

.composite-btn:hover {
  background: #005fa3;
}

/* ==============================
   CapitalArt Mockup Gallery Styles
   ============================== */

.gallery-header {
  text-align: center;
  margin-bottom: 2em;
}

.gallery-header h1 {
  font-size: 2.4em;
  margin-bottom: 0.3em;
}

.gallery-header p {
  font-size: 1.1em;
  color: #666;
}

#gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 2em;
  padding: 1em;
}

.gallery-item {
  background: #fff;
  border: 1px solid #ddd;
  padding: 1em;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s ease;
}

.gallery-item:hover {
  transform: scale(1.02);
}

.gallery-item img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  margin-bottom: 0.5em;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

.gallery-item h2 {
  font-size: 1.1em;
  margin: 0.5em 0 0.2em;
  color: #007acc;
}

.gallery-item p {
  font-size: 0.95em;
  color: #444;
}

.gallery-footer {
  text-align: center;
  margin-top: 4em;
  padding: 1em;
  font-size: 0.9em;
  color: #777;
}

```

---
## 📄 Capitalart-Mockup-Generator/scripts/compatible-image-generator.jsx

```jsx
// ===================================================
// Photoshop JSX Script: Export JPEG (Harvey Norman Safe)
// ===================================================

// Set export folder (same as current file)
var doc = app.activeDocument;
var exportFolder = doc.path;

// Set output name
var fileName = doc.name.replace(/\.[^\.]+$/, '') + "_HN-safe.jpg";
var saveFile = new File(exportFolder + "/" + fileName);

// JPEG export options
var jpegOptions = new JPEGSaveOptions();
jpegOptions.quality = 9; // Max quality
jpegOptions.embedColorProfile = true;
jpegOptions.formatOptions = FormatOptions.STANDARDBASELINE; // Not Progressive
jpegOptions.matte = MatteType.NONE;

// Export
doc.saveAs(saveFile, jpegOptions, true);

// Alert when done
alert("✅ Export complete:\n" + fileName + "\nReady for Harvey Norman printing.");

```

---
## 📄 Capitalart-Mockup-Generator/scripts/backup_mockup_structure.sh

```sh
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
echo "----------------------------------------------------"
```

---
## 📄 Capitalart-Mockup-Generator/scripts/export-artwork-layers.jsx

```jsx
#target photoshop

function zeroPad(n, width) {
    width = width || 2;
    n = n + '';
    return n.length >= width ? n : new Array(width - n.length + 1).join('0') + n;
}

var doc = app.activeDocument;
var outputFolder = Folder.selectDialog("Select folder to save artwork layers");

if (outputFolder == null) {
    alert("Export cancelled.");
} else {
    for (var i = 0; i < doc.layerSets.length; i++) {
        var group = doc.layerSets[i];
        var groupName = group.name;
        var index = zeroPad(i + 1); // Starts at 01

        // Hide all groups
        for (var j = 0; j < doc.layerSets.length; j++) {
            doc.layerSets[j].visible = false;
        }

        group.visible = true;

        // Save as PNG
        var saveFile = new File(outputFolder + "/artwork-layer-" + index + ".png");
        var opts = new PNGSaveOptions();
        opts.compression = 9;
        doc.saveAs(saveFile, opts, true, Extension.LOWERCASE);
    }

    alert("✅ All groups exported as PNGs!");
}

```

---
## 📄 Capitalart-Mockup-Generator/scripts/fix_srgb_profiles.py

```py
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
## 📄 Capitalart-Mockup-Generator/scripts/backup_mockup_generator.sh

```sh
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
echo "----------------------------------------------------"
```

---
## 📄 Capitalart-Mockup-Generator/scripts/rename-4x5-mockup-layers.jsx

```jsx
// ====================================================
// 📄 Rename Only Visible Layers to "4x5-mockup-01" etc.
// ✅ Works inside groups too
// 🛠️ Usage: File > Scripts > Browse... in Photoshop
// ====================================================

function renameVisibleLayers(prefix) {
    if (!app.documents.length) {
        alert("No document open!");
        return;
    }

    var doc = app.activeDocument;
    var count = 1;

    function pad(num, size) {
        var s = "00" + num;
        return s.substr(s.length - size);
    }

    function processLayer(layer) {
        if (!layer.visible) return; // Skip hidden layers

        if (layer.typename === "ArtLayer") {
            layer.name = prefix + pad(count, 2);
            count++;
        } else if (layer.typename === "LayerSet") {
            for (var i = 0; i < layer.layers.length; i++) {
                processLayer(layer.layers[i]);
            }
        }
    }

    // Reverse order for correct stacking
    for (var i = doc.layers.length - 1; i >= 0; i--) {
        processLayer(doc.layers[i]);
    }

    alert("✅ Visible layers renamed as '" + prefix + "##'");
}

// Run it with your desired prefix
renameVisibleLayers("4x5-mockup-");

```

---
## 📄 Capitalart-Mockup-Generator/scripts/find-matching-png-files.py

```py
import os
import shutil
from PIL import Image # Pillow library for image processing
import imagehash

# --- Pillow Configuration for Large Images ---
# Set to None to disable decompression bomb check, or a sufficiently large number.
Image.MAX_IMAGE_PIXELS = None # Allow processing of very large images

# --- Script Configuration ---
# Path to the folder containing your original JPG images
jpg_originals_dir = "/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/Current-Listing-Original-Images"

# IMPORTANT: Ensure this path is correct and does NOT have extra quotes around it.
png_search_dir = "/Users/robin/Documents/sorted-images" # <<< ENSURE THIS IS CORRECT

# Path to the folder where matching PNG images will be copied
destination_dir = "/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/matching-png-images"

# PNG File Size Filters
max_png_filesize_mb = 50  # PNGs larger than this will be skipped
MAX_PNG_FILESIZE_BYTES = max_png_filesize_mb * 1024 * 1024

min_png_filesize_mb = 4   # PNGs smaller than or equal to this will be skipped (must be > 4MB)
MIN_PNG_FILESIZE_BYTES = min_png_filesize_mb * 1024 * 1024

# PNG Dimension Filter
min_png_long_edge_px = 2400 # PNGs must have at least one dimension (width or height) >= this value

# Perceptual hash settings
hash_size = 8  # Higher is more sensitive to changes, 8 is a good default.
similarity_threshold = 5 # Lower means images must be MORE similar. 0 is an exact hash match.

# --- Helper Functions ---
def get_image_dimensions(image_path):
    """Gets the dimensions (width, height) of an image."""
    try:
        with Image.open(image_path) as img: # Use 'with' to ensure file is closed
            return img.size  # Returns (width, height)
    except FileNotFoundError:
        print(f"    Error (get_dimensions): Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"    Error (get_dimensions): Could not open or read dimensions for {image_path}: {e}")
        return None

def get_image_hash(image_path, hash_size_val=8):
    """Generates a perceptual hash (phash) for an image."""
    try:
        # Open image using 'with' to ensure it's closed properly
        with Image.open(image_path) as img:
            img_gray = img.convert('L') # Convert to grayscale for hashing
            return imagehash.phash(img_gray, hash_size=hash_size_val)
    except FileNotFoundError:
        # This error should ideally be caught before calling get_image_hash,
        # but it's good to have a fallback.
        print(f"    Error (get_hash): Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"    Error (get_hash): Could not process/hash {os.path.basename(image_path)}: {e}")
        return None

def find_png_files_recursively(search_directory):
    """Finds all PNG files in a directory and all its subdirectories."""
    png_files_list = []
    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.lower().endswith(".png"):
                png_files_list.append(os.path.join(root, file))
    return png_files_list

# --- Main Script Logic ---
def main():
    print("--- Image Matching Script Initializing ---")
    # Using a static date here as an example.
    print(f"Script version date: May 19, 2025 (ACST)") # User context: Current date is May 19, 2025

    # Validate input directories
    if not os.path.isdir(jpg_originals_dir):
        print(f"Error: Originals JPG directory not found: {jpg_originals_dir}")
        return

    if not os.path.isdir(png_search_dir) or png_search_dir == "/path/to/your/parent/png_folder_with_subdirectories": # Default placeholder check
        print(f"Error: PNG search directory not found or not configured correctly: {png_search_dir}")
        print("Please update the 'png_search_dir' variable in the script.")
        return

    if not os.path.exists(destination_dir):
        try:
            os.makedirs(destination_dir)
            print(f"Created destination directory: {destination_dir}")
        except OSError as e:
            print(f"Error creating destination directory {destination_dir}: {e}")
            return

    print(f"\nScanning for original JPG images in: {jpg_originals_dir}")
    try:
        original_jpg_files = [
            os.path.join(jpg_originals_dir, f)
            for f in os.listdir(jpg_originals_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
    except OSError as e:
        print(f"Error reading JPG directory {jpg_originals_dir}: {e}")
        return

    if not original_jpg_files:
        print(f"No JPG files found in {jpg_originals_dir}")
        return
    print(f"Found {len(original_jpg_files)} JPG(s) to process.")

    print(f"\nSearching for all PNG files recursively in: {png_search_dir}")
    all_potential_pngs = find_png_files_recursively(png_search_dir)

    if not all_potential_pngs:
        print(f"No PNG files found in {png_search_dir} or its subdirectories.")
        return
    print(f"Initially found {len(all_potential_pngs)} potential PNG(s).")

    print(f"\nApplying PNG file filters:")
    print(f"  - Min file size: > {min_png_filesize_mb}MB")
    print(f"  - Max file size: <= {max_png_filesize_mb}MB")
    print(f"  - Min long edge dimension: >= {min_png_long_edge_px}px")
    print(f"This filtering step may take some time as it involves checking file sizes and image dimensions...")

    all_pngs_to_check = []
    skipped_reason_counts = {"size_too_small": 0, "size_too_large": 0, "dim_too_small": 0, "access_error": 0, "dimension_error": 0}

    for i, png_path in enumerate(all_potential_pngs):
        png_filename = os.path.basename(png_path)
        # Print progress for filtering stage
        if (i + 1) % 100 == 0 or i == len(all_potential_pngs) -1 : # Print every 100 files or for the last file
            print(f"  Filtering progress: Checked {i+1}/{len(all_potential_pngs)} potential PNGs...")

        try:
            file_size = os.path.getsize(png_path)

            # 1. Filter by MINIMUM file size
            if file_size <= MIN_PNG_FILESIZE_BYTES:
                skipped_reason_counts["size_too_small"] += 1
                continue # Skip to next PNG

            # 2. Filter by MAXIMUM file size
            if file_size > MAX_PNG_FILESIZE_BYTES:
                skipped_reason_counts["size_too_large"] += 1
                continue # Skip to next PNG

            # 3. Filter by MINIMUM dimension (long edge)
            dimensions = get_image_dimensions(png_path)
            if dimensions:
                width, height = dimensions
                if not (width >= min_png_long_edge_px or height >= min_png_long_edge_px):
                    skipped_reason_counts["dim_too_small"] += 1
                    continue # Skip to next PNG
            else:
                # get_image_dimensions would have printed an error
                skipped_reason_counts["dimension_error"] += 1
                continue # Skip if dimensions could not be read

            # If all filters passed, add to the list for hashing
            all_pngs_to_check.append(png_path)

        except FileNotFoundError:
            print(f"  Warning: PNG file {png_filename} found during scan but not accessible during filtering. Skipping.")
            skipped_reason_counts["access_error"] +=1
        except Exception as e:
            print(f"  Warning: Could not process {png_filename} during filtering: {e}. Skipping.")
            skipped_reason_counts["access_error"] += 1

    print(f"\nPNG Filtering Complete:")
    print(f"  Skipped {skipped_reason_counts['size_too_small']} PNGs (too small, <= {min_png_filesize_mb}MB)")
    print(f"  Skipped {skipped_reason_counts['size_too_large']} PNGs (too large, > {max_png_filesize_mb}MB)")
    print(f"  Skipped {skipped_reason_counts['dim_too_small']} PNGs (dimensions < {min_png_long_edge_px}px on long edge)")
    print(f"  Skipped {skipped_reason_counts['dimension_error']} PNGs (could not read dimensions)")
    print(f"  Skipped {skipped_reason_counts['access_error']} PNGs (file access/other errors during filter)")

    if not all_pngs_to_check:
        print(f"\nNo PNG files remaining after applying all filters.")
        return

    print(f"\nProcessing {len(all_pngs_to_check)} PNG(s) that passed all filters.")
    print(f"Using perceptual hash (phash) with hash size: {hash_size}")
    print(f"Similarity threshold set to: {similarity_threshold} (a lower number means images must be more similar).")
    if Image.MAX_IMAGE_PIXELS is None:
        print("Pillow's MAX_IMAGE_PIXELS limit is disabled; large dimension images will be attempted.")
    else:
        print(f"Pillow's MAX_IMAGE_PIXELS is set to: {Image.MAX_IMAGE_PIXELS}")


    copied_count = 0
    unique_copied_pngs = set() # To avoid copying the same PNG multiple times

    png_hashes_map = {}
    total_pngs_to_hash = len(all_pngs_to_check)
    print(f"\nCalculating hashes for {total_pngs_to_hash} filtered PNG files (this may take a while)...")

    for i, png_path in enumerate(all_pngs_to_check):
        current_num = i + 1
        png_filename = os.path.basename(png_path)
        print(f"  [{current_num}/{total_pngs_to_hash}] Hashing PNG: {png_filename}...")
        
        png_hash = get_image_hash(png_path, hash_size)
        if png_hash:
            png_hashes_map[png_path] = png_hash
    
    print(f"\nFinished calculating PNG hashes.")
    print(f"Successfully calculated {len(png_hashes_map)} PNG hashes out of {total_pngs_to_hash} attempted.")
    if len(png_hashes_map) < total_pngs_to_hash:
        print(f"Warning: {total_pngs_to_hash - len(png_hashes_map)} PNG(s) could not be hashed (see error messages above for details).")


    for jpg_path in original_jpg_files:
        jpg_filename = os.path.basename(jpg_path)
        print(f"\nProcessing original JPG: {jpg_filename}")
        
        jpg_hash = get_image_hash(jpg_path, hash_size)

        if not jpg_hash:
            print(f"  Could not generate hash for {jpg_filename}. Skipping.")
            continue

        found_match_for_current_jpg = False
        for png_path, png_hash in png_hashes_map.items():
            hash_difference = jpg_hash - png_hash

            if hash_difference <= similarity_threshold:
                png_filename_match = os.path.basename(png_path)
                print(f"  MATCH FOUND: '{jpg_filename}' is similar to '{png_filename_match}' (Difference: {hash_difference})")
                destination_file_path = os.path.join(destination_dir, png_filename_match)

                if png_path not in unique_copied_pngs:
                    if not os.path.exists(destination_file_path):
                        try:
                            shutil.copy2(png_path, destination_file_path)
                            print(f"    Copied '{png_filename_match}' to '{destination_dir}'")
                            copied_count += 1
                            unique_copied_pngs.add(png_path)
                        except Exception as e:
                            print(f"    Error copying {png_filename_match}: {e}")
                    else:
                        print(f"    Skipped copying '{png_filename_match}', file with that name already exists in destination.")
                        unique_copied_pngs.add(png_path) # Still mark as processed
                else:
                    print(f"    '{png_filename_match}' was already identified as a match and processed for a previous JPG.")
                found_match_for_current_jpg = True

        if not found_match_for_current_jpg:
            print(f"  No PNG found sufficiently similar to {jpg_filename} (based on current filters and threshold of {similarity_threshold}).")

    print(f"\n--- Image Matching Process Complete ---")
    print(f"Total unique matching PNG images copied: {copied_count}")
    if len(original_jpg_files) > 0 and len(all_pngs_to_check) > 0 : # Check if there were PNGs to check after filtering
        if copied_count == 0:
            print(f"No matches found. Consider:")
            print(f"  - Adjusting 'similarity_threshold' (currently {similarity_threshold}).")
            print(f"  - Reviewing PNG filter settings (sizes: >{min_png_filesize_mb}MB & <={max_png_filesize_mb}MB; dimension: >={min_png_long_edge_px}px).")
            print(f"  - Ensuring 'png_search_dir' is correct and contains the expected PNGs.")
        else:
            print(f"Matching files are in: {destination_dir}")
    elif len(original_jpg_files) > 0 and len(all_potential_pngs) > 0 and len(all_pngs_to_check) == 0:
        print("No PNGs passed the filtering criteria. No matching could be performed.")
        print(f"  Review PNG filter settings (sizes: >{min_png_filesize_mb}MB & <={max_png_filesize_mb}MB; dimension: >={min_png_long_edge_px}px).")


    print("-----------------------------------------")

if __name__ == "__main__":
    # Before running, ensure you have Pillow and ImageHash installed:
    # pip install Pillow ImageHash
    main()

```

---
## 📄 Capitalart-Mockup-Generator/scripts/openai_vision_test.py

```py
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
## 📄 Capitalart-Mockup-Generator/scripts/generate_all_coordinates.py

```py
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
## 📄 Capitalart-Mockup-Generator/scripts/generate_composites.py

```py
import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

# =========================== [CONFIGURABLES] ===========================
possible_aspect_ratios = [
    "1x1",
    "2x3",
    "3x2",
    "3x4",
    "4x3",
    "4x5",
    "5x4",
    "5x7",
    "7x5",
    "9x16",
    "16x9",
    "A-Series-Horizontal",
    "A-Series-Vertical"
]
# =======================================================================

overall_artworks_found = False

for aspect_ratio in possible_aspect_ratios:
    # --- [1. Directory Setup] ---
    input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
    input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
    input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
    output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
    os.makedirs(output_root_dir, exist_ok=True)

    # --- [2. Helper Functions] ---
    def resize_image_for_long_edge(image: Image.Image, target_long_edge=2000) -> Image.Image:
        width, height = image.size
        if width > height:
            new_width = target_long_edge
            new_height = int(height * (target_long_edge / width))
        else:
            new_height = target_long_edge
            new_width = int(width * (target_long_edge / height))
        return image.resize((new_width, new_height), Image.LANCZOS)

    def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
        draw = ImageDraw.Draw(img)
        for x, y in points:
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
        return img

    def apply_perspective_transform(art_img, mockup_img, dst_coords):
        w, h = art_img.size
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

    # --- [3. Find Artworks] ---
    if not os.path.exists(input_artworks_dir):
        print(f"⚠️ Artwork folder not found for {aspect_ratio}: {input_artworks_dir}")
        continue

    artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not artwork_files:
        print(f"⚠️ No artwork files found in {aspect_ratio} folder: {input_artworks_dir}")
        continue
    else:
        overall_artworks_found = True

    print(f"\n✨ Processing {len(artwork_files)} artworks for {aspect_ratio} aspect ratio...")

    # === [4. Process Each Artwork] ===
    for index, artwork_file in enumerate(artwork_files):
        # --- [4.1 Clean Base Name] ---
        cleaned_base_name = os.path.splitext(artwork_file)[0]
        cleaned_base_name = cleaned_base_name.replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
        cleaned_base_name = "-".join(filter(None, cleaned_base_name.split('-')))

        # --- [4.2 Output Set Directory (Artwork-Based Naming)] ---
        set_dir = os.path.join(output_root_dir, f"{cleaned_base_name}-Mockups")
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")

        # --- [4.3 Generate Preview Image & Save to Output Folder] ---
        preview_img = resize_image_for_long_edge(art_img.copy(), target_long_edge=2000)
        preview_filename = f"{cleaned_base_name}-THUMB-01.jpg"
        preview_path = os.path.join(set_dir, preview_filename)  # <-- NOW in the set_dir (output, not input)
        target_file_size_kb = 700
        initial_quality = 85
        min_quality = 50
        current_quality = initial_quality
        file_size_bytes = float('inf')

        print(f"Attempting to save preview for {artwork_file} under {target_file_size_kb}KB...")

        while file_size_bytes > (target_file_size_kb * 1024) and current_quality >= min_quality:
            preview_img.convert("RGB").save(preview_path, "JPEG", quality=current_quality, optimize=True)
            file_size_bytes = os.path.getsize(preview_path)
            file_size_kb = file_size_bytes / 1024

            if file_size_kb > target_file_size_kb:
                print(f"  Current size: {file_size_kb:.2f}KB (Quality: {current_quality}). Reducing quality...")
                current_quality -= 5
            else:
                print(f"  Achieved target! Final size: {file_size_kb:.2f}KB (Quality: {current_quality}).")
                break

            if current_quality < min_quality:
                print(f"  Warning: Minimum quality ({min_quality}) reached. Final size: {file_size_kb:.2f}KB. Target may not be met.")

        print(f"✅ Saved Preview: {preview_filename} (Long edge 2000px, placed in {set_dir})")

        # --- [4.4 Copy Original and TXT if exists] ---
        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, os.path.splitext(artwork_file)[0] + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, os.path.splitext(artwork_file)[0] + ".txt"))

        # --- [4.5 Prepare Mockup Processing] ---
        if not os.path.exists(input_mockups_dir):
            print(f"⚠️ Mockup folder not found for {aspect_ratio}: {input_mockups_dir}")
            continue

        art_img_for_composite = resize_image_for_long_edge(art_img, target_long_edge=2000)

        # -------- [4.6 MOCKUP NUMBERING RESET FOR THIS ARTWORK] --------
        mockup_seq = 1

        # --- [4.7 Process Each Mockup for this Artwork] ---
        for mockup_file in sorted(os.listdir(input_mockups_dir)):
            if "-THUMB-01.jpg" in mockup_file:
                continue
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")
            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file} in {aspect_ratio} folder.")
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
            composite = apply_perspective_transform(art_img_for_composite, mockup_img, dst_coords)

            # --- [4.8 Sequential Naming of Mockups] ---
            output_filename = f"{cleaned_base_name}-MU-{mockup_seq:02d}.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved Composite: {output_filename}")
            mockup_seq += 1

    print(f"🎯 Finished processing for {aspect_ratio} aspect ratio.")

# === [5. Summary Message] ===
if not overall_artworks_found:
    print("\n⚠️ No artwork files found in any of the specified aspect ratio folders.")
else:
    print("\n🎉 All composite sets and previews completed for all found aspect ratios.")

```

---
## 📄 Capitalart-Mockup-Generator/scripts/utils.py

```py
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
## 📄 Capitalart-Mockup-Generator/scripts/dupe-png-files-and-rename.py

```py
import os
import shutil

# --- Script Configuration ---
# Path to the base directory containing the subfolders (e.g., "matching-png-images")
# Each subfolder within this directory is expected to contain one JPG and one PNG.
organization_base_dir = "/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/matching-png-images"

# --- Main Script Logic ---
def main():
    print("--- PNG Duplication Script Initializing ---")
    print(f"Script version date: May 20, 2025 (ACST)")
    print(f"Looking for subfolders in: {organization_base_dir}")

    if not os.path.isdir(organization_base_dir):
        print(f"Error: Base directory not found: {organization_base_dir}")
        return

    duplicated_png_count = 0
    subfolders_processed = 0
    subfolders_skipped = 0

    # Iterate through items in the organization_base_dir
    for item_name in os.listdir(organization_base_dir):
        item_path = os.path.join(organization_base_dir, item_name)

        # Check if the item is a directory (a subfolder for an artwork)
        if os.path.isdir(item_path):
            subfolder_path = item_path
            subfolder_name = item_name
            print(f"\nProcessing subfolder: {subfolder_name}")
            subfolders_processed += 1

            jpg_file_path = None
            original_png_file_path = None
            jpg_filename_no_ext = None

            # Find the JPG and PNG file in the subfolder
            files_in_subfolder = os.listdir(subfolder_path)
            found_jpgs = []
            found_pngs = []

            for file_in_subfolder in files_in_subfolder:
                if file_in_subfolder.lower().endswith((".jpg", ".jpeg")):
                    found_jpgs.append(os.path.join(subfolder_path, file_in_subfolder))
                elif file_in_subfolder.lower().endswith(".png"):
                    # We need to be careful not to pick up an already duplicated PNG
                    # if the script is run multiple times.
                    # For now, let's assume the user's cleanup means there's one "original" PNG.
                    # A more robust way would be to identify the PNG that *doesn't* match the JPG name.
                    found_pngs.append(os.path.join(subfolder_path, file_in_subfolder))

            # Validate files found
            if len(found_jpgs) == 1:
                jpg_file_path = found_jpgs[0]
                jpg_filename_no_ext = os.path.splitext(os.path.basename(jpg_file_path))[0]
                print(f"  Found JPG: {os.path.basename(jpg_file_path)}")
            elif len(found_jpgs) == 0:
                print(f"  Warning: No JPG file found in {subfolder_name}. Skipping this subfolder.")
                subfolders_skipped += 1
                continue
            else:
                print(f"  Warning: Multiple JPG files found in {subfolder_name}. Skipping this subfolder.")
                subfolders_skipped += 1
                continue

            # Now, try to find the "original" PNG.
            # If a PNG already named like the JPG exists, we want to identify the *other* PNG.
            target_duplicate_png_name = jpg_filename_no_ext + ".png"
            
            if len(found_pngs) == 1:
                # If only one PNG, assume it's the one to duplicate.
                original_png_file_path = found_pngs[0]
                print(f"  Found PNG: {os.path.basename(original_png_file_path)}")
            elif len(found_pngs) == 2:
                # If two PNGs, one might be the original and one the already-duplicated one.
                # We want to copy the one that ISN'T named like the JPG.
                png1_name = os.path.basename(found_pngs[0])
                png2_name = os.path.basename(found_pngs[1])

                if png1_name == target_duplicate_png_name: # png1 is the duplicate
                    original_png_file_path = found_pngs[1] # so png2 must be the original
                elif png2_name == target_duplicate_png_name: # png2 is the duplicate
                    original_png_file_path = found_pngs[0] # so png1 must be the original
                else:
                    # Neither are named like the JPG, this is an ambiguous case if we expect one original.
                    # For now, let's just pick the first one and let the duplicate check handle it.
                    # Or, better, ask user to ensure one original.
                    print(f"  Warning: Two PNGs found in {subfolder_name}, and neither is named '{target_duplicate_png_name}'.")
                    print(f"    Found: '{png1_name}' and '{png2_name}'.")
                    print(f"    Please ensure only one 'source' PNG exists or one is named like the JPG for clear duplication.")
                    print(f"    Attempting to use '{png1_name}' as source.")
                    original_png_file_path = found_pngs[0]


                if original_png_file_path:
                     print(f"  Identified source PNG: {os.path.basename(original_png_file_path)}")

            elif len(found_pngs) == 0:
                print(f"  Warning: No PNG file found in {subfolder_name}. Skipping this subfolder.")
                subfolders_skipped += 1
                continue
            else: # More than 2 PNGs
                print(f"  Warning: More than two PNG files found in {subfolder_name}. Skipping, as source PNG is ambiguous.")
                subfolders_skipped += 1
                continue
            
            if not original_png_file_path: # If logic above failed to set it
                print(f"  Could not definitively identify a source PNG in {subfolder_name}. Skipping.")
                subfolders_skipped +=1
                continue


            # Define the target path for the duplicate PNG
            duplicate_png_target_path = os.path.join(subfolder_path, target_duplicate_png_name)

            # Check if the target duplicate already exists
            if os.path.exists(duplicate_png_target_path):
                # If the existing duplicate is identical to the source, it's fine.
                # If not, it's an issue. For simplicity, we'll just skip if it exists.
                # A more advanced check could compare file sizes or even hashes if necessary.
                if os.path.abspath(original_png_file_path) == os.path.abspath(duplicate_png_target_path):
                    print(f"  The identified 'original' PNG is already named like the JPG. No duplication needed.")
                else:
                    print(f"  Duplicate PNG '{target_duplicate_png_name}' already exists. Skipping duplication.")
            else:
                # Create the duplicate
                try:
                    shutil.copy2(original_png_file_path, duplicate_png_target_path)
                    print(f"    Successfully duplicated '{os.path.basename(original_png_file_path)}' to '{target_duplicate_png_name}'")
                    duplicated_png_count += 1
                except Exception as e:
                    print(f"    Error duplicating PNG in {subfolder_name}: {e}")
        else:
            # print(f"Skipping non-directory item: {item_name}") # Optional: for debugging
            pass


    print(f"\n--- PNG Duplication Process Complete ---")
    print(f"Subfolders scanned: {subfolders_processed}")
    if subfolders_skipped > 0:
        print(f"Subfolders skipped due to issues (see warnings above): {subfolders_skipped}")
    print(f"PNGs duplicated: {duplicated_png_count}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

```

---
## 📄 Capitalart-Mockup-Generator/scripts/google_vision_basic_analyzer.py

```py
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
## 📄 Capitalart-Mockup-Generator/scripts/generate_folder_structure.py

```py
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
    print(f"✅ Folder structure saved to: {output_file}")
```

---
## 📄 Capitalart-Mockup-Generator/scripts/organise-jpg-and-png-files.py

```py
import os
import shutil
from PIL import Image # Pillow library for image processing
import imagehash

# --- Pillow Configuration for Large Images ---
# Consistent with the previous script
Image.MAX_IMAGE_PIXELS = None # Allow processing of very large images

# --- Script Configuration ---
# Path to the folder containing your original JPG images (these will be MOVED)
jpg_originals_source_dir = "/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/Current-Listing-Original-Images"

# Path to the folder where "loose" matching PNGs currently reside AND
# where the new organized subfolders will be created.
# PNGs from the root of this directory will be MOVED into subfolders.
organization_base_dir = "/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/matching-png-images"

# Perceptual hash settings - SHOULD MATCH THE SCRIPT THAT POPULATED matching-png-images
hash_size = 8
similarity_threshold = 5 # Lower means images must be MORE similar.

# --- Helper Functions ---
def get_image_hash(image_path, hash_size_val=8):
    """Generates a perceptual hash (phash) for an image."""
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L') # Convert to grayscale for hashing
            return imagehash.phash(img_gray, hash_size=hash_size_val)
    except FileNotFoundError:
        print(f"    Error (get_hash): Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"    Error (get_hash): Could not process/hash {os.path.basename(image_path)}: {e}")
        return None

# --- Main Script Logic ---
def main():
    print("--- JPG and PNG Organizer Script Initializing ---")
    print(f"Script version date: May 19, 2025 (ACST)")

    # Validate input directories
    if not os.path.isdir(jpg_originals_source_dir):
        print(f"Error: JPG originals source directory not found: {jpg_originals_source_dir}")
        return

    if not os.path.isdir(organization_base_dir):
        print(f"Error: Organization base directory not found: {organization_base_dir}")
        print(f"This directory should contain the PNGs copied by the previous script and is where subfolders will be created.")
        return

    print(f"\nSource JPGs will be read from: {jpg_originals_source_dir}")
    print(f"Source PNGs will be read from the root of: {organization_base_dir}")
    print(f"Organized folders will be created inside: {organization_base_dir}")
    print(f"Using hash size: {hash_size} and similarity threshold: {similarity_threshold}")

    # 1. Get list of all JPG files from the source directory
    try:
        original_jpg_files = [
            os.path.join(jpg_originals_source_dir, f)
            for f in os.listdir(jpg_originals_source_dir)
            if os.path.isfile(os.path.join(jpg_originals_source_dir, f)) and \
               f.lower().endswith((".jpg", ".jpeg"))
        ]
    except OSError as e:
        print(f"Error reading JPG source directory {jpg_originals_source_dir}: {e}")
        return

    if not original_jpg_files:
        print(f"No JPG files found in {jpg_originals_source_dir} to organize.")
        # return # Allow to continue if only PNGs need organizing, though less common use case

    # 2. Get list of all PNG files from the ROOT of the organization_base_dir
    # These are the PNGs that were previously matched and copied.
    try:
        available_png_paths = [
            os.path.join(organization_base_dir, f)
            for f in os.listdir(organization_base_dir)
            if os.path.isfile(os.path.join(organization_base_dir, f)) and \
               f.lower().endswith(".png")
        ]
    except OSError as e:
        print(f"Error reading PNGs from organization base directory {organization_base_dir}: {e}")
        return

    if not available_png_paths:
        print(f"No PNG files found in the root of {organization_base_dir} to organize.")
        if not original_jpg_files: # If no JPGs either, then nothing to do.
            return


    # 3. Pre-calculate hashes for all available PNGs
    png_hashes_map = {} # Store as {png_path: hash_object}
    print(f"\nCalculating hashes for {len(available_png_paths)} available PNGs from {organization_base_dir}...")
    for i, png_path in enumerate(available_png_paths):
        png_filename = os.path.basename(png_path)
        # print(f"  [{i+1}/{len(available_png_paths)}] Hashing PNG: {png_filename}...") # Verbose
        png_hash = get_image_hash(png_path, hash_size)
        if png_hash:
            png_hashes_map[png_path] = png_hash
        else:
            print(f"  Could not hash PNG: {png_filename}. It will not be matched.")

    print(f"Successfully calculated {len(png_hashes_map)} PNG hashes.")
    if not png_hashes_map and available_png_paths:
        print("Warning: None of the available PNGs could be hashed. No PNGs will be moved.")


    moved_jpg_count = 0
    moved_png_count = 0
    created_folder_count = 0

    # 4. Process each JPG
    print(f"\nProcessing {len(original_jpg_files)} JPG files...")
    for jpg_path in original_jpg_files:
        jpg_filename_with_ext = os.path.basename(jpg_path)
        jpg_filename_no_ext = os.path.splitext(jpg_filename_with_ext)[0]
        print(f"\nProcessing JPG: {jpg_filename_with_ext}")

        # Create target subfolder for this JPG
        target_jpg_subfolder = os.path.join(organization_base_dir, jpg_filename_no_ext)
        if not os.path.exists(target_jpg_subfolder):
            try:
                os.makedirs(target_jpg_subfolder)
                print(f"  Created subfolder: {target_jpg_subfolder}")
                created_folder_count +=1
            except OSError as e:
                print(f"  Error creating subfolder {target_jpg_subfolder}: {e}. Skipping this JPG.")
                continue
        else:
            print(f"  Subfolder already exists: {target_jpg_subfolder}")

        # Move the JPG into its new subfolder
        destination_jpg_path = os.path.join(target_jpg_subfolder, jpg_filename_with_ext)
        try:
            if not os.path.exists(destination_jpg_path): # Avoid error if JPG somehow already there
                shutil.move(jpg_path, destination_jpg_path)
                print(f"  Moved JPG '{jpg_filename_with_ext}' to '{target_jpg_subfolder}'")
                moved_jpg_count += 1
            else:
                print(f"  JPG '{jpg_filename_with_ext}' already exists in target subfolder. Original at '{jpg_path}' will not be moved again.")
                # If original JPG is not moved, we might want to hash the one already in the subfolder,
                # or decide on a strategy. For now, let's hash the one already in the destination.
                # Or, more simply, if we don't move it, we can hash the original path.
                # Let's assume if it's there, it's the correct one. We'll hash the one in destination.
                # For simplicity, if the JPG is already in the destination, we'll still use its hash to find PNGs.
                # The key is that the JPG *is* in that folder.
        except Exception as e:
            print(f"  Error moving JPG '{jpg_filename_with_ext}' to '{target_jpg_subfolder}': {e}. Skipping PNG matching for this JPG.")
            continue # Skip to next JPG if its own move failed

        # Calculate hash for the (now moved, or already existing) JPG
        # We should hash the JPG from its new location if moved, or its current location if not moved.
        # For consistency, let's always refer to destination_jpg_path for hashing if it exists.
        current_jpg_hash_path = destination_jpg_path if os.path.exists(destination_jpg_path) else jpg_path
        jpg_hash = get_image_hash(current_jpg_hash_path, hash_size)

        if not jpg_hash:
            print(f"  Could not generate hash for JPG '{jpg_filename_with_ext}'. Cannot match PNGs for it.")
            continue # Skip PNG matching for this JPG

        # Find and move matching PNGs from the available_png_paths list
        # Iterate over a copy of keys if modifying the dict, or manage a list of paths to remove
        png_paths_to_potentially_move = list(png_hashes_map.keys()) # Get current available PNGs

        for png_path_to_check in png_paths_to_potentially_move:
            if png_path_to_check not in png_hashes_map: # Already moved or failed to hash
                continue

            png_hash_candidate = png_hashes_map[png_path_to_check]
            hash_difference = jpg_hash - png_hash_candidate

            if hash_difference <= similarity_threshold:
                png_filename_to_move = os.path.basename(png_path_to_check)
                destination_png_path = os.path.join(target_jpg_subfolder, png_filename_to_move)
                print(f"    MATCH FOUND: JPG '{jpg_filename_with_ext}' is similar to PNG '{png_filename_to_move}' (Diff: {hash_difference})")

                try:
                    if os.path.exists(png_path_to_check): # Ensure source PNG still exists at original path
                        shutil.move(png_path_to_check, destination_png_path)
                        print(f"      Moved PNG '{png_filename_to_move}' to '{target_jpg_subfolder}'")
                        moved_png_count += 1
                        # Remove this PNG from further consideration by any JPG
                        del png_hashes_map[png_path_to_check]
                    else:
                        print(f"      Error: Source PNG '{png_filename_to_move}' no longer found at '{png_path_to_check}'. Already moved?")
                        # If it was already moved, it might have been by a previous JPG if thresholds are loose.
                        # Or if this script is re-run. We should remove it from map anyway.
                        if png_path_to_check in png_hashes_map:
                             del png_hashes_map[png_path_to_check]

                except Exception as e:
                    print(f"      Error moving PNG '{png_filename_to_move}': {e}")


    print(f"\n--- Organization Process Complete ---")
    print(f"Folders created: {created_folder_count}")
    print(f"JPGs moved: {moved_jpg_count}")
    print(f"PNGs moved: {moved_png_count}")
    remaining_pngs_in_root = [f for f in os.listdir(organization_base_dir) if os.path.isfile(os.path.join(organization_base_dir, f)) and f.lower().endswith(".png")]
    if remaining_pngs_in_root:
        print(f"PNGs remaining in the root of '{organization_base_dir}': {len(remaining_pngs_in_root)}")
        print(f"  (These may not have matched any JPGs based on the threshold, or failed to hash/move.)")
    else:
        print(f"No PNGs remaining in the root of '{organization_base_dir}'.")
    
    if os.path.exists(jpg_originals_source_dir):
        remaining_jpgs_in_source = [f for f in os.listdir(jpg_originals_source_dir) if os.path.isfile(os.path.join(jpg_originals_source_dir, f)) and f.lower().endswith((".jpg", ".jpeg"))]
        if remaining_jpgs_in_source:
            print(f"JPGs remaining in the original source directory '{jpg_originals_source_dir}': {len(remaining_jpgs_in_source)}")
            print(f"  (These may have failed to process or their subfolders could not be created.)")
        else:
             print(f"Original JPG source directory '{jpg_originals_source_dir}' is now empty of JPGs.")


    print("-----------------------------------------")

if __name__ == "__main__":
    # Ensure Pillow and ImageHash are installed:
    # pip install Pillow ImageHash
    main()

```

---
## 📄 Capitalart-Mockup-Generator/scripts/ai-jpeg-reset.sh

```sh
#!/bin/bash

#####################################################################################
#  ai-jpeg-reset.sh — Purify AI JPEGs for Fujifilm Imagine/Harvey Norman Print Labs #
# --------------------------------------------------------------------------------- #
#  Author: Robin “Robbie” Custance + ChatGPT Assist                                #
#  Last updated: 18-May-2025                                                        #
#                                                                                   #
#  DESCRIPTION:                                                                     #
#    - Converts every .jpg/.jpeg in AI-JPEG-Originals into an 8-bit TIFF, then      #
#      exports a fully standard, lab-safe JPEG into AI-JPEG-Reset.                  #
#    - Strips all metadata, forces sRGB, 8-bit, baseline, 4:2:0, 300 DPI.           #
#    - Removes all traces of Midjourney/AI segment oddities by using TIFF as a      #
#      “reset” intermediary.                                                        #
#    - Should resolve “invalid JPEG” errors on Fujifilm Imagine, Harvey Norman,     #
#      and BigW photo kiosks.                                                       #
#                                                                                   #
#  HOW TO USE:                                                                      #
#    1. Place original images in .../AI-JPEG-Originals                              #
#    2. Run this script (see below)                                                 #
#    3. Retrieve “purified” JPEGs from .../AI-JPEG-Reset                            #
#                                                                                   #
#  REQUIREMENTS: ImageMagick installed (`brew install imagemagick`)                 #
#                                                                                   #
#  USAGE:                                                                           #
#    chmod +x ai-jpeg-reset.sh                                                      #
#    ./ai-jpeg-reset.sh                                                             #
#####################################################################################

in_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/AI-JPEG-Originals"
tmp_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/AI-JPEG-TIFFS"
out_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/AI-JPEG-Reset"

mkdir -p "$tmp_dir" "$out_dir"

icc_profile="/Library/ColorSync/Profiles/sRGB2014.icc" # Use IEC61966-2.1.icc if sRGB2014 causes issues

for f in "$in_dir"/*.jpg "$in_dir"/*.jpeg; do
  [ -e "$f" ] || continue
  base=$(basename "$f" | sed 's/ /_/g')
  name="${base%.*}"

  tiff_file="$tmp_dir/${name}.tiff"
  out_file="$out_dir/${name}-RESET.jpg"

  echo "Converting $base to TIFF..."
  magick "$f" -strip -depth 8 -colorspace sRGB "$tiff_file"

  echo "Exporting $name as fresh JPEG..."
  magick "$tiff_file" \
    -strip \
    -depth 8 \
    -colorspace sRGB \
    -profile "$icc_profile" \
    -units PixelsPerInch -density 300 \
    -sampling-factor 2x2,1x1,1x1 \
    -interlace none \
    -quality 85 \
    "$out_file"

  # Optional: Remove intermediate TIFF to save space
  rm "$tiff_file"

  # Validate the output
  final_size=$(stat -f%z "$out_file")
  echo "✅ $out_file created: $((final_size/1024/1024)) MB"

done

echo "Batch conversion complete. All 'RESET' JPEGs are ready in $out_dir."

```

---
## 📄 Capitalart-Mockup-Generator/scripts/write_all_composite_generators.py

```py
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
## 📄 Capitalart-Mockup-Generator/scripts/gather_mockup_code_to_text.sh

```sh
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
