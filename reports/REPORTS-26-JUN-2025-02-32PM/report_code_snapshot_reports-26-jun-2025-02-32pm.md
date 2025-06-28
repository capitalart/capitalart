# 🧠 CapitalArt Targeted Snapshot — REPORTS-26-JUN-2025-02-32PM


---
## 📄 mockup_selector_ui.py

```python
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

```python
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

```python
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

```python
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
