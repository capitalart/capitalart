# 🧠 CapitalArt Snapshot — REPORTS-26-JUN-2025-02-39PM


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
## 📄 capitalart-total-nuclear.py

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 📦 CapitalArt Project Utility Toolkit
# 🔧 FILE: capitalart-total-nuclear.py
# 🧠 RULESET: Robbie's Rules™ — Include everything that matters, exclude fluff
# ===========================================

# --- [ 1a: Standard Library Imports | total-nuclear-1a ] ---
import os
import sys
import datetime
import subprocess
from pathlib import Path

# --- [ 1b: Typing Imports | total-nuclear-1b ] ---
from typing import List, Generator

# ===========================================
# 2. ⏰ Timestamp Utility
# ===========================================

def get_timestamp() -> str:
    """Returns formatted timestamp for folder and filenames."""
    return datetime.datetime.now().strftime("REPORTS-%d-%b-%Y-%I-%M%p").upper()

# ===========================================
# 3. 📁 Directory Management
# ===========================================

def create_reports_folder() -> Path:
    """Creates the timestamped reports folder."""
    timestamp = get_timestamp()
    folder_path = Path("reports") / timestamp
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created report folder: {folder_path}")
    return folder_path

# ===========================================
# 4. 📂 Target File Inclusion Rules
# ===========================================

INCLUDE_PATHS = [
    Path("mockup_selector_ui.py"),
    Path("mockup_categoriser.py"),
    Path("main.py"),
    Path("generate_folder_tree.py"),
    Path("capitalart-total-nuclear.py"),
    Path("requirements.txt"),
    Path("templates"),
    Path("static"),
    Path("scripts"),
]

ALLOWED_EXTENSIONS = [".py", ".sh", ".jsx", ".txt", ".html", ".js", ".css"]

def get_included_files() -> Generator[Path, None, None]:
    """
    Yields files from INCLUDE_PATHS that match allowed extensions.
    """
    for path in INCLUDE_PATHS:
        if path.is_file() and path.suffix in ALLOWED_EXTENSIONS:
            yield path
        elif path.is_dir():
            for file in path.rglob("*"):
                if file.is_file() and file.suffix in ALLOWED_EXTENSIONS:
                    yield file

# ===========================================
# 5. 📜 Code Snapshot Generator
# ===========================================

def gather_code_snapshot(folder: Path) -> Path:
    """Creates a snapshot report of included files."""
    md_path = folder / f"report_code_snapshot_{folder.name.lower()}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# 🧠 CapitalArt Snapshot — {folder.name}\n\n")
        for file in get_included_files():
            rel_path = file.relative_to(Path("."))
            md_file.write(f"\n---\n## 📄 {rel_path}\n\n```{file.suffix[1:]}\n")
            try:
                with open(file, "r", encoding="utf-8") as f:
                    md_file.write(f.read())
            except Exception as e:
                md_file.write(f"⚠️ Could not read file: {e}")
            md_file.write("\n```\n")
    print(f"✅ Snapshot saved to: {md_path}")
    return md_path

# ===========================================
# 6. 🧪 Dependency Check Utilities
# ===========================================

def show_dependency_issues():
    """Prints pip check and pip list --outdated results to terminal."""
    print("\n🔍 Running pip check...")
    subprocess.run(["pip", "check"])

    print("\n📦 Outdated packages:")
    subprocess.run(["pip", "list", "--outdated"])

# ===========================================
# 7. 🚀 Main Entry Point
# ===========================================

def main():
    print("🧠 Starting CapitalArt Report Generator...")
    reports_folder = create_reports_folder()
    gather_code_snapshot(reports_folder)
    show_dependency_issues()
    print("🎉 All done! Snapshot complete and system health checked.")

# ===========================================
# 🔚 Run Script
# ===========================================

if __name__ == "__main__":
    main()

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
## 📄 scripts/analyze.js

```js

```
