# 🧠 CapitalArt Snapshot — REPORTS-27-JUN-2025-11-19PM


---
## 📄 mockup_selector_ui.py

```py
import os
import random
from pathlib import Path
from typing import List, Dict

from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

# --- [1. CONFIGURATION] ---
BASE_DIR = Path("inputs/mockups/4x5-categorised")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mockup-secret-key")

# --- [2. Serve Mockup Images] ---
@app.route("/mockup-img/<category>/<filename>")
def mockup_img(category, filename):
    folder = BASE_DIR / category
    return send_from_directory(str(folder.resolve()), filename)

# --- [3. Category Helpers] ---
def get_categories() -> List[str]:
    return sorted([f.name for f in BASE_DIR.iterdir() if f.is_dir()])

def random_image(category: str) -> str | None:
    folder = BASE_DIR / category
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images += list(folder.glob(ext))
    return random.choice(images).name if images else None

def init_slots():
    """One slot per category; each slot starts with one random image."""
    categories = get_categories()
    session["slots"] = [{"category": c, "image": random_image(c)} for c in categories]

def compute_options(slots: List[Dict]) -> List[List[str]]:
    """Ultimate freedom: for each slot, all categories always available."""
    all_cats = get_categories()
    return [all_cats for _ in slots]

# --- [4. Routes] ---
@app.route("/select", methods=["GET"])
def select():
    if "slots" not in session:
        init_slots()
    slots = session["slots"]
    options = compute_options(slots)
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
    return redirect(url_for("select"))

@app.route("/swap", methods=["POST"])
def swap():
    slot_idx = int(request.form["slot"])
    new_cat = request.form["new_category"]
    slots = session.get("slots", [])
    if 0 <= slot_idx < len(slots):
        slots[slot_idx]["category"] = new_cat
        slots[slot_idx]["image"] = random_image(new_cat)
        session["slots"] = slots
    return redirect(url_for("select"))

@app.route("/reset", methods=["POST"])
def reset():
    init_slots()
    return redirect(url_for("select"))

if __name__ == "__main__":
    app.run(debug=True, port=5050)
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

# Folders or files to ignore (case insensitive)
IGNORE_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "env", ".idea", ".DS_Store",
    "node_modules", ".nojekyll", ".pytest_cache", ".mypy_cache"
}

# File extensions to ignore (add as needed, e.g., '.log', '.tmp')
IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".swp"
}

# ============================== [ HELPER FUNCTION ] ==============================

def should_ignore(entry):
    # Ignore by exact name
    if entry in IGNORE_NAMES:
        return True
    # Ignore by extension
    _, ext = os.path.splitext(entry)
    if ext in IGNORE_EXTENSIONS:
        return True
    return False

def generate_tree(start_path: str, prefix: str = "") -> str:
    tree_str = ""
    try:
        entries = sorted(os.listdir(start_path))
    except PermissionError:
        # Just in case you hit a protected dir (unlikely in your use)
        return tree_str
    entries = [e for e in entries if not should_ignore(e)]

    for idx, entry in enumerate(entries):
        full_path = os.path.join(start_path, entry)
        connector = "└── " if idx == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry}\n"

        if os.path.isdir(full_path) and not should_ignore(entry):
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
{% extends "main.html" %}
{% block title %}CapitalArt Home{% endblock %}
{% block content %}
<div class="home-hero">
  <h1>🎨 Welcome to CapitalArt Listing Machine</h1>
  <p style="font-size:1.15em; max-width:600px; margin:1em auto 2em auto;">
    G’day! This is your one-stop hub for prepping, previewing, and perfecting your artwork listings, mockups, and all things gallery magic.  
    <br><br>
    <strong>Workflow:</strong> Select your dream mockup lineup, review the listing with full Pulitzer-worthy description, and get everything export-ready for Etsy, Sellbrite, or wherever your art’s headed.
  </p>
</div>
<div class="home-actions" style="display:flex;flex-wrap:wrap;justify-content:center;gap:2em;">
  <a href="{{ url_for('select') }}" class="composite-btn" style="min-width:200px;text-align:center;">🖼️ Start Mockup Selection</a>
  <a href="{{ url_for('review') }}" class="composite-btn" style="background:#666;">🔎 Review Latest Listing</a>
</div>
<section style="max-width:700px;margin:3em auto 0 auto;text-align:left;">
  <h2>How It Works</h2>
  <ol style="font-size:1.08em;line-height:1.6;">
    <li><b>Mockup Selector:</b> Pick one hero image from each room/category. Regenerate or swap till you love the lineup.</li>
    <li><b>Review:</b> See all chosen mockups, the main artwork, and your custom AI-powered listing description in one tidy spot.</li>
    <li><b>Approval & Export:</b> When you’re happy, lock it in for final export—ready for uploading and selling. (Export coming soon!)</li>
  </ol>
</section>
{% endblock %}

```

---
## 📄 templates/mockup_selector.html

```html
{% extends "main.html" %}
{% block title %}Select Mockups | CapitalArt{% endblock %}
{% block content %}
<h1>🖼️ Select Your Mockup Lineup</h1>
<div class="grid">
  {% for slot, options in zipped %}
  <div class="item">
    {% if slot.image %}
      <img src="{{ url_for('mockup_img', category=slot.category, filename=slot.image) }}" alt="{{ slot.category }}" />
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
        <option value="{{ c }}" {% if c == slot.category %}selected{% endif %}>{{ c }}</option>
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
{% endblock %}

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
   CapitalArt Mockup Selector & Approval UI
   Full Style Sheet — Robbie Mode™
   ============================== */

/* --------- [ 0. Global Styles & Variables ] --------- */
:root {
  --main-bg: #f9f9f9;
  --main-txt: #222;
  --accent: #007acc;
  --accent-dark: #005fa3;
  --border: #ddd;
  --card-bg: #fff;
  --shadow: 0 2px 6px rgba(0,0,0,0.06);
  --radius: 8px;
  --thumb-radius: 5px;
  --menu-height: 64px;
  --gallery-gap: 2em;
}

body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  background: var(--main-bg);
  color: var(--main-txt);
  margin: 0;
  padding: 0;
  min-height: 100vh;
}

/* --------- [ 1. Header/Menu/Nav ] --------- */
header, nav {
  background: var(--accent);
  color: #fff;
  height: var(--menu-height);
  display: flex;
  align-items: center;
  padding: 0 2em;
  font-size: 1.08em;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
nav a {
  color: #fff;
  text-decoration: none;
  margin-right: 2em;
  font-weight: 500;
  letter-spacing: 0.01em;
  transition: color 0.2s;
}
nav a:hover,
nav a.active {
  color: #ffe873;
}
.logo {
  font-size: 1.22em;
  font-weight: bold;
  margin-right: 2.5em;
  letter-spacing: 0.04em;
}

/* --------- [ 2. Main Layout ] --------- */
main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5em 1em 2em 1em;
}
@media (max-width: 700px) {
  main { padding: 1.1em 0.4em; }
}

/* --------- [ 3. Gallery/Grid View ] --------- */
.grid,
#gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--gallery-gap);
  margin-bottom: 2em;
  padding: 1em 0;
}
@media (max-width: 600px) {
  .grid, #gallery {
    grid-template-columns: 1fr;
    gap: 1.1em;
  }
}

.item, .gallery-item {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1em;
  box-shadow: var(--shadow);
  text-align: center;
  transition: transform 0.18s cubic-bezier(.5,1.4,.6,.9), box-shadow 0.18s;
  position: relative;
}
.item:hover, .gallery-item:hover {
  transform: scale(1.027);
  box-shadow: 0 8px 22px rgba(0,0,0,0.10);
  z-index: 3;
}
.item img, .gallery-item img {
  max-width: 100%;
  height: auto;
  border-radius: var(--thumb-radius);
  margin-bottom: 0.7em;
  box-shadow: 0 1px 6px rgba(0,0,0,0.09);
  background: #eee;
  cursor: pointer;
  transition: box-shadow 0.15s;
}
.item img:focus,
.gallery-item img:focus {
  outline: 2.5px solid var(--accent);
}

/* --------- [ 4. Approval/Action Buttons ] --------- */
.btn,
button,
select {
  font-size: 1em;
  padding: 0.47em 1.1em;
  border: 1px solid #bbb;
  border-radius: 4px;
  margin: 0.15em 0.12em;
  cursor: pointer;
  background: #f7f7f7;
  transition: background 0.2s, border 0.18s;
}
.btn-approve { background: #5cb85c; color: #fff; border: none; }
.btn-reject { background: #ef4e4e; color: #fff; border: none; }
.btn-fullscreen { background: var(--accent); color: #fff; border: none; }
.btn:hover,
button:hover,
select:hover,
.btn-approve:hover { background: #d1f8da; }
.btn-reject:hover { background: #ffd1d1; color: #b91c1c; }
.btn-fullscreen:hover { background: var(--accent-dark); }

.composite-btn {
  display: block;
  margin: 3em auto 0;
  padding: 0.75em 2em;
  font-size: 1.09em;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 7px;
  cursor: pointer;
  font-weight: 600;
  box-shadow: 0 1px 6px rgba(0,0,0,0.09);
  transition: background 0.22s;
}
.composite-btn:hover { background: var(--accent-dark); }

input[type="checkbox"] {
  transform: scale(1.2);
  margin-right: 0.5em;
}

/* --------- [ 5. Description Panel (Etsy Style) ] --------- */
.desc-panel {
  background: #fafaff;
  border: 1.5px solid #dde1e9;
  border-radius: 8px;
  padding: 1.7em 2em;
  color: #232324;
  font-family: "Graphik Webfont", -apple-system, "Helvetica Neue", "Droid Sans", Arial, sans-serif;
  font-size: 1.08em;
  line-height: 1.7;
  max-width: 470px;
  margin: 2em auto 1em auto;
  overflow-x: auto;
  transition: box-shadow 0.15s;
  box-shadow: 0 2px 7px rgba(0,0,0,0.04);
}
.desc-panel h2 {
  font-size: 1.24em;
  color: var(--accent);
  margin-top: 0;
}
.desc-panel .expand-toggle {
  color: #888;
  font-size: 0.95em;
  margin-left: 0.7em;
  cursor: pointer;
  user-select: none;
  text-decoration: underline;
}
.desc-panel .expand-toggle:hover { color: var(--accent); }
.desc-short { max-height: 160px; overflow: hidden; position: relative; }
.desc-short::after {
  content: '...';
  position: absolute; right: 0; bottom: 0; background: #fafaff; padding: 0 0.4em;
}
@media (max-width: 600px) {
  .desc-panel { max-width: 97vw; padding: 1em 0.7em; font-size: 1em; }
}

/* --------- [ 6. Modal/Fullscreen Image View ] --------- */
.modal-bg {
  display: none;
  position: fixed; z-index: 99;
  left: 0; top: 0; width: 100vw; height: 100vh;
  background: rgba(34,34,34,0.68);
}
.modal-bg.active { display: flex; align-items: center; justify-content: center; }
.modal-img {
  background: #fff;
  border-radius: 11px;
  padding: 0.8em;
  max-width: 94vw;
  max-height: 93vh;
  box-shadow: 0 5px 26px rgba(0,0,0,0.22);
}
.modal-img img {
  max-width: 88vw;
  max-height: 80vh;
  border-radius: 7px;
}
.modal-close {
  position: absolute;
  top: 2.3vh;
  right: 2.6vw;
  font-size: 2em;
  color: #fff;
  background: none;
  border: none;
  cursor: pointer;
  z-index: 101;
  text-shadow: 0 2px 6px #000;
}
.modal-close:focus { outline: 2px solid #ffe873; }

/* --------- [ 7. Footer ] --------- */
footer, .gallery-footer {
  text-align: center;
  margin-top: 4em;
  padding: 1.2em 0;
  font-size: 1em;
  color: #777;
  background: #f2f2f2;
  border-top: 1px solid #ececec;
  letter-spacing: 0.01em;
}
footer a { color: var(--accent); text-decoration: underline; }
footer a:hover { color: var(--accent-dark); }

/* --------- [ 8. Light/Dark Mode Ready (toggle with class .dark) ] --------- */
body.dark, .dark main {
  background: #191e23 !important;
  color: #f1f1f1 !important;
}
body.dark header, body.dark nav {
  background: #14171a;
  color: #eee;
}
body.dark .item, body.dark .gallery-item,
body.dark .desc-panel, body.dark .modal-img {
  background: #252b30;
  color: #eaeaea;
  border-color: #444;
}
body.dark .desc-panel { box-shadow: 0 3px 10px rgba(0,0,0,0.33); }
body.dark .gallery-footer, body.dark footer {
  background: #1a1a1a;
  color: #bbb;
  border-top: 1px solid #252b30;
}

/* --------- [ 9. Accessibility/Print/Safe Tweaks ] --------- */
:focus-visible {
  outline: 2.2px solid #ffa52a;
  outline-offset: 1.5px;
}
@media print {
  header, nav, .composite-btn, .btn, button, select, .gallery-footer, footer { display: none !important; }
  .desc-panel { border: none !important; box-shadow: none !important; }
  body { background: #fff !important; color: #222 !important; }
  main { padding: 0 !important; }
}

/* --------- [ 10. Misc — Spacing, Inputs, Forms ] --------- */
form { margin: 0.4em 0; }
label { display: inline-block; margin-bottom: 0.2em; font-weight: 500; }
input, textarea {
  border: 1px solid #bbb;
  border-radius: 4px;
  padding: 0.3em 0.55em;
  font-size: 1em;
  background: #fff;
  color: #232324;
}
input:focus, textarea:focus { border-color: var(--accent); }

::-webkit-scrollbar {
  width: 9px; background: #eee; border-radius: 5px;
}
::-webkit-scrollbar-thumb {
  background: #ccc; border-radius: 7px;
}
::-webkit-scrollbar-thumb:hover { background: #aaa; }

/* ----- End CapitalArt Approval UI Stylesheet — Robbie Mode™ ----- */

```
