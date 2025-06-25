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
