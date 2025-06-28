# ============================== [ mockup_selector_ui.py ] ==============================
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
