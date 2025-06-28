# ==================== [ capitalart_ui.py ] ====================
# Unified Flask App — CapitalArt Workflow (Robbie Mode™)
# ---------------------------------------------------------------

import os
import random
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash

BASE_DIR = Path(__file__).parent.resolve()
MOCKUPS_DIR = BASE_DIR / "inputs" / "mockups" / "4x5-categorised"
ARTWORK_DIR = BASE_DIR / "outputs" / "processed"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mockup-secret-key")

# --- [ Helper functions ] ---

def get_categories():
    """List all mockup categories (excluding Uncategorised)."""
    return sorted([
        folder.name for folder in MOCKUPS_DIR.iterdir()
        if folder.is_dir() and folder.name.lower() != "uncategorised"
    ])

def random_image(category):
    """Return random image filename for category, or None."""
    cat_dir = MOCKUPS_DIR / category
    images = [f.name for f in cat_dir.glob("*.png")]
    return random.choice(images) if images else None

def init_slots():
    """Session: one random mockup per category."""
    cats = get_categories()
    session["slots"] = [{"category": c, "image": random_image(c)} for c in cats]

def compute_options(slots):
    """Return the full category list for every slot."""
    cats = get_categories()
    return [cats for _ in slots]

# --- [ ROUTES ] ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/select", methods=["GET", "POST"])
def select():
    if "slots" not in session or request.args.get("reset") == "1":
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

@app.route("/proceed", methods=["POST"])
def proceed():
    # In real workflow: call compositing, then redirect to /review
    flash("Compositing is simulated — jump to review for now.")
    return redirect(url_for("review"))

@app.route("/review")
def review():
    # Here you’d show the artwork, all selected mockups, description, etc.
    slots = session.get("slots", [])
    # Simulate an artwork for review (replace with real data lookup)
    artwork = {
        "seo_name": "tawny-frogmouth-dot-artwork-by-robin-custance-rjc-0121",
        "title": "Tawny Frogmouth Dot Artwork by Robin Custance",
        "main_image": "tawny-frogmouth-dot-artwork-by-robin-custance-rjc-0121.jpg",
        "thumb": "tawny-frogmouth-dot-artwork-by-robin-custance-rjc-0121-THUMB.jpg",
        "description": "A sample Pulitzer-worthy, SEO-optimised, heartfelt artwork description goes here. All text and tags.",
    }
    # Your real pipeline should load this from the artwork folder’s JSON file!
    return render_template("review.html", slots=slots, artwork=artwork)

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect(url_for("select"))

# --- [ Serve static mockups easily ] ---
@app.route("/mockup-img/<category>/<filename>")
def mockup_img(category, filename):
    return send_from_directory(MOCKUPS_DIR / category, filename)

# --- [ Entry point ] ---
if __name__ == "__main__":
    app.run(debug=True, port=5050)
