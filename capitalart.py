#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================
# CapitalArt Main Flask App (All-in-One)
# File: capitalart.py
# Maintainer: Robin Custance (Robbie Mode™)
# =========================================================

import os
import sys
import json
import uuid
import subprocess
import random
from pathlib import Path
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_from_directory, flash
)

# === [ 1. ENVIRONMENT & PATHS ] ===

BASE_DIR = Path(__file__).parent.resolve()
MOCKUPS_DIR = BASE_DIR / "inputs" / "mockups" / "4x5-categorised"
ARTWORKS_DIR = BASE_DIR / "inputs" / "artworks"
ARTWORK_PROCESSED_DIR = BASE_DIR / "outputs" / "processed"
SELECTIONS_DIR = BASE_DIR / "outputs" / "selections"
LOGS_DIR = BASE_DIR / "logs"
COMPOSITES_DIR = BASE_DIR / "outputs" / "composites"

ANALYZE_SCRIPT_PATH = BASE_DIR / "scripts" / "analyze_artwork.py"

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mockup-secret-key")

def get_categories():
    return sorted([
        folder.name for folder in MOCKUPS_DIR.iterdir()
        if folder.is_dir() and folder.name.lower() != "uncategorised"
    ])

def random_image(category):
    cat_dir = MOCKUPS_DIR / category
    images = [f.name for f in cat_dir.glob("*.png")]
    return random.choice(images) if images else None

def init_slots():
    cats = get_categories()
    session["slots"] = [{"category": c, "image": random_image(c)} for c in cats]

def compute_options(slots):
    cats = get_categories()
    return [cats for _ in slots]

def list_artworks():
    artworks = []
    for aspect_dir in sorted(ARTWORKS_DIR.iterdir()):
        if not aspect_dir.is_dir():
            continue
        for img in aspect_dir.glob("*.[jJ][pP][gG]"):
            artworks.append({
                "aspect": aspect_dir.name,
                "filename": img.name,
                "title": img.stem.replace("-", " ").title()
            })
    artworks = sorted(artworks, key=lambda x: (x["aspect"], x["filename"]))
    return artworks

@app.route("/")
def home():
    return render_template("index.html", menu=get_menu())

@app.route("/select", methods=["GET", "POST"])
def select():
    if "slots" not in session or request.args.get("reset") == "1":
        init_slots()
    slots = session["slots"]
    options = compute_options(slots)
    zipped = list(zip(slots, options))
    return render_template("mockup_selector.html", zipped=zipped, menu=get_menu())

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
    slots = session.get("slots", [])
    if not slots:
        flash("No mockups selected!", "danger")
        return redirect(url_for("select"))
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    selection_id = str(uuid.uuid4())
    selection_file = SELECTIONS_DIR / f"{selection_id}.json"
    with open(selection_file, "w") as f:
        json.dump(slots, f, indent=2)
    script_path = BASE_DIR / "scripts" / "generate_composites.py"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"composites_{selection_id}.log"
    try:
        result = subprocess.run(
            ["python3", str(script_path), str(selection_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=BASE_DIR,
        )
        with open(log_file, "w") as log:
            log.write("=== STDOUT ===\n")
            log.write(result.stdout)
            log.write("\n\n=== STDERR ===\n")
            log.write(result.stderr)
        if result.returncode == 0:
            flash("Composites generated successfully!", "success")
        else:
            flash("Composite generation failed. See logs for details.", "danger")
    except Exception as e:
        with open(log_file, "a") as log:
            log.write(f"\n\n=== Exception ===\n{str(e)}")
        flash("Error running the composite generator.", "danger")
    # After generation, show latest composites preview
    return redirect(url_for("composites_preview"))

@app.route("/review")
def review():
    slots = session.get("slots", [])
    artwork = {
        "seo_name": "tawny-frogmouth-dot-artwork-by-robin-custance-rjc-0121",
        "title": "Tawny Frogmouth Dot Artwork by Robin Custance",
        "main_image": "tawny-frogmouth-dot-artwork-by-robin-custance-rjc-0121.jpg",
        "thumb": "tawny-frogmouth-dot-artwork-by-robin-custance-rjc-0121-THUMB.jpg",
        "description": "A sample Pulitzer-worthy, SEO-optimised, heartfelt artwork description goes here. All text and tags.",
    }
    return render_template("review.html", slots=slots, artwork=artwork, menu=get_menu())

@app.route("/review/<aspect>/<filename>")
def review_artwork(aspect, filename):
    original_basename = Path(filename).stem
    processed_root = ARTWORK_PROCESSED_DIR
    listing_json = None
    seo_folder = None
    for folder in processed_root.iterdir():
        if folder.is_dir():
            possible_json = folder / f"{folder.name}-listing.json"
            if possible_json.exists():
                with open(possible_json, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if Path(data.get("filename", "")).stem == original_basename:
                            listing_json = data
                            seo_folder = folder.name
                            break
                    except Exception:
                        continue
    if not listing_json:
        return render_template(
            "review_artwork.html",
            artwork={
                "seo_name": original_basename,
                "title": original_basename.replace("-", " ").title(),
                "main_image": filename,
                "thumb": f"{original_basename}-THUMB.jpg",
                "description": "(No AI listing found. Try re-analyzing.)",
                "aspect": aspect,
                "missing": True,
                "tags": [],
                "materials": [],
                "primary_colour": "",
                "secondary_colour": "",
            },
            ai_listing=None,
            ai_description="(No AI listing description found)",
            fallback_text=None,
            tags=[],
            materials=[],
            used_fallback_naming=False,
            generic_text="",
            raw_ai_output="",
            menu=get_menu(),
        )
    ai_listing = listing_json.get("ai_listing", {})
    desc = None
    fallback_text = None
    ai_title = None
    if isinstance(ai_listing, dict):
        ai_title = ai_listing.get("title")
        desc = ai_listing.get("description")
        fallback_text = ai_listing.get("fallback_text")
    elif isinstance(ai_listing, str):
        desc = ai_listing

    if not desc and isinstance(ai_listing, dict) and "fallback_text" in ai_listing:
        fallback_text = ai_listing["fallback_text"]
        desc = fallback_text

    if not desc:
        desc = listing_json.get("generic_text", "(No AI listing description found)")

    tags = listing_json.get("tags") or (ai_listing.get("tags", []) if isinstance(ai_listing, dict) else [])
    materials = listing_json.get("materials") or (ai_listing.get("materials", []) if isinstance(ai_listing, dict) else [])

    generic_text = listing_json.get("generic_text", "")

    combined_description = desc.strip()
    if generic_text:
        combined_description += "\n\n" + generic_text.strip()

    primary_colour = listing_json.get("primary_colour", "")
    secondary_colour = listing_json.get("secondary_colour", "")
    used_fallback_naming = bool(listing_json.get("used_fallback_naming", False))
    if isinstance(ai_listing, (dict, list)):
        raw_ai_output = json.dumps(ai_listing, indent=2, ensure_ascii=False)[:800]
    else:
        raw_ai_output = str(ai_listing)[:800]
    artwork = {
        "seo_name": seo_folder,
        "title": ai_title or listing_json.get("title") or seo_folder.replace("-", " ").title(),
        "main_image": f"outputs/processed/{seo_folder}/{seo_folder}.jpg",
        "thumb": f"outputs/processed/{seo_folder}/{seo_folder}-THUMB.jpg",
        "description": combined_description.strip(),
        "aspect": aspect,
        "tags": tags,
        "materials": materials,
        "has_tags": bool(tags),
        "has_materials": bool(materials),
        "primary_colour": primary_colour,
        "secondary_colour": secondary_colour,
    }
    return render_template(
        "review_artwork.html",
        artwork=artwork,
        ai_listing=ai_listing,
        ai_description=desc,
        fallback_text=fallback_text,
        tags=tags,
        materials=materials,
        used_fallback_naming=used_fallback_naming,
        generic_text=generic_text,
        raw_ai_output=raw_ai_output,
        menu=get_menu(),
    )

@app.route("/static/outputs/processed/<seo_folder>/<filename>")
def processed_image(seo_folder, filename):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    return send_from_directory(folder, filename)

@app.route("/artwork-review")
def artwork_review():
    artworks = list_artworks()
    return render_template("artwork_review.html", artworks=artworks, menu=get_menu())

@app.route("/artwork-img/<aspect>/<filename>")
def artwork_image(aspect, filename):
    folder = ARTWORKS_DIR / aspect
    return send_from_directory(str(folder.resolve()), filename)

@app.route("/artworks")
def artworks():
    return render_template("artworks.html", artworks=list_artworks())

@app.route("/mockup-img/<category>/<filename>")
def mockup_img(category, filename):
    return send_from_directory(MOCKUPS_DIR / category, filename)

@app.route("/composite-img/<folder>/<filename>")
def composite_img(folder, filename):
    """Serve generated composite images from disk."""
    return send_from_directory(COMPOSITES_DIR / folder, filename)

@app.route("/composites-preview")
def composites_preview():
    """Show a grid preview of the most recently generated composites."""
    COMPOSITES_DIR.mkdir(parents=True, exist_ok=True)
    subdirs = [d for d in COMPOSITES_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        return render_template("composites_preview.html", images=None, menu=get_menu())
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    images = sorted([f.name for f in latest.glob("*.jpg")])
    return render_template(
        "composites_preview.html",
        images=images,
        folder=latest.name,
        menu=get_menu()
    )

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect(url_for("select"))

@app.route("/analyze/<aspect>/<filename>", methods=["POST"])
def analyze_artwork(aspect, filename):
    artwork_path = ARTWORKS_DIR / aspect / filename
    script_path = ANALYZE_SCRIPT_PATH
    log_id = str(uuid.uuid4())
    log_file = LOGS_DIR / f"analyze_{log_id}.log"
    feedback = request.form.get("feedback", "")
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cmd = ["python3", str(script_path), str(artwork_path)]
        if feedback.strip():
            print(f"[Analysis Feedback]: {feedback}")
            # Optionally, save feedback to a temp file for the script to use
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300
        )
        with open(log_file, "w") as log:
            log.write("=== STDOUT ===\n")
            log.write(result.stdout)
            log.write("\n\n=== STDERR ===\n")
            log.write(result.stderr)
        if result.returncode == 0:
            flash(f"✅ Analysis complete for {filename}", "success")
        else:
            flash(f"❌ Analysis failed for {filename}: {result.stderr}", "danger")
    except Exception as e:
        with open(log_file, "a") as log:
            log.write(f"\n\n=== Exception ===\n{str(e)}")
        flash(f"❌ Error running analysis: {str(e)}", "danger")
    return redirect(url_for("review_artwork", aspect=aspect, filename=filename))

def get_menu():
    return [
        {"name": "Mockup Selector", "url": url_for('select')},
        {"name": "Artwork Gallery", "url": url_for('artworks')},
        {"name": "Artwork Review", "url": url_for('artwork_review')},
        {"name": "Review Listing", "url": url_for('review')}
    ]

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    print(f"🎨 Starting CapitalArt UI at http://localhost:{port}/ ...")
    app.run(debug=debug, port=port)
