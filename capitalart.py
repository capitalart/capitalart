#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================
# CapitalArt Main Flask App (All-in-One, Robbie Mode™)
# File: capitalart.py
# Maintainer: Robin Custance
# =========================================================

# ========== SECTION 0. IMPORTS & ENVIRONMENT SETUP ==========

import os
import sys
import json
import uuid
import subprocess
import random
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from PIL import Image, ImageDraw
import cv2
import numpy as np

# ========== SECTION 1. PATHS & CONSTANTS ==========

BASE_DIR = Path(__file__).parent.resolve()
MOCKUPS_DIR = BASE_DIR / "inputs" / "mockups" / "4x5-categorised"
ARTWORKS_DIR = BASE_DIR / "inputs" / "artworks"
ARTWORK_PROCESSED_DIR = BASE_DIR / "outputs" / "processed"
SELECTIONS_DIR = BASE_DIR / "outputs" / "selections"
LOGS_DIR = BASE_DIR / "logs"
COMPOSITES_DIR = BASE_DIR / "outputs" / "composites"
FINALISED_DIR = BASE_DIR / "outputs" / "finalised-artwork"
COORDS_ROOT = BASE_DIR / "inputs" / "coords"
ANALYZE_SCRIPT_PATH = BASE_DIR / "scripts" / "analyze_artwork.py"
GENERATE_SCRIPT_PATH = BASE_DIR / "scripts" / "generate_composites.py"

# ========== SECTION 2. FLASK APP INITIALISATION ==========

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mockup-secret-key")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR / "composites-workflow.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ========== SECTION 3. UTILITY FUNCTIONS ==========

## 3.1. Get sorted mockup categories
def get_categories():
    return sorted([folder.name for folder in MOCKUPS_DIR.iterdir() if folder.is_dir() and folder.name.lower() != "uncategorised"])

## 3.2. Get random image for a category
def random_image(category):
    cat_dir = MOCKUPS_DIR / category
    images = [f.name for f in cat_dir.glob("*.png")]
    return random.choice(images) if images else None

## 3.3. Initialise slots in session
def init_slots():
    cats = get_categories()
    session["slots"] = [{"category": c, "image": random_image(c)} for c in cats]

## 3.4. Compute options (categories) for mockup slots
def compute_options(slots):
    cats = get_categories()
    return [cats for _ in slots]

## 3.5. Resize image for preview or transformation
def resize_image_for_long_edge(image: Image.Image, target_long_edge: int = 2000) -> Image.Image:
    width, height = image.size
    if width > height:
        new_width = target_long_edge
        new_height = int(height * (target_long_edge / width))
    else:
        new_height = target_long_edge
        new_width = int(width * (target_long_edge / height))
    return image.resize((new_width, new_height), Image.LANCZOS)

## 3.6. Apply perspective transform (for composite generation)
def apply_perspective_transform(art_img: Image.Image, mockup_img: Image.Image, dst_coords: list) -> Image.Image:
    w, h = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask)
    return composite

## 3.7. Find most recent composite output folder
def latest_composite_folder() -> str | None:
    latest_time = 0
    latest_folder = None
    for folder in ARTWORK_PROCESSED_DIR.iterdir():
        if not folder.is_dir():
            continue
        images = list(folder.glob("*-MU-*.jpg"))
        if not images:
            continue
        recent = max(images, key=lambda p: p.stat().st_mtime)
        if recent.stat().st_mtime > latest_time:
            latest_time = recent.stat().st_mtime
            latest_folder = folder.name
    return latest_folder

## 3.8. List all artworks available for analysis
def list_artworks():
    artworks = []
    for aspect_dir in sorted(ARTWORKS_DIR.iterdir()):
        if not aspect_dir.is_dir():
            continue
        for img in aspect_dir.glob("*.[jJ][pP][gG]"):
            artworks.append(
                {"aspect": aspect_dir.name, "filename": img.name, "title": img.stem.replace("-", " ").title()}
            )
    artworks = sorted(artworks, key=lambda x: (x["aspect"], x["filename"]))
    return artworks

# ========== SECTION 4. MAIN ROUTES ==========

# --- 4.1. Home and Artwork Gallery ---

@app.route("/")
def home():
    return render_template("index.html", menu=get_menu())

@app.route("/artworks")
def artworks():
    return render_template("artworks.html", artworks=list_artworks(), menu=get_menu())

# --- 4.2. Mockup Selector UI ---

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
    log_file = LOGS_DIR / f"composites_{selection_id}.log"
    try:
        result = subprocess.run(
            ["python3", str(GENERATE_SCRIPT_PATH), str(selection_file)],
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

    latest = latest_composite_folder()
    if latest:
        session["latest_seo_folder"] = latest
        logging.info("Generated composites for %s", latest)
        return redirect(url_for("composites_specific", seo_folder=latest))
    return redirect(url_for("composites_preview"))

# --- 4.3. Analyze & Generate Composites (Combined) ---

@app.route("/analyze/<aspect>/<filename>", methods=["POST"])
def analyze_artwork(aspect, filename):
    """
    1. Run AI artwork analysis (analyze_artwork.py)
    2. Trigger generate_composites.py immediately after
    3. Redirect to review page with listing & preview mockups
    """
    # === Run Analysis ===
    artwork_path = ARTWORKS_DIR / aspect / filename
    log_id = str(uuid.uuid4())
    log_file = LOGS_DIR / f"analyze_{log_id}.log"
    try:
        cmd = ["python3", str(ANALYZE_SCRIPT_PATH), str(artwork_path)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        with open(log_file, "w") as log:
            log.write("=== STDOUT ===\n")
            log.write(result.stdout)
            log.write("\n\n=== STDERR ===\n")
            log.write(result.stderr)
        if result.returncode != 0:
            flash(f"❌ Analysis failed for {filename}: {result.stderr}", "danger")
            return redirect(url_for("artworks"))
    except Exception as e:
        with open(log_file, "a") as log:
            log.write(f"\n\n=== Exception ===\n{str(e)}")
        flash(f"❌ Error running analysis: {str(e)}", "danger")
        return redirect(url_for("artworks"))

    # === Find SEO folder just generated ===
    # Look for the folder in outputs/processed/ that matches the new listing
    original_basename = Path(filename).stem
    processed_root = ARTWORK_PROCESSED_DIR
    seo_folder = None
    for folder in processed_root.iterdir():
        if folder.is_dir():
            possible_json = folder / f"{folder.name}-listing.json"
            if possible_json.exists():
                with open(possible_json, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if Path(data.get("filename", "")).stem == original_basename:
                            seo_folder = folder.name
                            break
                    except Exception:
                        continue
    if not seo_folder:
        flash("Analysis complete, but no SEO folder/listing found.", "warning")
        return redirect(url_for("artworks"))

    # === Run Composite Generation ===
    try:
        cmd = ["python3", str(GENERATE_SCRIPT_PATH), seo_folder]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=BASE_DIR, timeout=600)
        composite_log_file = LOGS_DIR / f"composite_gen_{log_id}.log"
        with open(composite_log_file, "w") as log:
            log.write("=== STDOUT ===\n")
            log.write(result.stdout)
            log.write("\n\n=== STDERR ===\n")
            log.write(result.stderr)
        if result.returncode != 0:
            flash("Artwork analyzed, but mockup generation failed. See logs.", "danger")
    except Exception as e:
        flash(f"Composites generation error: {e}", "danger")

    # === Redirect to review page showing both listing and preview mockups ===
    return redirect(url_for("review_artwork", aspect=aspect, filename=filename))

# --- 4.4. Review Artwork, Preview Mockups & AI Listing ---

@app.route("/review/<aspect>/<filename>")
def review_artwork(aspect, filename):
    # [The function body remains almost as before, fetching the listing and displaying mockups...]
    # ... (for brevity, copy your current robust implementation here)
    # For full context: See previous assistant responses, or paste your current function here.
    # (Let me know if you want it rewritten or improved/sectioned!)

    # ---[omitted for brevity]---
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
    # Fallback dummy if not found
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
            mockup_previews=[],
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
    materials = listing_json.get("materials") or (
        ai_listing.get("materials", []) if isinstance(ai_listing, dict) else []
    )
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
    # --- Collect preview mockups
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    preview_images = sorted(folder.glob(f"{seo_folder}-MU-*.jpg"))
    mockup_previews = [f"outputs/processed/{seo_folder}/{img.name}" for img in preview_images]
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
        mockup_previews=mockup_previews,
        menu=get_menu(),
    )

# --- 4.5. Static & Utility Routes ---

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

@app.route("/mockup-img/<category>/<filename>")
def mockup_img(category, filename):
    return send_from_directory(MOCKUPS_DIR / category, filename)

@app.route("/composite-img/<folder>/<filename>")
def composite_img(folder, filename):
    return send_from_directory(COMPOSITES_DIR / folder, filename)

@app.route("/composites-preview")
def composites_preview():
    latest = latest_composite_folder()
    if not latest:
        return render_template("composites_preview.html", images=None, menu=get_menu())
    return redirect(url_for("composites_specific", seo_folder=latest))

@app.route("/composites/<seo_folder>")
def composites_specific(seo_folder):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    if not folder.exists():
        flash("Artwork folder not found", "danger")
        return redirect(url_for("composites_preview"))
    images = sorted(folder.glob(f"{seo_folder}-MU-*.jpg"))
    listing = None
    listing_file = folder / f"{seo_folder}-listing.json"
    if listing_file.exists():
        try:
            with open(listing_file, "r", encoding="utf-8") as f:
                listing = json.load(f)
        except Exception as e:
            logging.error("Failed reading %s: %s", listing_file, e)
    display_images = []
    mockups = listing.get("mockups", []) if isinstance(listing, dict) else []
    for idx, img in enumerate(images):
        cat = None
        if idx < len(mockups):
            cat = Path(mockups[idx]).parent.name
        display_images.append(
            {
                "filename": img.name,
                "category": cat,
                "index": idx,
            }
        )
    return render_template(
        "composites_preview.html",
        images=display_images,
        seo_folder=seo_folder,
        listing=listing,
        menu=get_menu(),
    )

# --- 4.6. Composite Regeneration & Approval ---

@app.route("/regenerate_composite/<seo_folder>/<int:slot_index>", methods=["POST"])
def regenerate_composite(seo_folder, slot_index):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    listing_path = folder / f"{seo_folder}-listing.json"
    if not listing_path.exists():
        flash("Listing metadata missing", "danger")
        return redirect(url_for("composites_specific", seo_folder=seo_folder))
    with open(listing_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mockups = data.get("mockups", [])
    if slot_index < 0 or slot_index >= len(mockups):
        flash("Invalid slot", "danger")
        return redirect(url_for("composites_specific", seo_folder=seo_folder))
    mockup_path = Path(mockups[slot_index])
    aspect = data.get("aspect_ratio")
    coords_path = COORDS_ROOT / aspect / f"{mockup_path.stem}.json"
    art_path = folder / f"{seo_folder}.jpg"
    output_path = folder / f"{seo_folder}-MU-{slot_index+1:02d}.jpg"
    try:
        with open(coords_path, "r", encoding="utf-8") as cf:
            c = json.load(cf)["corners"]
        dst = [[c[0]["x"], c[0]["y"]], [c[1]["x"], c[1]["y"]], [c[3]["x"], c[3]["y"]], [c[2]["x"], c[2]["y"]]]
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image_for_long_edge(art_img)
        mock_img = Image.open(mockup_path).convert("RGBA")
        composite = apply_perspective_transform(art_img, mock_img, dst)
        composite.convert("RGB").save(output_path, "JPEG", quality=85)
        logging.info("Regenerated composite %s slot %s", seo_folder, slot_index)
        flash("Composite regenerated", "success")
    except Exception as e:
        logging.error("Error regenerating composite %s slot %s: %s", seo_folder, slot_index, e)
        flash("Failed to regenerate composite", "danger")
    return redirect(url_for("composites_specific", seo_folder=seo_folder))

@app.route("/approve_composites/<seo_folder>", methods=["POST"])
def approve_composites(seo_folder):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    listing_path = folder / f"{seo_folder}-listing.json"
    if not listing_path.exists():
        flash("Listing not found", "danger")
        return redirect(url_for("composites_specific", seo_folder=seo_folder))
    with open(listing_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["composites_approved"] = True
    with open(listing_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    final_root = FINALISED_DIR / seo_folder
    final_root.mkdir(parents=True, exist_ok=True)
    for img in folder.glob(f"{seo_folder}-MU-*.jpg"):
        shutil.copy2(img, final_root / img.name)
    shutil.copy2(listing_path, final_root / listing_path.name)
    shutil.copy2(folder / f"{seo_folder}.jpg", final_root / f"{seo_folder}.jpg")
    shutil.make_archive(str(final_root), "zip", final_root)
    logging.info("Approved composites for %s", seo_folder)
    flash("Composites approved and finalised", "success")
    return redirect(url_for("composites_specific", seo_folder=seo_folder))

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect(url_for("select"))

# ========== SECTION 5. MENU FUNCTION ==========

def get_menu():
    return [
        {"name": "Mockup Selector", "url": url_for("select")},
        {"name": "Artwork Gallery", "url": url_for("artworks")},
        {"name": "Artwork Review", "url": url_for("artwork_review")},
    ]

# ========== SECTION 6. APP RUNNER ==========

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    print(f"🎨 Starting CapitalArt UI at http://localhost:{port}/ ...")
    app.run(debug=debug, port=port)

