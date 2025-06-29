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
import re # Make sure 're' is imported here
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
COORDS_ROOT = BASE_DIR / "inputs" / "Coordinates"
ANALYZE_SCRIPT_PATH = BASE_DIR / "scripts" / "analyze_artwork.py"
GENERATE_SCRIPT_PATH = BASE_DIR / "scripts" / "generate_composites.py"

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disables the warning (careful: disables protection)

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

## 3.8. Find most recent analyzed artwork
def latest_analyzed_artwork() -> dict | None:
    latest_time = 0
    latest_info = None
    for folder in ARTWORK_PROCESSED_DIR.iterdir():
        if not folder.is_dir():
            continue
        listing = folder / f"{folder.name}-listing.json"
        if not listing.exists():
            continue
        t = listing.stat().st_mtime
        if t > latest_time:
            latest_time = t
            try:
                with open(listing, "r", encoding="utf-8") as f:
                    data = json.load(f)
                latest_info = {
                    "aspect": data.get("aspect_ratio"),
                    "filename": data.get("filename"),
                }
            except Exception:
                continue
    return latest_info

# =========================================================
# SECTION 3.9: Description Text Combining & Cleaning
# =========================================================

## 3.9.1. Clean Display Text (Collapse Excess Newlines)
def clean_display_text(text: str) -> str:
    """
    Cleans a string for display by:
    - Stripping leading/trailing whitespace.
    - Collapsing any sequence of two or more newlines down to exactly two.
      (So there is never more than a single blank line between paragraphs.)
    """
    if not text:
        return ""
    # Strip all leading/trailing whitespace, including spaces and newlines
    cleaned = text.strip()
    # Replace sequences of 2 or more newlines with exactly two newlines (one blank line)
    cleaned = re.sub(r'\n{2,}', '\n\n', cleaned)
    return cleaned

## 3.9.2. Combine All Listing Parts for Display
def build_full_listing_text(ai_desc: str, generic_text: str) -> str:
    """
    Combines all listing description components for clean display.
    - Cleans each part individually.
    - Joins them with a standard double newline (paragraph break).
    - Cleans the final result again, ensuring no excess blank lines.
    """
    # Clean individual parts (avoids inherited trailing newlines)
    parts = [clean_display_text(ai_desc), clean_display_text(generic_text)]
    # Filter out empty sections
    combined = "\n\n".join([p for p in parts if p])
    # FINAL CLEAN: Collapse any accidental extra lines introduced by join
    return clean_display_text(combined)

# ========== SECTION 4. MAIN ROUTES ==========

# --- 4.1. Home and Artwork Gallery ---

@app.route("/")
def home():
    latest = latest_analyzed_artwork()
    return render_template("index.html", menu=get_menu(), latest_artwork=latest)

@app.route("/artworks")
def artworks():
    """Gallery page showing ready and processed artworks."""
    processed, processed_names = list_processed_artworks()
    ready = list_ready_to_analyze(processed_names)
    return render_template(
        "artworks.html",
        ready_artworks=ready,
        processed_artworks=processed,
        menu=get_menu(),
    )

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

# =============================================================
# 4.4. REVIEW ARTWORK, PREVIEW MOCKUPS & AI LISTING
# =============================================================

@app.route("/review/<aspect>/<filename>")
def review_artwork(aspect, filename):
    """
    SECTION 4.4: Review analyzed artwork, show AI-generated listing,
    and preview composite mockups (with per-mockup regenerate and category swap UI).
    """
    # --- 4.4.1. Locate listing JSON & canonical SEO folder ---
    seo_folder = Path(filename).stem
    listing_path = ARTWORK_PROCESSED_DIR / seo_folder / f"{seo_folder}-listing.json"
    if not listing_path.exists():
        # Fallback: Show error if not found
        flash(f"Artwork listing not found: {listing_path}", "danger")
        return render_template(
            "review_artwork.html",
            artwork={
                "seo_name": seo_folder,
                "title": seo_folder.replace("-", " ").title(),
                "main_image": filename,
                "thumb": f"{seo_folder}-THUMB.jpg",
                "aspect": aspect,
                "missing": True,
                "tags": [],
                "materials": [],
                "primary_colour": "",
                "secondary_colour": "",
                "full_listing_text": "(No AI listing description found. Try re-analyzing.)"
            },
            mockup_previews=[],
            categories=[],
            ai_listing=None,
            ai_description="(No AI listing description found)",
            fallback_text=None,
            used_fallback_naming=False,
            generic_text="",
            raw_ai_output="",
            menu=get_menu(),
        )

    # --- 4.4.2. Load and parse listing JSON ---
    try:
        with open(listing_path, "r", encoding="utf-8") as f:
            listing_json = json.load(f)
    except Exception as e:
        app.logger.error(f"Error loading artwork {filename}: {e}")
        flash(f"Error loading artwork: {e}", "danger")
        return redirect(url_for("artworks"))

    # --- 4.4.3. Extract all fields for display ---
    ai_listing = listing_json.get("ai_listing", {})
    parsed_fallback = {}
    fallback_text = None
    if isinstance(ai_listing, dict):
        fallback_text = ai_listing.get("fallback_text")
        if fallback_text:
            try:
                parsed_fallback = json.loads(fallback_text)
            except Exception:
                parsed_fallback = {}
    desc = (
        parsed_fallback.get("description")
        or (ai_listing.get("description") if isinstance(ai_listing, dict) else None)
        or listing_json.get("generic_text", "")
    )
    ai_title = (
        parsed_fallback.get("title")
        or (ai_listing.get("title") if isinstance(ai_listing, dict) else None)
        or listing_json.get("title")
    )
    tags = (
        listing_json.get("tags")
        or (ai_listing.get("tags") if isinstance(ai_listing, dict) else None)
        or parsed_fallback.get("tags")
        or []
    )
    materials = (
        listing_json.get("materials")
        or (ai_listing.get("materials") if isinstance(ai_listing, dict) else None)
        or parsed_fallback.get("materials")
        or []
    )
    generic_text = listing_json.get("generic_text", "")

    # --- 4.4.4. Combine & clean description blocks ---
    full_listing_text = build_full_listing_text(desc, generic_text)

    # --- 4.4.5. Extract mockup previews, categories, and meta ---
    primary_colour = listing_json.get("primary_colour") or parsed_fallback.get("primary_colour", "")
    secondary_colour = listing_json.get("secondary_colour") or parsed_fallback.get("secondary_colour", "")
    used_fallback_naming = bool(listing_json.get("used_fallback_naming", False))
    raw_ai_output = (
        json.dumps(ai_listing, indent=2, ensure_ascii=False)[:800]
        if isinstance(ai_listing, (dict, list))
        else str(ai_listing)[:800]
    )
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    images = sorted(folder.glob(f"{seo_folder}-MU-*.jpg"))
    mockups_from_json = listing_json.get("mockups", []) if listing_json else []
    mockup_previews = []
    for idx, img in enumerate(images):
        cat = ""
        if idx < len(mockups_from_json):
            try:
                cat = Path(mockups_from_json[idx]).parent.name
            except Exception:
                cat = ""
        mockup_previews.append({
            "filename": img.name,
            "category": cat,
            "index": idx,
        })
    categories = get_categories()

    # --- 4.4.6. Build artwork dict for template ---
    artwork = {
        "seo_name": seo_folder,
        "title": ai_title or listing_json.get("title") or seo_folder.replace("-", " ").title(),
        "main_image": f"outputs/processed/{seo_folder}/{seo_folder}.jpg",
        "thumb": f"outputs/processed/{seo_folder}/{seo_folder}-THUMB.jpg",
        "aspect": aspect,
        "tags": tags,
        "materials": materials,
        "has_tags": bool(tags),
        "has_materials": bool(materials),
        "primary_colour": primary_colour,
        "secondary_colour": secondary_colour,
        "full_listing_text": full_listing_text,
    }

    # --- 4.4.7. Render review_artwork.html with context ---
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
        categories=categories,
        menu=get_menu(),
    )

# =============================================================
# 4.4b. REVIEW PAGE: INDIVIDUAL MOCKUP REGENERATE/SWAP ACTIONS
# =============================================================

from werkzeug.exceptions import NotFound

## 4.4b.1. Helper: Find SEO folder from filename
def find_seo_folder_from_filename(aspect, filename):
    basename = Path(filename).stem
    for folder in ARTWORK_PROCESSED_DIR.iterdir():
        if not folder.is_dir():
            continue
        listing_file = folder / f"{folder.name}-listing.json"
        if not listing_file.exists():
            continue
        try:
            with open(listing_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if Path(data.get("filename", "")).stem == basename:
                return folder.name
        except Exception:
            continue
    raise NotFound("SEO folder not found for this artwork.")

## 4.4b.2. Helper: Regenerate one mockup (same category)
def regenerate_one_mockup(seo_folder, slot_idx):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    listing_file = folder / f"{seo_folder}-listing.json"
    if not listing_file.exists():
        return False
    with open(listing_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mockups = data.get("mockups", [])
    if slot_idx < 0 or slot_idx >= len(mockups):
        return False
    mockup_path = Path(mockups[slot_idx])
    category = mockup_path.parent.name
    mockup_files = list((MOCKUPS_DIR / category).glob("*.png"))
    if not mockup_files:
        return False
    import random
    new_mockup = random.choice(mockup_files)
    aspect = data.get("aspect_ratio")
    coords_path = COORDS_ROOT / aspect / f"{new_mockup.stem}.json"
    art_path = folder / f"{seo_folder}.jpg"
    output_path = folder / f"{seo_folder}-MU-{slot_idx+1:02d}.jpg"
    try:
        with open(coords_path, "r", encoding="utf-8") as cf:
            c = json.load(cf)["corners"]
        dst = [[c[0]["x"], c[0]["y"]], [c[1]["x"], c[1]["y"]], [c[3]["x"], c[3]["y"]], [c[2]["x"], c[2]["y"]]]
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image_for_long_edge(art_img)
        mock_img = Image.open(new_mockup).convert("RGBA")
        composite = apply_perspective_transform(art_img, mock_img, dst)
        composite.convert("RGB").save(output_path, "JPEG", quality=85)
        data["mockups"][slot_idx] = str(new_mockup)
        with open(listing_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Regenerate error: {e}")
        return False

## 4.4b.3. Helper: Swap mockup to a new category and regenerate
def swap_one_mockup(seo_folder, slot_idx, new_category):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    listing_file = folder / f"{seo_folder}-listing.json"
    if not listing_file.exists():
        return False
    with open(listing_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mockups = data.get("mockups", [])
    if slot_idx < 0 or slot_idx >= len(mockups):
        return False
    mockup_files = list((MOCKUPS_DIR / new_category).glob("*.png"))
    if not mockup_files:
        return False
    import random
    new_mockup = random.choice(mockup_files)
    aspect = data.get("aspect_ratio")
    coords_path = COORDS_ROOT / aspect / f"{new_mockup.stem}.json"
    art_path = folder / f"{seo_folder}.jpg"
    output_path = folder / f"{seo_folder}-MU-{slot_idx+1:02d}.jpg"
    try:
        with open(coords_path, "r", encoding="utf-8") as cf:
            c = json.load(cf)["corners"]
        dst = [[c[0]["x"], c[0]["y"]], [c[1]["x"], c[1]["y"]], [c[3]["x"], c[3]["y"]], [c[2]["x"], c[2]["y"]]]
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image_for_long_edge(art_img)
        mock_img = Image.open(new_mockup).convert("RGBA")
        composite = apply_perspective_transform(art_img, mock_img, dst)
        composite.convert("RGB").save(output_path, "JPEG", quality=85)
        data["mockups"][slot_idx] = str(new_mockup)
        with open(listing_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Swap error: {e}")
        return False

## 4.4b.4. POST routes for Regenerate/Swap on Review Page
@app.route("/review/<aspect>/<filename>/regenerate/<int:slot_idx>", methods=["POST"])
def review_regenerate_mockup(aspect, filename, slot_idx):
    seo_folder = find_seo_folder_from_filename(aspect, filename)
    regenerate_one_mockup(seo_folder, slot_idx)
    return redirect(url_for("review_artwork", aspect=aspect, filename=filename))

@app.route("/review/<aspect>/<filename>/swap/<int:slot_idx>", methods=["POST"])
def review_swap_mockup(aspect, filename, slot_idx):
    new_cat = request.form["new_category"]
    seo_folder = find_seo_folder_from_filename(aspect, filename)
    swap_one_mockup(seo_folder, slot_idx, new_cat)
    return redirect(url_for("review_artwork", aspect=aspect, filename=filename))

# =============================================================
# 4.5. STATIC & UTILITY ROUTES
# =============================================================

@app.route("/static/outputs/processed/<seo_folder>/<filename>")
def processed_image(seo_folder, filename):
    folder = ARTWORK_PROCESSED_DIR / seo_folder
    return send_from_directory(folder, filename)

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

# =============================================================
# 5. MENU FUNCTION
# =============================================================

def get_menu():
    """
    SECTION 5: Returns a list of dicts for navigation.
    - Home: Always present
    - Artwork Gallery: Always present
    - Review Latest Listing: Only enabled if there is a latest artwork
    """
    menu = [
        {"name": "Home", "url": url_for("home")},
        {"name": "Artwork Gallery", "url": url_for("artworks")},
    ]
    latest = latest_analyzed_artwork()
    if latest:
        menu.append({
            "name": "Review Latest Listing",
            "url": url_for("review_artwork", aspect=latest["aspect"], filename=latest["filename"])
        })
    else:
        menu.append({
            "name": "Review Latest Listing",
            "url": None
        })
    return menu

# =============================================================
# SECTION X: Artwork Listing Helpers
# =============================================================

def prettify_slug(slug: str) -> str:
    """Return a human friendly title from a slug or filename."""
    name = os.path.splitext(slug)[0]
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name.title()


def list_processed_artworks() -> tuple[list[dict], set[str]]:
    """Collect processed artworks and a set of their original filenames."""
    items: list[dict] = []
    processed_names: set[str] = set()
    for folder in ARTWORK_PROCESSED_DIR.iterdir():
        if not folder.is_dir():
            continue
        listing_path = folder / f"{folder.name}-listing.json"
        if not listing_path.exists():
            continue
        try:
            with open(listing_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        original_name = data.get("filename")
        if original_name:
            processed_names.add(original_name)
        items.append(
            {
                "seo_folder": folder.name,
                "filename": original_name or f"{folder.name}.jpg",
                "aspect": data.get("aspect_ratio", ""),
                "title": data.get("title") or prettify_slug(folder.name),
                "thumb": f"{folder.name}-THUMB.jpg",
            }
        )
    items.sort(key=lambda x: x["title"].lower())
    return items, processed_names


def list_ready_to_analyze(processed_names: set[str]) -> list[dict]:
    """Return artworks present in inputs/ that are not yet processed."""
    ready: list[dict] = []
    for aspect_dir in ARTWORKS_DIR.iterdir():
        if not aspect_dir.is_dir():
            continue
        aspect = aspect_dir.name
        for img_path in sorted(aspect_dir.glob("*.jpg")):
            if img_path.name in processed_names:
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                # Corrupted or unreadable image; skip gracefully
                continue
            ready.append(
                {
                    "aspect": aspect,
                    "filename": img_path.name,
                    "title": prettify_slug(img_path.stem),
                    "thumb": img_path.name,
                }
            )
    ready.sort(key=lambda x: x["title"].lower())
    return ready

# =============================================================
# 6. APP RUNNER
# =============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    print(f"🎨 Starting CapitalArt UI at http://localhost:{port}/ ...")
    app.run(debug=debug, port=port)
