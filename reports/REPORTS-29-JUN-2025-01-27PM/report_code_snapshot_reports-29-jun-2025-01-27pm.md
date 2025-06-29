# 🧠 CapitalArt Code Snapshot — REPORTS-29-JUN-2025-01-27PM


---
## 📄 capitalart.py

```py
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
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from PIL import Image, ImageDraw
import cv2
import numpy as np

# === [ 1. ENVIRONMENT & PATHS ] ===

BASE_DIR = Path(__file__).parent.resolve()
MOCKUPS_DIR = BASE_DIR / "inputs" / "mockups" / "4x5-categorised"
ARTWORKS_DIR = BASE_DIR / "inputs" / "artworks"
ARTWORK_PROCESSED_DIR = BASE_DIR / "outputs" / "processed"
SELECTIONS_DIR = BASE_DIR / "outputs" / "selections"
LOGS_DIR = BASE_DIR / "logs"
COMPOSITES_DIR = BASE_DIR / "outputs" / "composites"
FINALISED_DIR = BASE_DIR / "outputs" / "finalised-artwork"

ANALYZE_SCRIPT_PATH = BASE_DIR / "scripts" / "analyze_artwork.py"

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mockup-secret-key")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR / "composites-workflow.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def get_categories():
    return sorted(
        [folder.name for folder in MOCKUPS_DIR.iterdir() if folder.is_dir() and folder.name.lower() != "uncategorised"]
    )


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


def resize_image_for_long_edge(image: Image.Image, target_long_edge: int = 2000) -> Image.Image:
    """Resize an image to a given long edge while keeping aspect ratio."""
    width, height = image.size
    if width > height:
        new_width = target_long_edge
        new_height = int(height * (target_long_edge / width))
    else:
        new_height = target_long_edge
        new_width = int(width * (target_long_edge / height))
    return image.resize((new_width, new_height), Image.LANCZOS)


def apply_perspective_transform(art_img: Image.Image, mockup_img: Image.Image, dst_coords: list) -> Image.Image:
    """Warp art_img into mockup_img using given corner coordinates."""
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


def latest_composite_folder() -> str | None:
    """Return the folder name containing the most recently modified composite."""
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

    latest = latest_composite_folder()
    if latest:
        session["latest_seo_folder"] = latest
        logging.info("Generated composites for %s", latest)
        return redirect(url_for("composites_specific", seo_folder=latest))
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
    """Redirect to the most recent composite preview if available."""
    latest = latest_composite_folder()
    if not latest:
        return render_template("composites_preview.html", images=None, menu=get_menu())
    return redirect(url_for("composites_specific", seo_folder=latest))


@app.route("/composites/<seo_folder>")
def composites_specific(seo_folder):
    """Display all composites for a given artwork folder."""
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


@app.route("/regenerate_composite/<seo_folder>/<int:slot_index>", methods=["POST"])
def regenerate_composite(seo_folder, slot_index):
    """Re-generate a single composite image for the given slot."""
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
    """Finalize and copy composites to the finalised folder."""
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
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
        {"name": "Mockup Selector", "url": url_for("select")},
        {"name": "Artwork Gallery", "url": url_for("artworks")},
        {"name": "Artwork Review", "url": url_for("artwork_review")},
        {"name": "Review Listing", "url": url_for("review")},
    ]


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    print(f"🎨 Starting CapitalArt UI at http://localhost:{port}/ ...")
    app.run(debug=debug, port=port)

```

---
## 📄 requirements.txt

```txt
aiohappyeyeballs==2.6.1
aiohttp==3.12.13
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.9.0
attrs==25.3.0
blinker==1.9.0
certifi==2025.6.15
charset-normalizer==3.4.2
click==8.2.1
distro==1.9.0
Flask==3.1.1
frozenlist==1.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.10.0
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
multidict==6.5.1
numpy==2.3.1
openai==1.93.0
opencv-python==4.11.0.86
pandas==2.3.0
pillow==11.2.1
propcache==0.3.2
pydantic==2.11.7
pydantic_core==2.33.2
Pygments==2.19.2
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
pytz==2025.2
requests==2.32.4
rich==14.0.0
six==1.17.0
sniffio==1.3.1
tqdm==4.67.1
typing-inspection==0.4.1
typing_extensions==4.14.0
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
yarl==1.20.1
scikit-learn==1.7.0

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
## 📄 smart_sign_artwork.py

```py
import os
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
import random
from pathlib import Path
import math

# === [ CapitalArt Lite: CONFIGURATION ] ===
# Paths are defined here for easy modification.

# The local directory where your sorted artworks are located.
INPUT_IMAGE_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/SmartArt-sign-System/artwork-input"
OUTPUT_SIGNED_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/SmartArt-sign-System/artwork-signed-output"
SIGNATURE_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/SmartArt-sign-System/signatures"

# Dictionary mapping logical color names to the full path of each signature PNG.
# Ensure these paths exactly match your file system.
SIGNATURE_PNGS = {
    "beige": Path(SIGNATURE_DIR) / "beige.png",
    "black": Path(SIGNATURE_DIR) / "black.png",
    "blue": Path(SIGNATURE_DIR) / "blue.png",
    "brown": Path(SIGNATURE_DIR) / "brown.png",
    "gold": Path(SIGNATURE_DIR) / "gold.png",
    "green": Path(SIGNATURE_DIR) / "green.png",
    "grey": Path(SIGNATURE_DIR) / "grey.png",
    "odd": Path(SIGNATURE_DIR) / "odd.png", # Placeholder, adjust RGB if needed
    "red": Path(SIGNATURE_DIR) / "red.png",
    "skyblue": Path(SIGNATURE_DIR) / "skyblue.png",
    "white": Path(SIGNATURE_DIR) / "white.png",
    "yellow": Path(SIGNATURE_DIR) / "yellow.png"
}

# Representative RGB values for each signature color for contrast calculation.
# These are approximations. You might adjust them for more accuracy if needed.
SIGNATURE_COLORS_RGB = {
    "beige": (245, 245, 220),
    "black": (0, 0, 0),
    "blue": (0, 0, 255),
    "brown": (139, 69, 19),
    "gold": (255, 215, 0),
    "green": (0, 255, 0),
    "grey": (128, 128, 128),
    "odd": (128, 128, 128), # Treat as a mid-grey for contrast if actual color is unknown/variable
    "red": (255, 0, 0),
    "skyblue": (135, 206, 235),
    "white": (210, 210, 210),
    "yellow": (255, 255, 0)
}

SIGNATURE_SIZE_PERCENTAGE = 0.05 # 6% of long edge for signature size
SIGNATURE_MARGIN_PERCENTAGE = 0.03 # 3% margin from image edges
SMOOTHING_BUFFER_PIXELS = 3 # Extra pixels around the signature shape for smoothing
BLUR_RADIUS = 25 # Adjust for desired blur intensity of the smoothed patch (increased for more blend)
NUM_COLORS_FOR_ZONE_ANALYSIS = 2 # How many dominant colors to find in the signature zone for smoothing

# === [ CapitalArt Lite: UTILITY FUNCTIONS ] ===

def get_relative_luminance(rgb):
    """Calculates the relative luminance of an RGB color, per WCAG 2.0."""
    r, g, b = [x / 255.0 for x in rgb]
    
    # Apply gamma correction
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrast_ratio(rgb1, rgb2):
    """Calculates the WCAG contrast ratio between two RGB colors."""
    L1 = get_relative_luminance(rgb1)
    L2 = get_relative_luminance(rgb2)
    
    if L1 > L2:
        return (L1 + 0.05) / (L2 + 0.05)
    else:
        return (L2 + 0.05) / (L1 + 0.05)

def get_dominant_color_in_masked_zone(image_data_pixels, mask_pixels, num_colors=1):
    """
    Finds the most dominant color(s) within the part of the image
    that corresponds to the opaque areas of the mask.
    `image_data_pixels` should be a flat list of (R,G,B) tuples.
    `mask_pixels` should be a flat list of alpha values (0-255).
    """
    
    # Filter pixels from the image that are within the opaque part of the mask
    # We assume mask_pixels and image_data_pixels are aligned
    masked_pixels = []
    for i in range(len(mask_pixels)):
        if mask_pixels[i] > 0: # Check if the mask pixel is not fully transparent
            masked_pixels.append(image_data_pixels[i])
            
    if not masked_pixels:
        print("  Warning: No non-transparent pixels in mask for color analysis. Defaulting to black.")
        return (0, 0, 0) # Fallback if mask is entirely transparent
            
    pixels_array = np.array(masked_pixels).reshape(-1, 3)
    
    if pixels_array.shape[0] < num_colors:
        # Fallback to mean color if not enough pixels for clustering
        return tuple(map(int, np.mean(pixels_array, axis=0)))

    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto').fit(pixels_array)
    return tuple(map(int, kmeans.cluster_centers_[0]))

def get_contrasting_signature_path(background_rgb, signature_colors_map, signature_paths_map):
    """
    Chooses the signature PNG path that provides the best contrast
    against the given background color.
    """
    best_signature_name = None
    max_contrast = -1.0
    
    for sig_name, sig_rgb in signature_colors_map.items():
        # Skip if the signature file doesn't exist
        if not signature_paths_map.get(sig_name, '').is_file():
            continue

        contrast = get_contrast_ratio(background_rgb, sig_rgb)
        if contrast > max_contrast:
            max_contrast = contrast
            best_signature_name = sig_name
            
    if best_signature_name and best_signature_name in signature_paths_map:
        print(f"  Selected '{best_signature_name}' signature for contrast (Contrast: {max_contrast:.2f}).")
        return signature_paths_map[best_signature_name]
    
    # Fallback to black if no suitable signature found or an issue
    print(f"  Fallback: No best contrasting signature found. Using black.png.")
    return signature_paths_map.get("black", None)

# --- MAIN PROCESSING FUNCTION ---

def add_smart_signature(image_path):
    try:
        with Image.open(image_path).convert("RGB") as img:
            width, height = img.size

            # 1. Determine Signature Placement (bottom-left or bottom-right)
            choose_right = random.choice([True, False])

            # Calculate signature size based on long edge
            long_edge = max(width, height)
            signature_target_size = int(long_edge * SIGNATURE_SIZE_PERCENTAGE)
            
            # Calculate final position of the signature
            # This is where the signature will ultimately be pasted.
            # We need these coordinates to build the mask for smoothing.
            
            # Use a dummy signature image to get its aspect ratio for calculating paste size
            # (assuming all signatures have similar aspect ratio)
            dummy_sig_path = list(SIGNATURE_PNGS.values())[0] # Pick any signature to get initial aspect ratio
            with Image.open(dummy_sig_path).convert("RGBA") as dummy_sig:
                dummy_sig_width, dummy_sig_height = dummy_sig.size
                if dummy_sig_width > dummy_sig_height:
                    scaled_sig_width = signature_target_size
                    scaled_sig_height = int(dummy_sig_height * (scaled_sig_width / dummy_sig_width))
                else:
                    scaled_sig_height = signature_target_size
                    scaled_sig_width = int(dummy_sig_width * (scaled_sig_height / dummy_sig_height))
            
            margin_x = int(width * SIGNATURE_MARGIN_PERCENTAGE)
            margin_y = int(height * SIGNATURE_MARGIN_PERCENTAGE)

            if choose_right:
                sig_paste_x = width - scaled_sig_width - margin_x
            else:
                sig_paste_x = margin_x
            sig_paste_y = height - scaled_sig_height - margin_y


            # 2. Generate the Expanded Smoothing Mask based on Signature Shape
            signature_png_path_for_mask = list(SIGNATURE_PNGS.values())[0] # Use any signature to generate the base mask shape
            if not signature_png_path_for_mask or not Path(signature_png_path_for_mask).is_file():
                print(f"  ❌ Skipping {os.path.basename(image_path)}: Base signature for mask not found.")
                return

            with Image.open(signature_png_path_for_mask).convert("RGBA") as base_signature_img:
                # Resize the base signature to its target paste size
                base_signature_img_resized = base_signature_img.resize(
                    (scaled_sig_width, scaled_sig_height), Image.Resampling.LANCZOS
                )
                
                # Create a blank mask canvas (full image size)
                mask_canvas = Image.new("L", img.size, 0) # 'L' mode for grayscale (alpha)
                
                # Paste the resized signature's alpha channel onto the mask canvas
                # at the exact final paste position
                mask_alpha = base_signature_img_resized.split()[-1] # Get alpha channel
                mask_canvas.paste(mask_alpha, (sig_paste_x, sig_paste_y))
                
                # Expand the mask by blurring and re-thresholding (simulates dilation)
                # Apply blur to expand the shape
                expanded_mask = mask_canvas.filter(ImageFilter.GaussianBlur(SMOOTHING_BUFFER_PIXELS))
                # Re-threshold to make it solid again (optional, for crisp expanded edge)
                # If you want a softer halo, you can skip this step and use `expanded_mask` directly as alpha
                expanded_mask = expanded_mask.point(lambda x: 255 if x > 10 else 0)


            # 3. Analyze Artwork Pixels within the Expanded Mask for Dominant Color
            # Get all RGB pixels from the original image
            original_image_rgb_data = list(img.getdata())
            # Get all alpha pixels from the expanded mask
            expanded_mask_alpha_data = list(expanded_mask.getdata())

            dominant_zone_color = get_dominant_color_in_masked_zone(
                original_image_rgb_data, expanded_mask_alpha_data, NUM_COLORS_FOR_ZONE_ANALYSIS
            )
            print(f"  Dominant background color for zone: {dominant_zone_color}")


            # 4. Create and Apply the Smoothed Patch
            # Create a new RGB image filled with the dominant color
            smoothed_patch_base = Image.new("RGB", img.size, dominant_zone_color)
            
            # Apply blur to this color patch. The `expanded_mask` will control its visibility.
            # This blur will extend outwards from the shape
            smoothed_patch_blurred = smoothed_patch_base.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
            
            # Combine the blurred color patch with the expanded mask
            # This creates the final smoothed layer in the shape of the expanded signature
            # We need to make it RGBA for alpha_composite
            smoothed_patch_rgba = smoothed_patch_blurred.copy().convert("RGBA")
            smoothed_patch_rgba.putalpha(expanded_mask) # Use the expanded mask as its alpha channel

            # Composite the smoothed patch onto the original image
            img = Image.alpha_composite(img.convert("RGBA"), smoothed_patch_rgba).convert("RGB") # Convert back to RGB if desired


            # 5. Determine Complimentary Signature Color (and path to PNG)
            signature_png_path = get_contrasting_signature_path(
                dominant_zone_color, SIGNATURE_COLORS_RGB, SIGNATURE_PNGS
            )
            
            if not signature_png_path or not Path(signature_png_path).is_file():
                print(f"  ❌ Skipping {os.path.basename(image_path)}: Could not find a valid signature PNG at path: {signature_png_path}")
                return

            # 6. Place the Actual Signature
            with Image.open(signature_png_path).convert("RGBA") as signature_img:
                # Resize to the previously calculated scaled_sig_width/height
                signature_img = signature_img.resize(
                    (scaled_sig_width, scaled_sig_height), Image.Resampling.LANCZOS
                )
                
                # Paste signature onto the modified image
                img.paste(signature_img, (sig_paste_x, sig_paste_y), signature_img)

            # Save the signed image
            output_path = Path(OUTPUT_SIGNED_DIR) / os.path.basename(image_path)
            img.save(output_path)
            print(f"✅ Signed: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"❌ Error signing {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure output directory exists
    Path(OUTPUT_SIGNED_DIR).mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Smart Signature Batch Processing (Shape-Based Smoothing) ---")
    print(f"Reading artworks from: {INPUT_IMAGE_DIR}")
    print(f"Saving signed artworks to: {OUTPUT_SIGNED_DIR}")
    print(f"Using signatures from: {SIGNATURE_DIR}")
    
    processed_files = 0
    for root, _, files in os.walk(INPUT_IMAGE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.gif')):
                image_path = Path(root) / file
                print(f"\nProcessing {file}...")
                add_smart_signature(image_path)
                processed_files += 1

    print(f"\n--- Smart Signature Batch Processing Complete ---")
    print(f"Total files processed: {processed_files}")
    print(f"Check '{OUTPUT_SIGNED_DIR}' for your signed artworks.")
```

---
## 📄 folder_structure.txt

```txt
capitalart
├── .env
├── .gitignore
├── LICENSE
├── README.md
├── assets
│   ├── mockups
│   └── style.css
├── capitalart-total-nuclear.py
├── capitalart.py
├── changes
├── descriptions
│   ├── artworks.json
│   ├── autojourney_downloader_export_26_06_2025_15_02_55.csv
│   └── autojourney_downloader_export_26_06_2025_15_03_12.json
├── folder_structure.txt
├── generate_folder_tree.py
├── generic_texts
│   ├── 16x9.txt
│   ├── 1x1.txt
│   ├── 2x3.txt
│   ├── 3x2.txt
│   ├── 3x4.txt
│   ├── 4x3.txt
│   ├── 4x5.txt
│   ├── 5x4.txt
│   ├── 5x7.txt
│   ├── 7x5.txt
│   ├── 9x16.txt
│   ├── A-Series-Horizontal.txt
│   └── A-Series-Verical.txt
├── inputs
│   ├── Coordinates
│   │   ├── 16x9
│   │   │   ├── 16x9-Mockup-1.json
│   │   │   ├── 16x9-Mockup-10.json
│   │   │   ├── 16x9-Mockup-11.json
│   │   │   ├── 16x9-Mockup-12.json
│   │   │   ├── 16x9-Mockup-13.json
│   │   │   ├── 16x9-Mockup-14.json
│   │   │   ├── 16x9-Mockup-15.json
│   │   │   ├── 16x9-Mockup-16.json
│   │   │   ├── 16x9-Mockup-17.json
│   │   │   ├── 16x9-Mockup-18.json
│   │   │   ├── 16x9-Mockup-19.json
│   │   │   ├── 16x9-Mockup-2.json
│   │   │   ├── 16x9-Mockup-20.json
│   │   │   ├── 16x9-Mockup-21.json
│   │   │   ├── 16x9-Mockup-22.json
│   │   │   ├── 16x9-Mockup-23.json
│   │   │   ├── 16x9-Mockup-24.json
│   │   │   ├── 16x9-Mockup-25.json
│   │   │   ├── 16x9-Mockup-26.json
│   │   │   ├── 16x9-Mockup-27.json
│   │   │   ├── 16x9-Mockup-28.json
│   │   │   ├── 16x9-Mockup-29.json
│   │   │   ├── 16x9-Mockup-3.json
│   │   │   ├── 16x9-Mockup-30.json
│   │   │   ├── 16x9-Mockup-31.json
│   │   │   ├── 16x9-Mockup-32.json
│   │   │   ├── 16x9-Mockup-33.json
│   │   │   ├── 16x9-Mockup-34.json
│   │   │   ├── 16x9-Mockup-35.json
│   │   │   ├── 16x9-Mockup-36.json
│   │   │   ├── 16x9-Mockup-37.json
│   │   │   ├── 16x9-Mockup-38.json
│   │   │   ├── 16x9-Mockup-39.json
│   │   │   ├── 16x9-Mockup-4.json
│   │   │   ├── 16x9-Mockup-40.json
│   │   │   ├── 16x9-Mockup-41.json
│   │   │   ├── 16x9-Mockup-42.json
│   │   │   ├── 16x9-Mockup-43.json
│   │   │   ├── 16x9-Mockup-44.json
│   │   │   ├── 16x9-Mockup-45.json
│   │   │   ├── 16x9-Mockup-46.json
│   │   │   ├── 16x9-Mockup-47.json
│   │   │   ├── 16x9-Mockup-48.json
│   │   │   ├── 16x9-Mockup-49.json
│   │   │   ├── 16x9-Mockup-5.json
│   │   │   ├── 16x9-Mockup-50.json
│   │   │   ├── 16x9-Mockup-51.json
│   │   │   ├── 16x9-Mockup-52.json
│   │   │   ├── 16x9-Mockup-53.json
│   │   │   ├── 16x9-Mockup-54.json
│   │   │   ├── 16x9-Mockup-6.json
│   │   │   ├── 16x9-Mockup-7.json
│   │   │   ├── 16x9-Mockup-8.json
│   │   │   └── 16x9-Mockup-9.json
│   │   ├── 1x1
│   │   │   ├── 1x1-Mockup-1.json
│   │   │   ├── 1x1-Mockup-10.json
│   │   │   ├── 1x1-Mockup-11.json
│   │   │   ├── 1x1-Mockup-12.json
│   │   │   ├── 1x1-Mockup-13.json
│   │   │   ├── 1x1-Mockup-14.json
│   │   │   ├── 1x1-Mockup-15.json
│   │   │   ├── 1x1-Mockup-16.json
│   │   │   ├── 1x1-Mockup-17.json
│   │   │   ├── 1x1-Mockup-18.json
│   │   │   ├── 1x1-Mockup-19.json
│   │   │   ├── 1x1-Mockup-2.json
│   │   │   ├── 1x1-Mockup-20.json
│   │   │   ├── 1x1-Mockup-21.json
│   │   │   ├── 1x1-Mockup-22.json
│   │   │   ├── 1x1-Mockup-23.json
│   │   │   ├── 1x1-Mockup-24.json
│   │   │   ├── 1x1-Mockup-25.json
│   │   │   ├── 1x1-Mockup-26.json
│   │   │   ├── 1x1-Mockup-27.json
│   │   │   ├── 1x1-Mockup-28.json
│   │   │   ├── 1x1-Mockup-29.json
│   │   │   ├── 1x1-Mockup-3.json
│   │   │   ├── 1x1-Mockup-30.json
│   │   │   ├── 1x1-Mockup-31.json
│   │   │   ├── 1x1-Mockup-32.json
│   │   │   ├── 1x1-Mockup-33.json
│   │   │   ├── 1x1-Mockup-34.json
│   │   │   ├── 1x1-Mockup-35.json
│   │   │   ├── 1x1-Mockup-36.json
│   │   │   ├── 1x1-Mockup-37.json
│   │   │   ├── 1x1-Mockup-38.json
│   │   │   ├── 1x1-Mockup-39.json
│   │   │   ├── 1x1-Mockup-4.json
│   │   │   ├── 1x1-Mockup-40.json
│   │   │   ├── 1x1-Mockup-41.json
│   │   │   ├── 1x1-Mockup-42.json
│   │   │   ├── 1x1-Mockup-43.json
│   │   │   ├── 1x1-Mockup-44.json
│   │   │   ├── 1x1-Mockup-45.json
│   │   │   ├── 1x1-Mockup-46.json
│   │   │   ├── 1x1-Mockup-47.json
│   │   │   ├── 1x1-Mockup-48.json
│   │   │   ├── 1x1-Mockup-49.json
│   │   │   ├── 1x1-Mockup-5.json
│   │   │   ├── 1x1-Mockup-50.json
│   │   │   ├── 1x1-Mockup-51.json
│   │   │   ├── 1x1-Mockup-52.json
│   │   │   ├── 1x1-Mockup-53.json
│   │   │   ├── 1x1-Mockup-54.json
│   │   │   ├── 1x1-Mockup-55.json
│   │   │   ├── 1x1-Mockup-56.json
│   │   │   ├── 1x1-Mockup-57.json
│   │   │   ├── 1x1-Mockup-58.json
│   │   │   ├── 1x1-Mockup-59.json
│   │   │   ├── 1x1-Mockup-6.json
│   │   │   ├── 1x1-Mockup-60.json
│   │   │   ├── 1x1-Mockup-61.json
│   │   │   ├── 1x1-Mockup-62.json
│   │   │   ├── 1x1-Mockup-63.json
│   │   │   ├── 1x1-Mockup-64.json
│   │   │   ├── 1x1-Mockup-65.json
│   │   │   ├── 1x1-Mockup-66.json
│   │   │   ├── 1x1-Mockup-67.json
│   │   │   ├── 1x1-Mockup-68.json
│   │   │   ├── 1x1-Mockup-69.json
│   │   │   ├── 1x1-Mockup-7.json
│   │   │   ├── 1x1-Mockup-70.json
│   │   │   ├── 1x1-Mockup-71.json
│   │   │   ├── 1x1-Mockup-72.json
│   │   │   ├── 1x1-Mockup-73.json
│   │   │   ├── 1x1-Mockup-74.json
│   │   │   ├── 1x1-Mockup-8.json
│   │   │   └── 1x1-Mockup-9.json
│   │   ├── 2x3
│   │   │   ├── 2x3-Mockup-1.json
│   │   │   ├── 2x3-Mockup-10.json
│   │   │   ├── 2x3-Mockup-11.json
│   │   │   ├── 2x3-Mockup-12.json
│   │   │   ├── 2x3-Mockup-13.json
│   │   │   ├── 2x3-Mockup-14.json
│   │   │   ├── 2x3-Mockup-15.json
│   │   │   ├── 2x3-Mockup-16.json
│   │   │   ├── 2x3-Mockup-17.json
│   │   │   ├── 2x3-Mockup-18.json
│   │   │   ├── 2x3-Mockup-19.json
│   │   │   ├── 2x3-Mockup-2.json
│   │   │   ├── 2x3-Mockup-20.json
│   │   │   ├── 2x3-Mockup-21.json
│   │   │   ├── 2x3-Mockup-22.json
│   │   │   ├── 2x3-Mockup-23.json
│   │   │   ├── 2x3-Mockup-24.json
│   │   │   ├── 2x3-Mockup-25.json
│   │   │   ├── 2x3-Mockup-26.json
│   │   │   ├── 2x3-Mockup-3.json
│   │   │   ├── 2x3-Mockup-4.json
│   │   │   ├── 2x3-Mockup-5.json
│   │   │   ├── 2x3-Mockup-6.json
│   │   │   ├── 2x3-Mockup-7.json
│   │   │   ├── 2x3-Mockup-8.json
│   │   │   └── 2x3-Mockup-9.json
│   │   ├── 3x2
│   │   │   ├── 3x2-Mockup-1.json
│   │   │   ├── 3x2-Mockup-10.json
│   │   │   ├── 3x2-Mockup-11.json
│   │   │   ├── 3x2-Mockup-12.json
│   │   │   ├── 3x2-Mockup-13.json
│   │   │   ├── 3x2-Mockup-14.json
│   │   │   ├── 3x2-Mockup-15.json
│   │   │   ├── 3x2-Mockup-16.json
│   │   │   ├── 3x2-Mockup-17.json
│   │   │   ├── 3x2-Mockup-18.json
│   │   │   ├── 3x2-Mockup-19.json
│   │   │   ├── 3x2-Mockup-2.json
│   │   │   ├── 3x2-Mockup-20.json
│   │   │   ├── 3x2-Mockup-3.json
│   │   │   ├── 3x2-Mockup-4.json
│   │   │   ├── 3x2-Mockup-5.json
│   │   │   ├── 3x2-Mockup-6.json
│   │   │   ├── 3x2-Mockup-7.json
│   │   │   ├── 3x2-Mockup-8.json
│   │   │   └── 3x2-Mockup-9.json
│   │   ├── 3x4
│   │   │   ├── 3x4-Mockup-1.json
│   │   │   ├── 3x4-Mockup-10.json
│   │   │   ├── 3x4-Mockup-11.json
│   │   │   ├── 3x4-Mockup-12.json
│   │   │   ├── 3x4-Mockup-13.json
│   │   │   ├── 3x4-Mockup-14.json
│   │   │   ├── 3x4-Mockup-15.json
│   │   │   ├── 3x4-Mockup-16.json
│   │   │   ├── 3x4-Mockup-17.json
│   │   │   ├── 3x4-Mockup-18.json
│   │   │   ├── 3x4-Mockup-19.json
│   │   │   ├── 3x4-Mockup-2.json
│   │   │   ├── 3x4-Mockup-20.json
│   │   │   ├── 3x4-Mockup-3.json
│   │   │   ├── 3x4-Mockup-4.json
│   │   │   ├── 3x4-Mockup-5.json
│   │   │   ├── 3x4-Mockup-6.json
│   │   │   ├── 3x4-Mockup-7.json
│   │   │   ├── 3x4-Mockup-8.json
│   │   │   └── 3x4-Mockup-9.json
│   │   ├── 4x3
│   │   │   ├── 4x3-Mockup-1.json
│   │   │   ├── 4x3-Mockup-10.json
│   │   │   ├── 4x3-Mockup-11.json
│   │   │   ├── 4x3-Mockup-12.json
│   │   │   ├── 4x3-Mockup-13.json
│   │   │   ├── 4x3-Mockup-15.json
│   │   │   ├── 4x3-Mockup-16.json
│   │   │   ├── 4x3-Mockup-17.json
│   │   │   ├── 4x3-Mockup-19.json
│   │   │   ├── 4x3-Mockup-20.json
│   │   │   ├── 4x3-Mockup-3.json
│   │   │   ├── 4x3-Mockup-4.json
│   │   │   ├── 4x3-Mockup-5.json
│   │   │   ├── 4x3-Mockup-7.json
│   │   │   ├── 4x3-Mockup-8.json
│   │   │   └── 4x3-Mockup-9.json
│   │   ├── 4x5
│   │   │   ├── 4x5-mockup-01.json
│   │   │   ├── 4x5-mockup-02.json
│   │   │   ├── 4x5-mockup-03.json
│   │   │   ├── 4x5-mockup-04.json
│   │   │   ├── 4x5-mockup-05.json
│   │   │   ├── 4x5-mockup-06.json
│   │   │   ├── 4x5-mockup-07.json
│   │   │   ├── 4x5-mockup-08.json
│   │   │   ├── 4x5-mockup-09.json
│   │   │   ├── 4x5-mockup-10.json
│   │   │   ├── 4x5-mockup-11.json
│   │   │   ├── 4x5-mockup-12.json
│   │   │   ├── 4x5-mockup-13.json
│   │   │   ├── 4x5-mockup-14.json
│   │   │   ├── 4x5-mockup-15.json
│   │   │   ├── 4x5-mockup-16.json
│   │   │   ├── 4x5-mockup-17.json
│   │   │   ├── 4x5-mockup-18.json
│   │   │   ├── 4x5-mockup-19.json
│   │   │   ├── 4x5-mockup-20.json
│   │   │   ├── 4x5-mockup-21.json
│   │   │   ├── 4x5-mockup-22.json
│   │   │   ├── 4x5-mockup-23.json
│   │   │   ├── 4x5-mockup-24.json
│   │   │   ├── 4x5-mockup-25.json
│   │   │   ├── 4x5-mockup-26.json
│   │   │   ├── 4x5-mockup-27.json
│   │   │   ├── 4x5-mockup-28.json
│   │   │   ├── 4x5-mockup-29.json
│   │   │   ├── 4x5-mockup-30.json
│   │   │   ├── 4x5-mockup-31.json
│   │   │   ├── 4x5-mockup-32.json
│   │   │   ├── 4x5-mockup-33.json
│   │   │   ├── 4x5-mockup-34.json
│   │   │   ├── 4x5-mockup-35.json
│   │   │   ├── 4x5-mockup-36.json
│   │   │   ├── 4x5-mockup-37.json
│   │   │   ├── 4x5-mockup-38.json
│   │   │   ├── 4x5-mockup-39.json
│   │   │   ├── 4x5-mockup-40.json
│   │   │   ├── 4x5-mockup-41.json
│   │   │   ├── 4x5-mockup-42.json
│   │   │   ├── 4x5-mockup-43.json
│   │   │   ├── 4x5-mockup-44.json
│   │   │   ├── 4x5-mockup-45.json
│   │   │   ├── 4x5-mockup-46.json
│   │   │   ├── 4x5-mockup-47.json
│   │   │   ├── 4x5-mockup-48.json
│   │   │   ├── 4x5-mockup-49.json
│   │   │   ├── 4x5-mockup-50.json
│   │   │   ├── 4x5-mockup-51.json
│   │   │   ├── 4x5-mockup-52.json
│   │   │   ├── 4x5-mockup-53.json
│   │   │   ├── 4x5-mockup-54.json
│   │   │   ├── 4x5-mockup-55.json
│   │   │   ├── 4x5-mockup-56.json
│   │   │   ├── 4x5-mockup-57.json
│   │   │   ├── 4x5-mockup-58.json
│   │   │   ├── 4x5-mockup-59.json
│   │   │   ├── 4x5-mockup-60.json
│   │   │   ├── 4x5-mockup-61.json
│   │   │   ├── 4x5-mockup-62.json
│   │   │   ├── 4x5-mockup-63.json
│   │   │   ├── 4x5-mockup-64.json
│   │   │   ├── 4x5-mockup-65.json
│   │   │   ├── 4x5-mockup-66.json
│   │   │   ├── 4x5-mockup-67.json
│   │   │   ├── 4x5-mockup-68.json
│   │   │   ├── 4x5-mockup-69.json
│   │   │   ├── 4x5-mockup-70.json
│   │   │   ├── 4x5-mockup-71.json
│   │   │   ├── 4x5-mockup-72.json
│   │   │   ├── 4x5-mockup-73.json
│   │   │   ├── 4x5-mockup-74.json
│   │   │   ├── 4x5-mockup-75.json
│   │   │   ├── 4x5-mockup-76.json
│   │   │   ├── 4x5-mockup-77.json
│   │   │   └── 4x5-mockup-78.json
│   │   ├── 5x4
│   │   │   ├── 5x4-Mockup-1.json
│   │   │   ├── 5x4-Mockup-10.json
│   │   │   ├── 5x4-Mockup-11.json
│   │   │   ├── 5x4-Mockup-12.json
│   │   │   ├── 5x4-Mockup-13.json
│   │   │   ├── 5x4-Mockup-14.json
│   │   │   ├── 5x4-Mockup-15.json
│   │   │   ├── 5x4-Mockup-16.json
│   │   │   ├── 5x4-Mockup-17.json
│   │   │   ├── 5x4-Mockup-18.json
│   │   │   ├── 5x4-Mockup-19.json
│   │   │   ├── 5x4-Mockup-2.json
│   │   │   ├── 5x4-Mockup-20.json
│   │   │   ├── 5x4-Mockup-21.json
│   │   │   ├── 5x4-Mockup-22.json
│   │   │   ├── 5x4-Mockup-3.json
│   │   │   ├── 5x4-Mockup-4.json
│   │   │   ├── 5x4-Mockup-5.json
│   │   │   ├── 5x4-Mockup-6.json
│   │   │   ├── 5x4-Mockup-7.json
│   │   │   ├── 5x4-Mockup-8.json
│   │   │   └── 5x4-Mockup-9.json
│   │   └── 9x16
│   │       ├── 9x16-Mockup-1.json
│   │       ├── 9x16-Mockup-10.json
│   │       ├── 9x16-Mockup-11.json
│   │       ├── 9x16-Mockup-12.json
│   │       ├── 9x16-Mockup-13.json
│   │       ├── 9x16-Mockup-14.json
│   │       ├── 9x16-Mockup-15.json
│   │       ├── 9x16-Mockup-16.json
│   │       ├── 9x16-Mockup-17.json
│   │       ├── 9x16-Mockup-18.json
│   │       ├── 9x16-Mockup-19.json
│   │       ├── 9x16-Mockup-2.json
│   │       ├── 9x16-Mockup-20.json
│   │       ├── 9x16-Mockup-21.json
│   │       ├── 9x16-Mockup-22.json
│   │       ├── 9x16-Mockup-23.json
│   │       ├── 9x16-Mockup-24.json
│   │       ├── 9x16-Mockup-3.json
│   │       ├── 9x16-Mockup-4.json
│   │       ├── 9x16-Mockup-5.json
│   │       ├── 9x16-Mockup-6.json
│   │       ├── 9x16-Mockup-7.json
│   │       ├── 9x16-Mockup-8.json
│   │       └── 9x16-Mockup-9.json
│   ├── artworks
│   │   ├── 16x9
│   │   ├── 1x1
│   │   ├── 2x3
│   │   ├── 3x2
│   │   ├── 3x4
│   │   ├── 4x3
│   │   ├── 4x5
│   │   │   ├── gang-gang-cockatoo-male-generate-an-aboriginal-dot-painting-of-a-gang-gang-cockatoo-callocephalon-fi.jpg
│   │   │   └── night-seeds-rebirth-beneath-the-stars-desert-flora-sprouting-from-fire-country-under-a-swirling-star.jpg
│   │   ├── 5x4
│   │   ├── 5x7
│   │   ├── 7x5
│   │   ├── 9x16
│   │   ├── A-Series-Horizontal
│   │   └── A-Series-Vertical
│   └── mockups
│       └── 4x5-categorised
│           ├── Bedroom
│           │   ├── 4x5-mockup-10.png
│           │   ├── 4x5-mockup-11.png
│           │   ├── 4x5-mockup-12.png
│           │   └── 4x5-mockup-13.png
│           ├── Closeup
│           │   ├── 4x5-mockup-06.png
│           │   ├── 4x5-mockup-07.png
│           │   ├── 4x5-mockup-08.png
│           │   ├── 4x5-mockup-09.png
│           │   ├── 4x5-mockup-16.png
│           │   ├── 4x5-mockup-17.png
│           │   ├── 4x5-mockup-18.png
│           │   ├── 4x5-mockup-19.png
│           │   ├── 4x5-mockup-20.png
│           │   ├── 4x5-mockup-22.png
│           │   ├── 4x5-mockup-23.png
│           │   ├── 4x5-mockup-46.png
│           │   ├── 4x5-mockup-48.png
│           │   ├── 4x5-mockup-49.png
│           │   ├── 4x5-mockup-50.png
│           │   ├── 4x5-mockup-59.png
│           │   ├── 4x5-mockup-65.png
│           │   └── 4x5-mockup-72.png
│           ├── Dining-Room
│           │   ├── 4x5-mockup-75.png
│           │   └── 4x5-mockup-76.png
│           ├── Gallery-Wall
│           │   ├── 4x5-mockup-63.png
│           │   └── 4x5-mockup-64.png
│           ├── Hallway
│           │   ├── 4x5-mockup-25.png
│           │   ├── 4x5-mockup-55.png
│           │   ├── 4x5-mockup-60.png
│           │   └── 4x5-mockup-70.png
│           ├── Living Room
│           │   ├── 4x5-mockup-01.png
│           │   ├── 4x5-mockup-02.png
│           │   ├── 4x5-mockup-04.png
│           │   ├── 4x5-mockup-21.png
│           │   ├── 4x5-mockup-26.png
│           │   ├── 4x5-mockup-27.png
│           │   ├── 4x5-mockup-28.png
│           │   ├── 4x5-mockup-39.png
│           │   ├── 4x5-mockup-40.png
│           │   ├── 4x5-mockup-41.png
│           │   ├── 4x5-mockup-42.png
│           │   ├── 4x5-mockup-43.png
│           │   ├── 4x5-mockup-45.png
│           │   ├── 4x5-mockup-47.png
│           │   ├── 4x5-mockup-51.png
│           │   ├── 4x5-mockup-52.png
│           │   ├── 4x5-mockup-53.png
│           │   ├── 4x5-mockup-54.png
│           │   ├── 4x5-mockup-56.png
│           │   ├── 4x5-mockup-66.png
│           │   ├── 4x5-mockup-67.png
│           │   ├── 4x5-mockup-71.png
│           │   ├── 4x5-mockup-73.png
│           │   ├── 4x5-mockup-74.png
│           │   └── 4x5-mockup-78.png
│           ├── Nursery
│           │   ├── 4x5-mockup-03.png
│           │   └── 4x5-mockup-05.png
│           ├── Office
│           │   ├── 4x5-mockup-14.png
│           │   ├── 4x5-mockup-15.png
│           │   ├── 4x5-mockup-57.png
│           │   ├── 4x5-mockup-58.png
│           │   ├── 4x5-mockup-61.png
│           │   ├── 4x5-mockup-62.png
│           │   ├── 4x5-mockup-68.png
│           │   ├── 4x5-mockup-69.png
│           │   └── 4x5-mockup-77.png
│           ├── Outdoors
│           │   ├── 4x5-mockup-24.png
│           │   ├── 4x5-mockup-29.png
│           │   ├── 4x5-mockup-30.png
│           │   ├── 4x5-mockup-31.png
│           │   ├── 4x5-mockup-32.png
│           │   ├── 4x5-mockup-33.png
│           │   ├── 4x5-mockup-34.png
│           │   ├── 4x5-mockup-35.png
│           │   ├── 4x5-mockup-36.png
│           │   ├── 4x5-mockup-37.png
│           │   └── 4x5-mockup-38.png
│           └── Uncategorised
├── logs
│   ├── analyze-artwork-2025-06-28-1600.log
│   ├── analyze-artwork-2025-06-28-1702.log
│   ├── analyze-artwork-2025-06-28-1721.log
│   ├── analyze-artwork-2025-06-28-1736.log
│   ├── analyze-artwork-2025-06-28-1756.log
│   ├── analyze_088442e9-9497-485d-b582-2300d8d8bbbf.log
│   ├── analyze_170367a7-d63c-4092-a4ef-94d36ccd2cd7.log
│   ├── analyze_1ca028cb-dd7d-478e-a5f4-a8b99b4b143b.log
│   ├── analyze_235aea74-63f1-4832-b301-466929a1ba9e.log
│   ├── analyze_2511e6e8-ef04-4fe8-a195-14ef62a33c83.log
│   ├── analyze_2c29a3db-fd81-40c8-829d-4e225d3f131d.log
│   ├── analyze_2d26bba8-0477-46a3-b62e-19d48e509087.log
│   ├── analyze_5cfabf96-bb87-4af9-b8cb-7f8a96f794e4.log
│   ├── analyze_85d3fa11-7784-4105-9853-bdfba39bd72d.log
│   ├── analyze_89086057-cea9-47ad-be91-4a1bf8a690de.log
│   ├── analyze_8c9d36ae-a9ea-4d3e-a38b-ce080a21d77e.log
│   ├── analyze_91f42848-9227-4cd4-86c0-505d1d04e371.log
│   ├── analyze_a1648627-d0ce-4d80-a425-c63b7ebd0278.log
│   ├── analyze_b548455f-0553-4788-8a6c-57658afc8e73.log
│   ├── analyze_b5b48356-b5c4-41b3-bee0-7e54c179d5de.log
│   ├── analyze_b6a64043-6838-4c64-b2c9-8d688f9c5a6c.log
│   ├── analyze_ba16cf25-1ebe-4b04-8e70-8b9f7b163452.log
│   ├── analyze_c0788def-5865-4c1a-b7a2-172e33dd16f3.log
│   ├── analyze_dd0083a2-707c-4948-a918-7521d1262b3d.log
│   ├── analyze_e37f0a60-fa3b-45f7-962e-3263ad7495a4.log
│   ├── analyze_eebad4e8-aa8c-4668-bf34-6d3b93be63e9.log
│   ├── composites-workflow.log
│   ├── composites_4535c6f9-9da5-47f5-b513-d21a6c20a225.log
│   ├── composites_5478536b-1843-40c8-97b8-9aa00d5f906a.log
│   ├── composites_56e29048-5369-4c3e-bf15-fec81ca6b2bf.log
│   ├── composites_9a9eed43-132d-489e-9cad-6611d1210334.log
│   ├── composites_9e4ac00d-e0fc-4a4c-917d-d0784ec309ad.log
│   ├── composites_9fff1d6a-af8a-449d-a57b-4dadd6aca4c6.log
│   ├── composites_bab70799-4f83-406c-acb8-175fcb9f03d6.log
│   ├── composites_c69c5716-6a6a-4997-9e48-5668da48bcab.log
│   ├── composites_c6fc57bb-071a-4f2a-9e7c-f10608e66194.log
│   └── crops
│       ├── gang-gang-cockatoo-artwork-by-robin-custance-rjc-0001-crop.jpg
│       ├── gang-gang-cockatoo-dot-artwork-by-robin-custance-rjc-0001-crop.jpg
│       └── night-seeds-rebirth-beneath-the-stars-desert-flora-sprouting-from-fire-country-under-a-swirling-star-crop.jpg
├── mockup_categorisation_log.txt
├── mockup_categoriser.py
├── older-scripts
│   ├── ai-jpeg-reset.sh
│   ├── backup_mockup_generator.sh
│   ├── backup_mockup_structure.sh
│   ├── compatible-image-generator.jsx
│   ├── dupe-png-files-and-rename.py
│   ├── export-artwork-layers.jsx
│   ├── find-matching-png-files.py
│   ├── fix_srgb_profiles.py
│   ├── gather_mockup_code_to_text.sh
│   ├── generate_all_coordinates.py
│   ├── generate_composites.py
│   ├── generate_folder_structure.py
│   ├── google_vision_basic_analyzer.py
│   ├── openai_vision_test.py
│   ├── organise-jpg-and-png-files.py
│   ├── rename-4x5-mockup-layers.jsx
│   ├── utils.py
│   └── write_all_composite_generators.py
├── outputs
│   ├── Composites
│   │   ├── 16x9
│   │   ├── 1x1
│   │   ├── 2x3
│   │   ├── 3x2
│   │   ├── 3x4
│   │   ├── 4x3
│   │   ├── 4x5
│   │   ├── 5x4
│   │   ├── 5x7
│   │   ├── 7x5
│   │   ├── 9x16
│   │   ├── A-Series
│   │   ├── A-Series-Horizontal
│   │   └── A-Series-Vertical
│   ├── artwork_listing_master.json
│   └── processed
│       ├── gang-gang-cockatoo-artwork-by-robin-custance-rjc-0001
│       │   ├── gang-gang-cockatoo-artwork-by-robin-custance-rjc-0001-THUMB.jpg
│       │   ├── gang-gang-cockatoo-artwork-by-robin-custance-rjc-0001-listing.json
│       │   ├── gang-gang-cockatoo-artwork-by-robin-custance-rjc-0001.jpg
│       │   └── original-gang-gang-cockatoo-male-generate-an-aboriginal-dot-painting-of-a-gang-gang-cockatoo-callocephalon-fi.jpg
│       └── pending_mockups.json
├── package-lock.json
├── reports
│   ├── REPORTS-25-JUN-2025-04-18PM
│   │   └── report_code_snapshot_reports-25-jun-2025-04-18pm.md
│   ├── REPORTS-25-JUN-2025-04-27PM
│   │   └── report_code_snapshot_reports-25-jun-2025-04-27pm.md
│   ├── REPORTS-25-JUN-2025-04-29PM
│   │   ├── pip_check.txt
│   │   ├── pip_outdated.txt
│   │   └── report_code_snapshot_reports-25-jun-2025-04-29pm.md
│   ├── REPORTS-25-JUN-2025-04-30PM
│   │   └── report_code_snapshot_reports-25-jun-2025-04-30pm.md
│   ├── REPORTS-25-JUN-2025-06-27PM
│   │   └── report_code_snapshot_reports-25-jun-2025-06-27pm.md
│   ├── REPORTS-25-JUN-2025-10-45PM
│   │   └── report_code_snapshot_reports-25-jun-2025-10-45pm.md
│   ├── REPORTS-25-JUN-2025-10-53PM
│   │   └── report_code_snapshot_reports-25-jun-2025-10-53pm.md
│   ├── REPORTS-26-JUN-2025-02-32PM
│   │   └── report_code_snapshot_reports-26-jun-2025-02-32pm.md
│   ├── REPORTS-26-JUN-2025-02-39PM
│   │   └── report_code_snapshot_reports-26-jun-2025-02-39pm.md
│   ├── REPORTS-26-JUN-2025-02-47PM
│   │   └── report_code_snapshot_reports-26-jun-2025-02-47pm.md
│   ├── REPORTS-27-JUN-2025-11-19PM
│   │   └── report_code_snapshot_reports-27-jun-2025-11-19pm.md
│   ├── REPORTS-28-JUN-2025-10-51AM
│   │   └── report_code_snapshot_reports-28-jun-2025-10-51am.md
│   ├── REPORTS-28-JUN-2025-11-40AM
│   │   └── report_code_snapshot_reports-28-jun-2025-11-40am.md
│   ├── REPORTS-28-JUN-2025-12-10PM
│   │   └── report_code_snapshot_reports-28-jun-2025-12-10pm.md
│   └── REPORTS-29-JUN-2025-01-16PM
│       └── report_code_snapshot_reports-29-jun-2025-01-16pm.md
├── requirements.txt
├── scripts
│   ├── analyze_artwork.py
│   └── generate_composites.py
├── settings
│   └── Master-Etsy-Listing-Description-Writing-Onboarding.txt
├── smart_sign_artwork.py
├── smart_sign_test01.py
├── sort_and_prepare_midjourney_images.py
├── static
│   ├── css
│   │   └── style.css
│   └── js
│       └── main.js
├── templates
│   ├── artworks.html
│   ├── composites_preview.html
│   ├── gallery.html
│   ├── index.html
│   ├── main.html
│   ├── mockup_selector.html
│   ├── review.html
│   └── review_artwork.html
└── tests
    └── test_analyze_artwork.py

```

---
## 📄 mockup_categorisation_log.txt

```txt
4x5-mockup-57.png -> Uncategorised
4x5-mockup-43.png -> Uncategorised
4x5-mockup-42.png -> Uncategorised
4x5-mockup-56.png -> Uncategorised
rainbow-serpent-dreaming-RC-009-THUMB-01.jpg -> Uncategorised
4x5-mockup-40.png -> Uncategorised
4x5-mockup-54.png -> Uncategorised
4x5-mockup-68.png -> Uncategorised
4x5-mockup-69.png -> Uncategorised
4x5-mockup-55.png -> Uncategorised
4x5-mockup-41.png -> Uncategorised
4x5-mockup-45.png -> Uncategorised
4x5-mockup-51.png -> Uncategorised
4x5-mockup-50.png -> Uncategorised
4x5-mockup-44.png -> Uncategorised
4x5-mockup-78.png -> Uncategorised
4x5-mockup-52.png -> Uncategorised
4x5-mockup-46.png -> Uncategorised
4x5-mockup-47.png -> Uncategorised
4x5-mockup-53.png -> Uncategorised
4x5-mockup-34.png -> Uncategorised
4x5-mockup-20.png -> Uncategorised
4x5-mockup-08.png -> Uncategorised
4x5-mockup-09.png -> Uncategorised
4x5-mockup-21.png -> Uncategorised
4x5-mockup-35.png -> Uncategorised
4x5-mockup-23.png -> Uncategorised
4x5-mockup-37.png -> Uncategorised
4x5-mockup-36.png -> Uncategorised
4x5-mockup-22.png -> Uncategorised
4x5-mockup-57.png -> Uncategorised
4x5-mockup-43.png -> Uncategorised
4x5-mockup-42.png -> Uncategorised
4x5-mockup-57.png -> Uncategorised
4x5-mockup-43.png -> Uncategorised
4x5-mockup-42.png -> Living Room
4x5-mockup-56.png -> Living Room
4x5-mockup-40.png -> Living Room
4x5-mockup-54.png -> Living Room
4x5-mockup-68.png -> Office
4x5-mockup-69.png -> Office
4x5-mockup-55.png -> Living Room
4x5-mockup-41.png -> Living Room
4x5-mockup-45.png -> Living Room
4x5-mockup-51.png -> Shelf Display
4x5-mockup-50.png -> Living Room
4x5-mockup-44.png -> Living Room
4x5-mockup-78.png -> Living Room
4x5-mockup-52.png -> Living Room
4x5-mockup-46.png -> Shelf Display
4x5-mockup-47.png -> Living Room
4x5-mockup-53.png -> Living Room
4x5-mockup-34.png -> Dining Room
4x5-mockup-20.png -> Shelf Display
4x5-mockup-08.png -> Framed Closeup
4x5-mockup-09.png -> Framed Closeup
4x5-mockup-21.png -> Living Room
4x5-mockup-35.png -> Living Room
4x5-mockup-23.png -> Framed Closeup
4x5-mockup-37.png -> Dining Room
4x5-mockup-36.png -> Dining Room
4x5-mockup-22.png -> Shelf Display
4x5-mockup-26.png -> Living Room
4x5-mockup-32.png -> Dining Room
4x5-mockup-33.png -> Living Room
4x5-mockup-27.png -> Living Room
4x5-mockup-19.png -> Shelf Display
4x5-mockup-31.png -> Dining Room
4x5-mockup-25.png -> Living Room
4x5-mockup-24.png -> Living Room
4x5-mockup-30.png -> Dining Room
4x5-mockup-18.png -> Shelf Display
4x5-mockup-15.png -> Office
4x5-mockup-01.png -> Living Room
4x5-mockup-29.png -> Living Room
4x5-mockup-28.png -> Living Room
4x5-mockup-14.png -> Office
4x5-mockup-02.png -> Living Room
4x5-mockup-16.png -> Shelf Display
4x5-mockup-17.png -> Shelf Display
4x5-mockup-03.png -> Nursery
4x5-mockup-07.png -> Framed Closeup
4x5-mockup-13.png -> Bedroom
4x5-mockup-12.png -> Bedroom
4x5-mockup-06.png -> Framed Closeup
4x5-mockup-38.png -> Dining Room
4x5-mockup-10.png -> Bedroom
4x5-mockup-04.png -> Living Room
4x5-mockup-05.png -> Nursery
4x5-mockup-11.png -> Bedroom
4x5-mockup-39.png -> Living Room
4x5-mockup-76.png -> Dining Room
4x5-mockup-62.png -> Office
4x5-mockup-63.png -> Gallery Wall
4x5-mockup-77.png -> Office
4x5-mockup-61.png -> Office
4x5-mockup-75.png -> Dining Room
4x5-mockup-49.png -> Living Room
4x5-mockup-48.png -> Shelf Display
4x5-mockup-74.png -> Living Room
4x5-mockup-60.png -> Hallway
4x5-mockup-58.png -> Office
4x5-mockup-64.png -> Gallery Wall
4x5-mockup-70.png -> Hallway
4x5-mockup-71.png -> Living Room
4x5-mockup-65.png -> Framed Closeup
4x5-mockup-59.png -> Framed Closeup
4x5-mockup-73.png -> Living Room
4x5-mockup-67.png -> Living Room
4x5-mockup-66.png -> Living Room
4x5-mockup-72.png -> Shelf Display
4x5-mockup-51.png -> Living Room
4x5-mockup-27.png -> Living Room
4x5-mockup-31.png -> Outdoors
4x5-mockup-03.png -> Nursery
4x5-mockup-11.png -> Bedroom
4x5-mockup-61.png -> Office
4x5-mockup-75.png -> Dining-Room
4x5-mockup-60.png -> Hallway

```

---
## 📄 smart_sign_test01.py

```py
# === [ SmartArt Sign System: CapitalArt Lite | by Robin Custance | Robbiefied Edition 2025 ] ===
# Batch signature overlay for digital artworks with smart colour/contrast detection and smoothing

# --- [ 1. Standard Imports | SS-1.0 ] ---
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import KMeans

# --- [ 2. CONFIGURATION | SS-2.0 ] ---
# [EDIT HERE for file paths, signature PNGs, main parameters]

# [2.1] Input/output folders (change as needed)
INPUT_IMAGE_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/SmartArt-sign-System/artwork-input"
OUTPUT_SIGNED_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/SmartArt-sign-System/artwork-signed-output"
SIGNATURE_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/SmartArt-sign-System/signatures"

# [2.2] Map colour names to PNG paths (just add new ones here!)
SIGNATURE_PNGS = {
    "beige": Path(SIGNATURE_DIR) / "beige.png",
    "black": Path(SIGNATURE_DIR) / "black.png",
    "blue": Path(SIGNATURE_DIR) / "blue.png",
    "brown": Path(SIGNATURE_DIR) / "brown.png",
    "gold": Path(SIGNATURE_DIR) / "gold.png",
    "green": Path(SIGNATURE_DIR) / "green.png",
    "grey": Path(SIGNATURE_DIR) / "grey.png",
    "odd": Path(SIGNATURE_DIR) / "odd.png",
    "red": Path(SIGNATURE_DIR) / "red.png",
    "skyblue": Path(SIGNATURE_DIR) / "skyblue.png",
    "white": Path(SIGNATURE_DIR) / "white.png",
    "yellow": Path(SIGNATURE_DIR) / "yellow.png"
}

# [2.3] RGB for contrast calc (update for any new sigs)
SIGNATURE_COLORS_RGB = {
    "beige": (245, 245, 220),
    "black": (0, 0, 0),
    "blue": (0, 0, 255),
    "brown": (139, 69, 19),
    "gold": (255, 215, 0),
    "green": (0, 255, 0),
    "grey": (128, 128, 128),
    "odd": (128, 128, 128),
    "red": (255, 0, 0),
    "skyblue": (135, 206, 235),
    "white": (210, 210, 210),
    "yellow": (255, 255, 0)
}

# [2.4] Signature sizing, margins, and smoothing params
SIGNATURE_SIZE_PERCENTAGE = 0.05    # 5% of long edge
SIGNATURE_MARGIN_PERCENTAGE = 0.03  # 3% from edge
SMOOTHING_BUFFER_PIXELS = 3         # Blur zone around signature
BLUR_RADIUS = 25                    # Big blur for smoothing
NUM_COLORS_FOR_ZONE_ANALYSIS = 2    # KMeans clusters for color

# --- [ 3. UTILITY FUNCTIONS | SS-3.0 ] ---

# [3.1] Luminance for contrast calculation (WCAG method)
def get_relative_luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# [3.2] Contrast ratio between two colours
def get_contrast_ratio(rgb1, rgb2):
    L1 = get_relative_luminance(rgb1)
    L2 = get_relative_luminance(rgb2)
    return (L1 + 0.05) / (L2 + 0.05) if L1 > L2 else (L2 + 0.05) / (L1 + 0.05)

# [3.3] Find dominant color(s) in a mask zone
def get_dominant_color_in_masked_zone(image_data_pixels, mask_pixels, num_colors=1):
    masked_pixels = [image_data_pixels[i] for i in range(len(mask_pixels)) if mask_pixels[i] > 0]
    if not masked_pixels:
        print("  [SS-3.3] No mask zone pixels—using black.")
        return (0, 0, 0)
    pixels_array = np.array(masked_pixels).reshape(-1, 3)
    if pixels_array.shape[0] < num_colors:
        return tuple(map(int, np.mean(pixels_array, axis=0)))
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto').fit(pixels_array)
    return tuple(map(int, kmeans.cluster_centers_[0]))

# [3.4] Choose signature colour for best contrast
def get_contrasting_signature_path(background_rgb, signature_colors_map, signature_paths_map):
    best_signature_name, max_contrast = None, -1.0
    for sig_name, sig_rgb in signature_colors_map.items():
        if not signature_paths_map.get(sig_name, '').is_file():
            continue
        contrast = get_contrast_ratio(background_rgb, sig_rgb)
        if contrast > max_contrast:
            max_contrast = contrast
            best_signature_name = sig_name
    if best_signature_name and best_signature_name in signature_paths_map:
        print(f"  [SS-3.4] '{best_signature_name}' signature wins (Contrast: {max_contrast:.2f})")
        return signature_paths_map[best_signature_name]
    print("  [SS-3.4] Fallback: using black.png")
    return signature_paths_map.get("black", None)

# --- [ 4. MAIN SIGNATURE FUNCTION | SS-4.0 ] ---

def add_smart_signature(image_path):
    try:
        # [4.1] Open image and prep variables
        with Image.open(image_path).convert("RGB") as img:
            width, height = img.size
            choose_right = random.choice([True, False])  # Randomise L/R
            long_edge = max(width, height)
            sig_target_size = int(long_edge * SIGNATURE_SIZE_PERCENTAGE)

            # [4.2] Dummy signature to get aspect ratio
            dummy_sig_path = list(SIGNATURE_PNGS.values())[0]
            with Image.open(dummy_sig_path).convert("RGBA") as dummy_sig:
                dsw, dsh = dummy_sig.size
                if dsw > dsh:
                    sig_w = sig_target_size
                    sig_h = int(dsh * (sig_w / dsw))
                else:
                    sig_h = sig_target_size
                    sig_w = int(dsw * (sig_h / dsh))
            margin_x = int(width * SIGNATURE_MARGIN_PERCENTAGE)
            margin_y = int(height * SIGNATURE_MARGIN_PERCENTAGE)
            sig_x = width - sig_w - margin_x if choose_right else margin_x
            sig_y = height - sig_h - margin_y

            # [4.3] Build shape mask for zone analysis & smoothing
            with Image.open(dummy_sig_path).convert("RGBA") as base_sig:
                base_sig_rs = base_sig.resize((sig_w, sig_h), Image.Resampling.LANCZOS)
                mask_canvas = Image.new("L", img.size, 0)
                mask_alpha = base_sig_rs.split()[-1]
                mask_canvas.paste(mask_alpha, (sig_x, sig_y))
                expanded_mask = mask_canvas.filter(ImageFilter.GaussianBlur(SMOOTHING_BUFFER_PIXELS))
                expanded_mask = expanded_mask.point(lambda x: 255 if x > 10 else 0)

            # [4.4] Analyse zone colour under mask
            img_rgb_data = list(img.getdata())
            mask_alpha_data = list(expanded_mask.getdata())
            dom_zone_color = get_dominant_color_in_masked_zone(img_rgb_data, mask_alpha_data, NUM_COLORS_FOR_ZONE_ANALYSIS)
            print(f"  [SS-4.4] Dominant colour: {dom_zone_color}")

            # [4.5] Make blurred zone for smooth signature blending
            patch_base = Image.new("RGB", img.size, dom_zone_color)
            patch_blur = patch_base.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
            patch_rgba = patch_blur.copy().convert("RGBA")
            patch_rgba.putalpha(expanded_mask)
            img = Image.alpha_composite(img.convert("RGBA"), patch_rgba).convert("RGB")

            # [4.6] Pick and overlay signature (best contrast)
            sig_path = get_contrasting_signature_path(dom_zone_color, SIGNATURE_COLORS_RGB, SIGNATURE_PNGS)
            if not sig_path or not Path(sig_path).is_file():
                print(f"  [SS-4.6] No valid sig PNG: {sig_path}")
                return
            with Image.open(sig_path).convert("RGBA") as sig_img:
                sig_img = sig_img.resize((sig_w, sig_h), Image.Resampling.LANCZOS)
                img.paste(sig_img, (sig_x, sig_y), sig_img)

            # [4.7] Save output
            out_path = Path(OUTPUT_SIGNED_DIR) / os.path.basename(image_path)
            img.save(out_path)
            print(f"✅ [SS-4.7] Signed: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"❌ [SS-4.ERR] {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()

# --- [ 5. BATCH EXECUTION | SS-5.0 ] ---
if __name__ == "__main__":
    Path(OUTPUT_SIGNED_DIR).mkdir(parents=True, exist_ok=True)
    print("\n--- [SS-5.0] Smart Signature Batch Processing ---")
    print(f"Input : {INPUT_IMAGE_DIR}")
    print(f"Output: {OUTPUT_SIGNED_DIR}")
    print(f"Sigs  : {SIGNATURE_DIR}")
    processed = 0
    for root, _, files in os.walk(INPUT_IMAGE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.gif')):
                img_path = Path(root) / file
                print(f"\nProcessing {file} ...")
                add_smart_signature(img_path)
                processed += 1
    print(f"\n--- [SS-5.DONE] Batch complete: {processed} files signed! ---")
    print(f"Check your output folder: {OUTPUT_SIGNED_DIR}")


```

---
## 📄 sort_and_prepare_midjourney_images.py

```py
import os
import csv
import math
import shutil
import re
from PIL import Image, ImageFilter
from collections import Counter
from pathlib import Path
import numpy as np

# === [ CapitalArt Lite: CONFIGURATION ] ===
# Paths are defined here for easy modification.
# Ensure these directories exist or will be created when run locally.

# The CSV file exported from Autojourney.
CSV_PATH = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/autojourney_downloader_export_26_06_2025_15_02_55.csv"

# The local directory where raw, untransformed Midjourney images are initially downloaded.
INPUT_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/autojourney"

# The base local directory where sorted images (within aspect ratio subfolders)
# and the new enriched metadata CSV will be stored.
OUTPUT_DIR = "/Users/robin/Documents/A1 Etsy-Artworks-Listing-Workshop/Art-Studio-Workshop/Midjourney Downloads/aspect-ratios"

# The full path and filename for the newly generated CSV file containing enriched image metadata.
GENERATED_CSV = Path(OUTPUT_DIR) / "capitalart_image_metadata.csv"

# Predefined aspect ratio categories (Width / Height) for sorting images.
ASPECT_CATEGORIES = {
    "1x1": 1.0, "2x3": 2 / 3, "3x2": 3 / 2, "3x4": 3 / 4,
    "4x3": 4 / 3, "4x5": 4 / 5, "5x4": 5 / 4, "5x7": 5 / 7,
    "7x5": 7 / 5, "9x16": 9 / 16, "16x9": 16 / 9,
    "A-Series-Vertical": 11 / 14, "A-Series-Horizontal": 14 / 11,
}
# Tolerance (as a decimal) for aspect ratio matching. Allows for slight variations.
ASPECT_TOLERANCE = 0.02

# Curated palette of RGB colors and their friendly names for dominant color detection (ETSY LIST).
PALETTE_RGB = {
    "Beige": (245, 245, 220),
    "Black": (0, 0, 0),
    "Blue": (0, 0, 255),
    "Bronze": (205, 127, 50),
    "Brown": (139, 69, 19),
    "Clear": (255, 255, 255), # Often represented as white/transparent in palettes
    "Copper": (184, 115, 51),
    "Gold": (255, 215, 0),
    "Grey": (128, 128, 128),
    "Green": (0, 255, 0),
    "Orange": (255, 165, 0),
    "Pink": (255, 192, 203),
    "Purple": (128, 0, 128),
    "Rainbow": (127, 127, 255), # Placeholder/fallback for multi-color or unmatchable
    "Red": (255, 0, 0),
    "Rose gold": (183, 110, 121),
    "Silver": (192, 192, 192),
    "White": (255, 255, 255),
    "Yellow": (255, 255, 0)
}
# Maximum Euclidean distance threshold for mapping an image pixel's RGB to a palette color.
MAX_DISTANCE_THRESHOLD = 100

# Number of dominant colors to extract and include in metadata.
NUM_DOMINANT_COLORS_TO_EXTRACT = 2 # <-- CHANGED TO 2

# === [ CapitalArt Lite: UTILITY FUNCTIONS ] ===

def euclidean_dist(c1, c2):
    """Calculates the Euclidean distance between two RGB color tuples."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def closest_palette_color(rgb):
    """
    Returns the name of the closest colour from the PALETTE_RGB based on Euclidean distance.
    Returns 'Unknown' if no close match is found within MAX_DISTANCE_THRESHOLD.
    """
    closest_name = "Unknown"
    min_dist = float('inf')
    for name, palette_rgb in PALETTE_RGB.items():
        d = euclidean_dist(rgb, palette_rgb)
        if d < min_dist:
            min_dist = d
            closest_name = name
    return closest_name if min_dist < MAX_DISTANCE_THRESHOLD else "Unknown"

def get_dominant_colours(img_path, num_colours=NUM_DOMINANT_COLORS_TO_EXTRACT):
    """
    Analyzes an image to find its top `num_colours` most prominent named colours,
    mapped to the predefined PALETTE_RGB. Returns a list of color names.
    """
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((100, 100)) # Resize for faster analysis
            pixels = list(img.getdata())
            counts = Counter(pixels)
            top_rgb = [rgb for rgb, _ in counts.most_common(num_colours)]
            
            # Map to palette colors and ensure only unique colors are returned,
            # taking the first N unique ones.
            found_colors = []
            for rgb_color in top_rgb:
                mapped_color = closest_palette_color(rgb_color)
                if mapped_color != "Unknown" and mapped_color not in found_colors:
                    found_colors.append(mapped_color)
                if len(found_colors) == num_colours: # Stop once we have enough unique colors
                    break
            
            # Fill remaining spots with "Unknown" if not enough unique colors were found
            while len(found_colors) < num_colours:
                found_colors.append("Unknown")
            
            return found_colors

    except Exception as e:
        print(f"⚠️ Error reading image for dominant colors {img_path}: {e}")
        return ["Unknown"] * num_colours # Return "Unknown" for all requested colors if error occurs

def calculate_sharpness(img_path):
    """
    Calculates a sharpness score for an image using the variance of the Laplacian.
    A higher value generally indicates a sharper image.
    Returns -1.0 on error.
    """
    try:
        with Image.open(img_path) as img:
            # Convert to grayscale for Laplacian calculation
            gray_img = img.convert("L")
            # Apply Laplacian filter and convert to numpy array to use .var()
            laplacian_var = np.var(np.array(gray_img.filter(ImageFilter.FIND_EDGES)))
            return laplacian_var
    except Exception as e:
        print(f"⚠️ Error calculating sharpness for {img_path}: {e}")
        return -1.0 # Indicate error

def classify_aspect(width, height):
    """
    Classifies an image's aspect ratio (width/height) into one of the
    predefined ASPECT_CATEGORIES, allowing for a specified tolerance.
    """
    if height == 0:
        return "Unclassified"
    actual_ratio = width / height
    for label, expected in ASPECT_CATEGORIES.items():
        if abs(actual_ratio - expected) <= ASPECT_TOLERANCE:
            return label
    return "Unclassified"

def clean_prompt(prompt):
    """
    Cleans and formats a Midjourney prompt string for use as a SEO-friendly filename.
    """
    prompt = re.sub(r"[^\w\s-]", "", prompt)
    prompt = re.sub(r"[_]+", "-", prompt)
    prompt = re.sub(r"\s+", "-", prompt.strip())
    cleaned = prompt[:100].strip("-").lower()
    return cleaned if cleaned else "untitled-artwork"

# === [ CapitalArt Lite: MAIN PROCESS ] ===

def process_images():
    """
    Main function to execute the image sorting, renaming, and metadata generation workflow.
    This function handles CSV loading, image processing (locally), file operations,
    and the final CSV generation, including enhanced metadata.
    """
    print("🧠 Starting CapitalArt Lite: Image Sort, Rename, and Metadata Generation...")
    print(f"Loading CSV from: {CSV_PATH}")
    print(f"Processing images from: {INPUT_DIR}")
    print(f"Outputting to: {OUTPUT_DIR}")

    # Load Autojourney CSV from local file system
    rows = []
    try:
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Successfully loaded {len(rows)} entries from CSV.")
    except FileNotFoundError:
        print(f"❌ Error: Autojourney CSV file not found at {CSV_PATH}. Please check the path in CONFIGURATION.")
        return
    except Exception as e:
        print(f"❌ Error loading Autojourney CSV: {e}")
        return

    # Ensure the base output directory exists locally
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Prepare CSV headers for the new metadata file, including new fields
    headers = [
        "Filename", "New Name", "Prompt", "Aspect Ratio", "Width", "Height",
        "Sharpness Score",
        "Primary Colour", # <-- CHANGED
        "Secondary Colour" # <-- CHANGED
    ]
    headers.extend(["Image URL", "Grid Image"]) # Add original URLs

    metadata_rows = [headers]

    processed_count = 0
    
    for row in rows:
        original_name = row.get("Filename", "").strip()
        prompt = row.get("Prompt", "").strip()
        image_url = row.get("Image url", "")
        grid_url = row.get("Grid image", "")

        if not original_name:
            print(f"⚠️ Skipping row due to missing 'Filename' in CSV: {row}")
            continue

        input_path = Path(INPUT_DIR) / original_name

        # Initialize placeholders
        width, height = "Unknown", "Unknown"
        aspect = "Unclassified"
        colors = ["Unknown"] * NUM_DOMINANT_COLORS_TO_EXTRACT
        sharpness_score = "Unknown"

        # --- LOCAL FILE PROCESSING ---
        if input_path.is_file():
            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    aspect = classify_aspect(width, height)
                    colors = get_dominant_colours(input_path, NUM_DOMINANT_COLORS_TO_EXTRACT)
                    sharpness_score = calculate_sharpness(input_path)
            except Exception as e:
                print(f"⚠️ Error processing image {original_name} (dimensions/colors/sharpness): {e}")
        else:
            print(f"❌ Missing image file: {original_name} (Expected at {input_path}). Skipping image processing.")
            # If the file is missing, we still want to add metadata to CSV if possible
            # but with 'Unknown' values for image-derived fields.

        original_ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".png"
        safe_name = clean_prompt(prompt)
        new_filename_base = f"{safe_name}{original_ext}"
        
        output_folder_path = Path(OUTPUT_DIR) / aspect
        output_folder_path.mkdir(parents=True, exist_ok=True) # Ensure aspect ratio folder exists

        final_new_filename = new_filename_base
        new_path = output_folder_path / new_filename_base
        counter = 1
        while new_path.exists(): # Check for existing file to avoid overwrites
            final_new_filename = f"{safe_name}-{counter}{original_ext}"
            new_path = output_folder_path / final_new_filename
            counter += 1

        # --- LOCAL FILE COPYING ---
        if input_path.is_file(): # Only copy if the source file exists
            try:
                shutil.copy2(input_path, new_path) # Copy the file, preserving metadata
                processed_count += 1
                print(f"✅ Processed & Copied: '{original_name}' -> '{aspect}/{final_new_filename}'")
            except Exception as e:
                print(f"❌ Error copying {original_name} to {new_path}: {e}")
                continue # Skip to the next image if copy fails
        else:
            print(f"✅ Processed metadata only for: '{original_name}' (Image file not found for copying)")
            # If image file not found, we still count it as processed metadata-wise
            processed_count += 1

        # Prepare the data row for CSV
        data_row = [
            original_name, final_new_filename, prompt, aspect, width, height,
            sharpness_score
        ]
        data_row.extend(colors) # Add the two extracted dominant colors
        data_row.extend([image_url, grid_url]) # Add original URLs

        metadata_rows.append(data_row)

    # Write the enriched metadata to the new CSV file
    try:
        with open(GENERATED_CSV, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(metadata_rows)
        print(f"\n📦 CapitalArt Lite process complete!")
        print(f"Successfully processed {processed_count} images/entries.")
        print(f"Enriched metadata written to: {GENERATED_CSV}")
    except Exception as e:
        print(f"❌ Error writing generated metadata CSV: {e}")

# === [ CapitalArt Lite: ENTRY POINT ] ===

if __name__ == "__main__":
    process_images()
```

---
## 📄 capitalart-total-nuclear-v2.py

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 🔥 CapitalArt Total Nuclear Snapshot v2
# 🚀 Ultimate Dev Toolkit by Robbie Mode™
# python3 capitalart-total-nuclear-v2.py
# python3 capitalart-total-nuclear-v2.py --no-zip
# python3 capitalart-total-nuclear-v2.py --skip-git --skip-env
# ===========================================

# --- [ 1a: Standard Library Imports | nuclear-1a ] ---
import os
import sys
import datetime
import subprocess
import zipfile
import py_compile
from pathlib import Path
from typing import Generator
import argparse
from stat import S_ISREG

# --- [ 1b: Snapshot Configuration | nuclear-1b ] ---
ALLOWED_EXTENSIONS = {".py", ".sh", ".jsx", ".txt", ".html", ".js", ".css"}
EXCLUDED_EXTENSIONS = {".json"}
EXCLUDED_FOLDERS = {"venv", ".venv", "__MACOSX", ".git", ".vscode"}
EXCLUDED_FILES = {".DS_Store"}

# ===========================================
# 2. 🧭 Timestamp + Report Folder
# ===========================================

def get_timestamp() -> str:
    return datetime.datetime.now().strftime("REPORTS-%d-%b-%Y-%I-%M%p").upper()

def create_reports_folder() -> Path:
    timestamp = get_timestamp()
    folder_path = Path("reports") / timestamp
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created report folder: {folder_path}")
    return folder_path

# ===========================================
# 3. 🔍 Collect Valid Files
# ===========================================

def get_included_files() -> Generator[Path, None, None]:
    for path in Path(".").rglob("*"):
        if (
            path.is_file()
            and path.suffix in ALLOWED_EXTENSIONS
            and path.suffix not in EXCLUDED_EXTENSIONS
            and path.name not in EXCLUDED_FILES
            and not any(part in EXCLUDED_FOLDERS for part in path.parts)
        ):
            yield path

# ===========================================
# 4. 📄 Markdown Code Snapshot
# ===========================================

def gather_code_snapshot(folder: Path) -> Path:
    md_path = folder / f"report_code_snapshot_{folder.name.lower()}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# 🧠 CapitalArt Code Snapshot — {folder.name}\n\n")
        for file in get_included_files():
            rel_path = file.relative_to(Path("."))
            print(f"📄 Including file: {rel_path}")
            md_file.write(f"\n---\n## 📄 {rel_path}\n\n```{file.suffix[1:]}\n")
            try:
                with open(file, "r", encoding="utf-8") as f:
                    md_file.write(f.read())
            except Exception as e:
                md_file.write(f"[ERROR READING FILE: {e}]")
            md_file.write("\n```\n")
    print(f"✅ Code snapshot saved to: {md_path}")
    return md_path

# ===========================================
# 5. 📊 File Summary Table
# ===========================================

def generate_file_summary(folder: Path) -> None:
    summary_path = folder / "file_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 📊 File Summary\n\n")
        f.write("| File | Size (KB) | Last Modified |\n")
        f.write("|------|------------|----------------|\n")
        for file in get_included_files():
            size_kb = round(file.stat().st_size / 1024, 1)
            mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            rel = file.relative_to(Path("."))
            f.write(f"| `{rel}` | {size_kb} KB | {mtime:%Y-%m-%d %H:%M} |\n")
    print(f"📋 File summary written to: {summary_path}")

# ===========================================
# 6. 🧪 Python Syntax Validation
# ===========================================

def validate_python_files() -> None:
    print("\n🧪 Validating Python syntax...")
    for file in get_included_files():
        if file.suffix == ".py":
            try:
                py_compile.compile(file, doraise=True)
                print(f"✅ {file}")
            except py_compile.PyCompileError as e:
                print(f"❌ {file} → {e.msg}")

# ===========================================
# 7. 🧬 Git Status + Commit Info
# ===========================================

def log_git_status(folder: Path) -> None:
    git_path = folder / "git_snapshot.txt"
    with open(git_path, "w", encoding="utf-8") as f:
        f.write("🔧 Git Status:\n")
        subprocess.run(["git", "status"], stdout=f, stderr=subprocess.DEVNULL)
        f.write("\n🔁 Last Commit:\n")
        subprocess.run(["git", "log", "-1"], stdout=f, stderr=subprocess.DEVNULL)
        f.write("\n🧾 Diff Summary:\n")
        subprocess.run(["git", "diff", "--stat"], stdout=f, stderr=subprocess.DEVNULL)
    print(f"📘 Git snapshot written to: {git_path}")

# ===========================================
# 8. 🌐 Environment Metadata
# ===========================================

def log_environment_details(folder: Path) -> None:
    env_path = folder / "env_metadata.txt"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("🐍 Python Version:\n")
        subprocess.run(["python3", "--version"], stdout=f)
        f.write("\n🖥️ Platform Info:\n")
        subprocess.run(["uname", "-a"], stdout=f)
        f.write("\n📦 Installed Packages:\n")
        subprocess.run(["pip", "freeze"], stdout=f)
    print(f"📚 Environment metadata saved to: {env_path}")

# ===========================================
# 9. 📦 Zip the Report Folder
# ===========================================

def zip_report_folder(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in folder.rglob("*"):
            zipf.write(file, file.relative_to(folder.parent))
    print(f"📦 Report zipped to: {zip_path}")
    return zip_path

# ===========================================
# 10. 🧰 CLI Args
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(description="CapitalArt Dev Snapshot Generator")
    parser.add_argument("--no-zip", action="store_true", help="Skip ZIP archive creation")
    parser.add_argument("--skip-env", action="store_true", help="Skip environment metadata logging")
    parser.add_argument("--skip-validate", action="store_true", help="Skip Python syntax validation")
    parser.add_argument("--skip-git", action="store_true", help="Skip Git snapshot logging")
    return parser.parse_args()

# ===========================================
# 11. 🚀 Main Execution
# ===========================================

def main():
    args = parse_args()
    print("🎨 Generating CapitalArt Total Nuclear Snapshot (v2)...")

    report_folder = create_reports_folder()
    gather_code_snapshot(report_folder)
    generate_file_summary(report_folder)

    if not args.skip_validate:
        validate_python_files()
    if not args.skip_env:
        log_environment_details(report_folder)
    if not args.skip_git:
        log_git_status(report_folder)
    if not args.no_zip:
        zip_report_folder(report_folder)

    print("✅ Snapshot complete. All systems green, Robbie! 💚")

# ===========================================
# 🔚 Entry Point
# ===========================================

if __name__ == "__main__":
    main()

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
## 📄 settings/Master-Etsy-Listing-Description-Writing-Onboarding.txt

```txt
Here’s a comprehensive, step-by-step instruction set you can give to any future ChatGPT, OpenAI assistant, or even a human VA, to get them up to your exact standards for writing your Pulitzer-worthy Etsy artwork listings. This covers all your requirements, context, and critical nuances. You can copy, save, or tweak as needed!

🥇 Robin Custance Etsy Listing Writing – Master Instructions (2025)
Purpose:
Generate unique, professional, search-optimised artwork listings for Aboriginal and Australian digital art, to be sold as high-resolution digital downloads on Etsy or other platforms.

1. OVERVIEW
You are acting as a renowned art curator and professional art writer.
You will write the artwork listing (description, tags, materials, SEO filename, etc.) for digital Aboriginal/Australian art by Robin Custance (Robbie) for use on Etsy.
Your primary goal:
Create a unique, high-quality, fully search-optimised and art-accurate listing for each artwork—never formulaic, never generic.

2. FILES & ARTWORK
Each artwork comes with:

The original file name (used for admin/traceability only—never assume the filename is a reliable guide to the content or story)

A visual or other cues for the artwork itself

Sometimes reference docs for tag/material style or templates

3. FINAL OUTPUT FORMAT (REQUIRED FIELDS & ORDER)
For each artwork, output the following, in this exact order and style:

Original File Name
(original filename here, e.g. robin.5_A_serene_and_luminous_depiction_of_Waterhole_Dreaming_i_cc09e9da-b50b-418d-a2bd-ce9cb25489e7_1.jpg)

SEO Filename:
(A new, SEO-optimised filename: see parameters below)

Artwork Title:
(A unique, SEO-rich, buyer-friendly title—see parameters below)

Artwork Description:
(A Pulitzer-worthy, 400+ word, search-optimised and visually/artistically accurate description. See guidelines.)

Tags:
(tag1, tag2, tag3, ..., tag13)
(comma-separated, max 13, see parameters)

Materials:
(material1, material2, ..., material13)
(comma-separated, max 13, see parameters)

4. KEY PARAMETERS & REQUIREMENTS
SEO Filename:
Max 70 characters (including spaces)

Ends with: Artwork-by-Robin-Custance-RJC-XXXX.jpg (replace XXXX with SKU or sequential number as needed)

Begins with: 2–3 words from the artwork title, hyphen-separated, relating to the subject (not copied blindly from the original file name, and never padded)

No fluff, no filler

Artwork Title:
Max 140 characters

Must include or strongly reflect:

“High Resolution”

“Digital”

“Download”

“Artwork by Robin Custance” (or a direct equivalent)

Combine with clear, high-search terms a buyer would use (e.g., “Dot Art”, “Aboriginal Print”, “Australian Wall Art”, “Instant Download”, “Printable” etc.)

First 20 words: Strong search terms and subject clarity

NO “Step into,” “Immerse yourself,” “Discover the,” etc.

Artwork Description:
400+ words, no padding

Pulitzer-quality, unique, non-formulaic, and never “cookie-cutter”

Must be written by visually/artistic merit of the artwork (NOT the original file name unless it matches!)

Focus on: technique, visual detail, style, colour, artistic inspiration, story, cultural context (when appropriate)

High keyword density but always naturally written

Sensory, evocative, and professional, but never fluffy or generic

NO instructions, printing info, or general shop/artist bio (these are provided elsewhere)

NO leading “discover/step into/immerse/experience” phrases; assume the buyer is already viewing the artwork and looking for details

Tags:
13 max

Comma separated

Highly targeted to the specific artwork (style, subject, feeling, use, context)

Rotate/search-stack: Don’t simply copy the same tags every time

Include a mix of art technique, subject, Australian/Aboriginal terms, artist branding, and digital wall art phrases

Examples:
waterhole art, aboriginal dot painting, australian wall art, robin custance art, digital aboriginal print, outback home decor, sacred waterhole, high resolution art, modern dot artwork, indigenous australian art, contemporary dreamtime, native australian print, calming wall decor

Materials:
13 max

Comma separated

Up to 45 characters per entry

NO hyphens or repetition unless it’s truly accurate

Rotate and vary across listings; tailor to each artwork’s technique, style, and digital nature

Examples (mix and match as truly applies):
Digital artwork, High resolution JPEG file, Original digital painting, Digital brushwork, Australian art file, Dot art JPEG, Downloadable image, Archival quality digital image, Contemporary art digital file, Printable wall art, Professional digital design, Layered digital composition, High clarity art download

5. NUANCES & GOLDEN RULES
Never rely on the original file name for artwork description inspiration. Only use it for SKU tracking or reference. Always base descriptions and tags on the artwork’s visual, thematic, and artistic content.

Descriptions must be unique—do not reuse language, sentence structure, or tags across listings.

SEO filename and title must maximise search value but still sound like a human wrote them—no keyword stuffing.

Never include “how to print,” shop policies, or about the artist in this description section (these are added elsewhere).

No unnecessary intro sentences—every word should serve either SEO, curation, or compelling visual/artistic analysis.

Every tag and material should be handpicked for each artwork. If two listings use the same tag/material, it’s because both genuinely fit.

Stay within all Etsy limitations:

13 tags/materials max

Tag: 20 chars max each

Materials: 45 chars max each

SEO filename: 70 chars max

Title: 140 chars max

6. EXAMPLE OUTPUT (for future reference)
Original File Name
robin.5_A_serene_and_luminous_depiction_of_Waterhole_Dreaming_i_cc09e9da-b50b-418d-a2bd-ce9cb25489e7_1.jpg

SEO Filename:
Waterhole-Dreaming-Dot-Artwork-by-Robin-Custance-RJC-0032.jpg

Artwork Title:
Waterhole Dreaming Dot Art – High Resolution Digital Aboriginal Print, Instant Download | Calming Outback Artwork by Robin Custance

Artwork Description:
Luminous, tranquil, and full of life—“Waterhole Dreaming” is a high-resolution digital Aboriginal dot painting by Robin Custance. Inspired by Australia’s ancient waterholes, this vibrant artwork weaves together shimmering golds, deep blues, and subtle whites, each dot telling a story of gathering, renewal, and connection. The flowing, circular motifs reflect the cycles of nature and community central to the Dreaming, while modern digital detailing brings a fresh, contemporary energy.

This piece celebrates both the rich traditions of Aboriginal art and the bold potential of digital creation. The intricate layers invite you to explore the painting’s depths, finding new patterns and meaning every time you look. “Waterhole Dreaming” is more than just wall decor—it’s a daily reminder of resilience, harmony, and the sacred link between land, water, and story.

Ideal for Australian art lovers, dot painting fans, or anyone wanting to add a touch of calm and culture to their home, this digital download delivers gallery-quality clarity for striking prints at any size.

Robin Custance’s unique blend of tradition and innovation ensures every detail stands out, from the tiniest dot to the largest sweep of colour. Hang it in your living room, entryway, or creative space and let the spirit of the outback flow through your world.

Tags:
waterhole art, aboriginal dot painting, australian wall art, robin custance art, digital aboriginal print, outback home decor, sacred waterhole, high resolution art, modern dot artwork, indigenous australian art, contemporary dreamtime, native australian print, calming wall decor

Materials:
Digital artwork, High resolution JPEG file, Original digital painting, Digital brushwork, Australian art file, Dot art JPEG, Downloadable image, Archival quality digital image, Contemporary art digital file, Printable wall art, Professional digital design, Layered digital composition, High clarity art download

7. QUICK SUMMARY FOR ALL FUTURE ASSISTANTS:
All outputs must strictly follow the format and parameters above.

Never shortcut, pad, or reuse language—each artwork’s listing must be bespoke.

If unsure, ask for clarification before outputting.

Remember: You are writing as a top-tier curator and art copywriter, not a bot or a mass lister!

8. JSON OUTPUT (MANDATORY)
Every response must be a single valid JSON object with these exact keys:
seo_filename, title, description, tags, materials, primary_colour, secondary_colour.
Description must be at least 400 words and contain no markdown or HTML.
Tags and materials are arrays of strings, maximum 13 entries each.
Return only the JSON object without any extra text or formatting.

End of Instructions
(Feel free to add or update any of these standards as your business grows, Robbie!)
```

---
## 📄 tests/test_analyze_artwork.py

```py
import json
import os
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("OPENAI_API_KEY", "test")
import analyze_artwork as aa


def dummy_openai_response(content):
    class Choice:
        def __init__(self, text):
            self.message = type('m', (), {'content': text})
    class Resp:
        def __init__(self, text):
            self.choices = [Choice(text)]
    return Resp(content)


def run_test():
    sample_json = json.dumps({
        "seo_filename": "test-artwork-by-robin-custance-rjc-0001.jpg",
        "title": "Test Artwork – High Resolution Digital Aboriginal Print",
        "description": "Test description " * 50,
        "tags": ["test", "digital art"],
        "materials": ["Digital artwork", "High resolution JPEG file"],
        "primary_colour": "Black",
        "secondary_colour": "Brown"
    })
    with mock.patch.object(aa.client.chat.completions, 'create', return_value=dummy_openai_response(sample_json)):
        system_prompt = aa.read_onboarding_prompt()
        img = next(Path('inputs/artworks').rglob('*.jpg'))
        status = []
        entry = aa.analyze_single(img, system_prompt, None, status)
        print(json.dumps(entry, indent=2)[:200])


if __name__ == '__main__':
    run_test()


```

---
## 📄 generic_texts/2x3.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

2:3 (Vertical)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This captivating artwork is provided in a classic 2:3 portrait ratio, ideal for emphasizing height and creating a sense of grandeur. Your high-resolution JPEG file boasts 9600 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints. 

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 32 x 48 inches (81.3 x 121.9 cm) without compromising quality.

Ideal Uses for the 2:3 Aspect Ratio: The elegant 2:3 portrait ratio is perfect for compositions that benefit from vertical emphasis, such as towering landscapes, architectural marvels, or captivating figure studies. It draws the viewer's eye upward, creating a feeling of spaciousness and depth.

* 10x15" (25x38 cm): Great for narrow wall sections, intimate spaces, or as part of a vertical grouping.
* 16x24" (40x60 cm): A popular choice for classic framing, offering a timeless aesthetic in living rooms or studies.
* 20x30" (50x76 cm): A versatile medium size that brings a refined vertical element to any room.
* 24x36" (60x91 cm): Creates a powerful visual statement in hallways, entryways, or as a focal point in a master bedroom.
* 32x48" (81x122 cm): A grand print that truly commands attention and fills a large wall in open, spacious environments.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.


Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 generic_texts/4x5.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

4:5 (Vertical)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This exquisite artwork is provided in a harmonious 4:5 portrait ratio, widely popular for its balanced and pleasing proportions. Your high-resolution JPEG file boasts 11520 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 38.4 x 48 inches (97.5 x 121.9 cm) without compromising quality.

Ideal Uses for the 4:5 Aspect Ratio: The 4:5 portrait ratio is highly sought after for its aesthetically pleasing balance, often seen in fine art photography and portraiture. It provides ample vertical space while still feeling grounded, making it perfect for a wide range of subjects.

* 10x12.5" (25x32 cm): Excellent for bedside tables, mantels, or small gallery groupings, offering an intimate scale.
* 16x20" (40x50 cm): Incredibly versatile and fits well in almost any room, from living areas to studies, providing a classic proportion.
* 24x30" (60x76 cm): Makes a substantial and elegant statement, perfect for a dining room or foyer, drawing the eye effortlessly.
* 32x40" (81x102 cm): A commanding size that creates a significant visual impact in medium to large rooms.
* 38.4x48" (97.5x122 cm): Designed to be a captivating centerpiece in grand interiors, filling the wall with artistic presence.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!


```

---
## 📄 generic_texts/16x9.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

16:9 (Horizontal)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This dynamic artwork is provided in a cinematic 16:9 landscape ratio, delivering an expansive, widescreen experience. Your high-resolution JPEG file boasts 14400 x 8100 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints. 

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 27 inches (121.9 x 68.6 cm) without compromising quality.

Ideal Uses for the 16:9 Aspect Ratio: The 16:9 landscape ratio, familiar from film and television screens, is superb for capturing sweeping panoramas, wide-angle views, and immersive action scenes. It draws the viewer into the scene, making it feel grand and enveloping.

* 17.8x10" (45x25 cm): A natural fit for spaces where screens are prominent, like home offices or entertainment areas, or above smaller furniture.
* 24x13.5" (61x34 cm): Provides a contemporary widescreen element suitable for various rooms, offering a subtle yet expansive feel.
* 32x18" (81x45 cm): Works wonderfully above a sofa or console table, creating a modern and immersive focal point.
* 40x22.5" (102x57 cm): A significant widescreen piece that brings a sense of cinematic grandeur to a large wall.
* 48x27" (122x68.6 cm): Transforms a wall into a cinematic vista, perfect for large living rooms, media rooms, or open-plan designs.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.


Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 generic_texts/A-Series-Verical.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—proud Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour and dots. My journey began on Kaurna Country in Adelaide, and my family lines stretch back to Naracoorte (Boandik Country). Every artwork I create is a blend of ancient wisdom and modern wonder, with deep respect for Country and the powerful stories passed down through generations. My art is my way of giving back—celebrating our land, our history, and sharing a piece of Australian spirit with you, wherever you are in the world.

Dot painting, for me, is more than a technique—it’s a conversation with the past and a bridge to the future. I use both traditional and digital tools to honour old stories and invent new ones, always with a healthy dose of humour and warmth. When you bring my art into your home, you’re not just getting a pretty picture—you’re getting a piece of living culture, carefully crafted with heart and soul.

Did You Know? The Spirit & Story of Aboriginal Dot Painting
Aboriginal dot painting is one of Australia’s most iconic and respected art traditions, rooted in tens of thousands of years of deep connection to land, people, and spirit. While dot motifs have appeared in ancient rock art and ceremonial body painting across the continent, the world-famous “dot painting” style really took off in the early 1970s with the Papunya Tula art movement in the Central Desert.

Pioneering artists such as Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye created a whole new visual language—using dots to map out Dreamtime stories, songlines, and the sacred wisdom of Country. Dots were never just decoration: they protected the most important meanings from outsiders, while sharing the beauty, energy, and movement of the land with everyone.

Each dot can be seen as a footprint, a star, or a ripple in the sand—a symbol of connection between artist, Country, and viewer. Today, Aboriginal artists continue to keep this powerful tradition alive, blending ancient methods with new ideas, materials, and personal stories. No two dot paintings are ever the same; each one is a living story, capturing the resilience, creativity, and ongoing journey of Aboriginal people and places.

A-Series (Vertical)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This elegant artwork is provided in the versatile A-Series vertical ratio, aligning with international standard paper sizes and offering a clean, contemporary aesthetic. Your high-resolution JPEG file boasts 10182 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 33.94 x 48 inches (86.2 x 121.9 cm) (based on A1 dimensions, scaled) without compromising quality.

Ideal Uses for the A-Series Vertical Aspect Ratio: The A-Series vertical format is characterized by its consistent proportions across different sizes (e.g., A4, A3, A2), making it perfect for cohesive gallery walls or when you need standard framing options. It's ideal for art that benefits from a clear, structured vertical presentation.

* 10x14.1" (25x36 cm): A versatile size that fits well in smaller vertical spaces, perfect for studies or bedrooms.
* 11.7x16.5" (A3) (29.7x42 cm): Excellent for creating uniform gallery walls, especially when mixed with other A-series prints, a professional look.
* 16.5x23.4" (A2) (42x59.4 cm): A substantial size that adds a refined and structured vertical element to living areas or offices.
* 23.4x33.1" (A1) (59.4x84.1 cm): Makes a significant impression, suitable for larger wall spaces in a contemporary setting.
* 33.94x48" (86.2x122 cm): A truly grand sized piece that delivers exceptional visual presence and simplifies framing for large-scale impact.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!


```

---
## 📄 generic_texts/4x3.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

4:3 (Horizontal)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This compelling artwork is provided in a classic 4:3 landscape ratio, offering a slightly more square-like horizontal composition. Your high-resolution JPEG file boasts 14400 x 10800 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 36 inches (121.9 x 91.4 cm) without compromising quality.

Ideal Uses for the 4:3 Aspect Ratio: The 4:3 landscape ratio offers a broader, more open feel than traditional landscape formats, making it excellent for showcasing scenes with interesting foregrounds or expansive skies. It provides a comfortable viewing experience, resembling many digital display formats.

* 13.3x10" (34x25 cm): Fits well into smaller spaces like kitchen nooks, hallways, or compact offices.
* 24x18" (60x45 cm): A popular size for placing above sofas, sideboards, or console tables, creating a balanced horizontal line.
* 32x24" (81x60 cm): Fantastic for creating a relaxed yet impactful atmosphere in media rooms or lounges.
* 40x30" (102x76 cm): A substantial piece that brings depth and breadth to larger living areas or commercial spaces.
* 48x36" (122x91 cm): For a truly immersive experience, this print fills a wall with a grand and inviting presence.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 generic_texts/9x16.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

9:16 (Vertical)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This striking artwork is provided in a contemporary 9:16 portrait ratio, mirroring modern screen dimensions and offering a dramatic vertical emphasis. Your high-resolution JPEG file boasts 8100 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 27 x 48 inches (68.6 x 121.9 cm) without compromising quality.

Ideal Uses for the 9:16 Aspect Ratio: The 9:16 portrait ratio, often referred to as "vertical video" or "phone screen" format, is perfect for subjects that demand extreme verticality, such as towering natural formations, dramatic figure studies, or stylized architectural compositions. It creates a modern, immersive viewing experience.

* 10x17.8" (25x45 cm): Perfectly suited for slender wall sections, narrow nooks, or as a unique accent.
* 13.5x24" (34x61 cm): Offers a sleek, modern look in offices or minimalist living areas, drawing the eye upward.
* 18x32" (45x81 cm): Creates a significant vertical presence, ideal for adding height and drama to a room.
* 22.5x40" (57x102 cm): A commanding vertical piece that makes a strong artistic statement in contemporary spaces.
* 27x48" (68.6x122 cm): Creates a powerful, extreme vertical statement, ideal for high-ceiling rooms or as a striking focal point.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!



```

---
## 📄 generic_texts/5x7.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

5:7 (Vertical)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This stunning artwork is provided in a traditional 5:7 portrait ratio, a classic choice for showcasing detail and elegance. Your high-resolution JPEG file boasts 10286 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 34.3 x 48 inches (87.1 x 121.9 cm) without compromising quality.

Ideal Uses for the 5:7 Aspect Ratio: The 5:7 portrait ratio is a time-honored choice, often associated with framed photographs and fine art. Its slightly narrower proportion guides the eye directly to the subject, making it excellent for portraits, botanical studies, or any composition that benefits from a focused vertical emphasis.

* 10x14" (25x35 cm): Perfect for adding charm to side tables, mantels, or bookcases, providing a classic, refined touch.
* 15x21" (38x53 cm): Fits beautifully into smaller wall spaces or as part of a curated collection, offering a graceful vertical line.
* 20x28" (50x70 cm): An elegant mid-size option that brings sophistication to studies, hallways, or bedrooms.
* 25x35" (63.5x89 cm): Provides a substantial and elegant statement in dining rooms or studies, enhancing the room's ambiance.
* 34.3x48" (87.1x122 cm): For a truly commanding and refined presence, this large print will elevate the ambiance of any spacious room.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.


Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 generic_texts/7x5.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

7:5 (Horizontal)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This elegant artwork is provided in a classic 7:5 landscape ratio, offering a slightly more elongated horizontal view. Your high-resolution JPEG file boasts 14400 x 10286 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 34.3 inches (121.9 x 87.1 cm) without compromising quality.

Ideal Uses for the 7:5 Aspect Ratio: The 7:5 landscape ratio is a timeless and versatile choice, providing a more expansive horizontal canvas than a 4:3, yet more contained than a 16:9. It's ideal for capturing broad scenes while retaining intimacy, making it excellent for landscapes, cityscapes, or group compositions.

* 14x10" (35x25 cm): A great addition to horizontal display areas like shelves or console tables, offering a classic aesthetic.
* 21x15" (53x38 cm): Fits wonderfully above a dining room buffet or a narrower stretch of wall, providing a harmonious balance.
* 28x20" (70x50 cm): Brings a sophisticated touch to living rooms or large hallways, creating a compelling visual.
* 35x25" (89x63.5 cm): A substantial piece that beautifully captures broad scenes and enhances the spaciousness of a room.
* 48x34.3" (122x87.1 cm): Designed to command attention in spacious, open-plan areas with its impressive and immersive view.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!



```

---
## 📄 generic_texts/1x1.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

1:1 (Square)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This stunning artwork is provided in a versatile 1:1 square ratio, perfect for creating symmetrical displays and fitting a wide range of frames. Your high-resolution JPEG file boasts 14400 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints. 

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 48 inches (121.9 x 121.9 cm) without compromising quality.

Ideal Uses for the 1:1 Aspect Ratio: The timeless 1:1 square ratio offers balanced harmony, making it a versatile choice for a variety of spaces. Its inherent symmetry brings a sense of calm and order, perfect for creating visually pleasing displays.

* 10x10" (25x25 cm): Great for small spaces, shelves, or as part of a multi-piece grid.
* 16x16" (40x40 cm): Perfect for modern gallery walls or to complement other artwork.
* 24x24" (60x60 cm): A stylish medium size that suits any room or office, offering a balanced presence.
* 36x36" (90x90 cm): A large statement piece for significant impact in open areas.
* 48x48" (122x122 cm): A huge statement piece for big impact in open areas, truly dominating a wall.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 generic_texts/3x2.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

3:2 (Horizontal)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This striking artwork is provided in a timeless 3:2 landscape ratio, perfect for panoramic views and expansive scenes. Your high-resolution JPEG file boasts 14400 x 9600 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 32 inches (121.9 x 81.3 cm) without compromising quality.

Ideal Uses for the 3:2 Aspect Ratio: The classic 3:2 landscape ratio excels at capturing broad vistas, serene horizons, and dynamic action shots. It naturally guides the eye across the image, making it perfect for subjects that benefit from a wider field of view.

* 15x10" (38x25 cm): Excellent for adding a horizontal element to shelves, desks, or mantels in smaller areas.
* 24x16" (60x40 cm): Ideally suited for hanging above sideboards, console tables, or in a den.
* 30x20" (76x50 cm): A versatile medium size that looks great over a sofa or in a dining room, offering a broad view.
* 36x24" (91x60 cm): Provides a grand, immersive experience, perfect for dining rooms or large office spaces.
* 48x32" (122x81 cm): Designed to truly transform the atmosphere of a spacious room with its expansive presence.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!


```

---
## 📄 generic_texts/5x4.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

5:4 (Horizontal)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This beautiful artwork is provided in a classic 5:4 landscape ratio, known for its balanced, almost-square horizontal presentation. Your high-resolution JPEG file boasts 14400 x 11520 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 38.4 inches (121.9 x 97.5 cm) without compromising quality.

Ideal Uses for the 5:4 Aspect Ratio: The 5:4 landscape ratio offers a slightly more contained horizontal view, providing a sense of stability and focus. It's a fantastic choice for compositions where every element contributes to the overall scene, such as still life, architectural details, or serene landscapes.

* 12.5x10" (32x25 cm): Perfect for adding a touch of art to workspaces, kitchen counters, or compact display areas.
* 20x16" (50x40 cm): Integrates seamlessly into a gallery wall or as a standalone piece in a medium-sized room, offering a pleasant balance.
* 30x24" (76x60 cm): Creates a calm yet impactful presence in living rooms or dining areas, a versatile and popular choice.
* 40x32" (102x81 cm): A significant piece that brings a sense of grounded beauty and expansive detail to your space.
* 48x38.4" (122x97.5 cm): Delivers a broad, captivating view for expansive walls, making a powerful and elegant statement.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 generic_texts/3x4.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

3:4 (Vertical)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This captivating artwork is provided in a versatile 3:4 portrait ratio, offering a slightly wider vertical composition than the 2:3. Your high-resolution JPEG file boasts 10800 x 14400 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 36 x 48 inches (91.4 x 121.9 cm) without compromising quality.

Ideal Uses for the 3:4 Aspect Ratio: The 3:4 portrait ratio provides a balanced vertical emphasis that feels modern and approachable. It's an excellent choice for art that features strong vertical elements but also benefits from a bit more width, such as still life, portraits, or architectural details.

* 10x13.3" (25x34 cm): Perfect for bedrooms or home offices, offering an intimate feel on a small wall.
* 18x24" (45x60 cm): Integrates beautifully into gallery wall arrangements or can be paired with other artworks.
* 24x32" (60x81 cm): Makes a significant impact in living rooms or entryways, providing a strong vertical anchor.
* 30x40" (76x102 cm): A substantial piece that commands attention in larger rooms without overwhelming the space.
* 36x48" (91x122 cm): For a truly commanding presence, this print will be a focal point in any large room or open-plan area.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!
```

---
## 📄 generic_texts/A-Series-Horizontal.txt

```txt
About the Artist – Robin Custance
G’day, I’m Robin Custance—Aboriginal Aussie artist, part-time Kangaroo whisperer, and lifelong storyteller through colour, line, and imagination. My journey began on Kaurna Country here in Adelaide, and my roots reach back to the Naracoorte region (Boandik Country). 

Every artwork I create—whether it’s dot painting, digital design, or contemporary Aussie landscape—carries a deep respect for Country, community, and the vibrant stories handed down through generations.

No matter the style or subject, my intention as an artist is always to tell stories and share genuine emotion. Whether I’m working with ancient dot techniques, digital brushes, or bold, modern colours, I pour heart and meaning into every piece.

My art is a way to give back and keep culture strong—blending the timeless with the modern, and weaving together my heritage with today’s creative spirit. When you bring my art into your home, you’re welcoming a piece of Australian story and soul, crafted with passion and a fair bit of character.

Did You Know? Aboriginal Art & the Spirit of Dot Painting
Aboriginal art is one of the world’s oldest creative traditions, with roots stretching back tens of thousands of years. From ancient rock art to vibrant canvases, each style and technique tells a story about connection to land, people, and spirit.

Did you know that the famous “dot painting” style only became widely known in the early 1970s, through the ground-breaking Papunya Tula movement in Central Australia? Visionary artists like Kaapa Tjampitjinpa, Clifford Possum Tjapaltjarri, and Emily Kame Kngwarreye developed a new visual language using dots—not just for beauty, but to share and protect sacred Dreamtime stories and songlines.

In these works, each dot might represent a footprint, a star, or a ripple in the sand—a powerful link between artist, Country, and viewer.

Today, Aboriginal artists across Australia continue to adapt, innovate, and honour tradition in every medium—from dot paintings and digital art to sculpture and contemporary works. Every piece is part of a living story: a testament to resilience, creativity, and the unbreakable bond between people and place.

Thank you for supporting authentic Aboriginal and Australian art. Every purchase helps keep these stories and traditions alive!

A-Series (Horizontal)
What You’ll Receive: One high-resolution JPEG file (300 DPI). This striking artwork is provided in the versatile A-Series horizontal ratio, aligning with international standard paper sizes and offering a clean, contemporary aesthetic. Your high-resolution JPEG file boasts 14400 x 10182 pixels (with the long edge being 14400px), guaranteeing exceptional clarity and detail for large-format prints.

Maximum Printable Size (at 300 DPI): You can achieve a magnificent print up to 48 x 33.94 inches (121.9 x 86.2 cm) (based on A1 dimensions, scaled) without compromising quality.

Ideal Uses for the A-Series Horizontal Aspect Ratio: The A-Series horizontal format provides consistent proportions across different sizes, making it excellent for cohesive displays or when standard framing options are preferred. It's ideal for art that benefits from a clear, structured horizontal presentation, such as landscapes, cityscapes, or group compositions.

* 14.1x10" (36x25 cm): Great for smaller wall sections, desks, or as part of a multi-piece horizontal arrangement.
* 16.5x11.7" (A3) (42x29.7 cm): Excellent for creating a visually harmonious display, particularly when arranged with other standard-sized pieces.
* 23.4x16.5" (A2) (59.4x42 cm): A substantial size that brings a refined and structured horizontal element to living areas or studies.
* 33.1x23.4" (A1) (84.1x59.4 cm): Provides a broad and clean aesthetic, suitable for larger wall spaces in modern interiors.
* 48x33.94" (122x86.2 cm): A truly impressive, standard-sized piece that will beautifully fill a large wall space with clarity and impactful design.

Whether you want a cosy piece above your desk or a showstopper for your lounge, you’re sorted. And if you’re dreaming even bigger, just let me know—I’m always happy to help with custom sizes or display ideas.

Your image is available for download instantly and securely via Etsy—no PDFs, no third-party links, just simple and safe digital delivery.

Ready for Professional Printing: JPG file format for ease of use, or request other formats as needed.

Printing Tips:
For the best result, I always recommend using a quality professional print lab or one of the print-on-demand services listed below. This ensures vibrant colours and crisp detail—true to the spirit of the original artwork.
Avoid department store kiosks (like Big W or Harvey Norman), as they often can’t handle high-res art files and might leave you with a disappointing print.

Top 10 Print-On-Demand Services for Wall Art & Art Prints
1. Printful:
Renowned for gallery-quality posters, canvas, and framed prints. With global print hubs, you get fast, reliable shipping and excellent colour reproduction.

2. Printify:
A trusted network connecting you with print partners worldwide, offering affordable wall art options with lots of flexibility for sizing and materials.

3. Gelato:
Loved for local printing in 30+ countries. Their wall art—especially posters and framed prints—is top-notch and delivered fast. Great for supporting local wherever you are.

4. Gooten:
Specialists in home decor and wall art, Gooten offers consistent quality and worldwide shipping. Perfect for turning your digital art into beautiful, lasting pieces.

5. Prodigi:
Go-to for museum-quality fine art prints. Their giclée printing methods produce gallery-standard posters and canvases, capturing every detail and nuance.

6. Fine Art America (Pixels):
Huge range and artist-friendly platform. They handle everything from posters to metal and acrylic prints, with reliable fulfilment and global reach.

7. Displate:
For something different, Displate prints artwork directly onto premium metal panels—giving your art a bold, modern edge. Unique, collectible, and super durable.

8. Redbubble:
A favourite among indie artists. Redbubble offers posters, art prints, canvases, and a worldwide audience, so your art can brighten homes from Adelaide to Alaska.

9. Society6:
Trendy, stylish, and high-quality. Society6 caters to art lovers wanting something fresh, offering a wide range of print formats all made to a high spec.

10. InPrnt:
Respected for its professional artist community and gallery-quality art prints. If you want the best of the best, InPrnt delivers craftsmanship you can trust.

Important Notes:
This is a digital download only—no physical item will be shipped.
Colours may vary depending on your screen and printing method.
Personal use only—commercial use or redistribution is not permitted.
Need a different size or have a special request? Send me a message—I’m always happy to help!

❓ Frequently Asked Questions
Q: Is this a digital product?
A: Yes! You’ll get an instant download after purchase.

Q: Can I get a different size or format?
A: Absolutely—just send me a message before or after purchase.

Q: Is it for commercial use?
A: No, personal use only.

Q: Refunds?
A: Digital downloads are final sale, but I’ll fix any issues ASAP.

Q: Is it really limited edition?
A: 100%—only 25 total, ever.  🎨 LET’S CREATE SOMETHING BEAUTIFUL TOGETHER
At the end of the day, my art is a reflection of a lifetime of curiosity, hard work, and relentless passion. Whether you’re a seasoned collector or someone just dipping their toes into the world of digital art, I invite you to explore my work.
Each piece is more than an image - it’s a conversation, a piece of my soul, and a snapshot of a life lived with creativity and heart.

🙌 THANK YOU – FROM MY STUDIO TO YOUR HOME
Thank you for taking the time to explore my collection. I hope my art inspires you, makes you smile, and maybe even encourages you to see the world a little differently.
Here’s to art, to life, and to the beautiful stories we all share. 🎨✨
 
🚀 EXPLORE MY WORK: 
🖼 Browse my full collection: 👉 https://robincustance.etsy.com 📩 Have a question? Need help? Send me a message - I’d love to chat!

💫 WHY YOU’LL LOVE THIS ARTWORK
Ever wished you could bottle up that perfect moment - that golden sunset, that crisp evening breeze, that deep connection to the land? Well, consider this artwork your time machine.
✅ Instant Digital Download – No waiting, no shipping delays. Just click, download, print, and enjoy!
 
✅ Premium-Quality Digital Art – Every detail crisp, every colour vibrant, every brushstroke filled with warmth.
 
✅ Versatile Printing Options – Frame it, stretch it on canvas, print on acrylic - make it yours.
 
✅ Authentic Australian Outback Scene – Inspired by real landscapes, real moments, real connection.
 
✅ Perfect Gift for Nature Lovers & Adventure Seekers – Because nothing says "I get you" like an artwork that stirs the soul.
 
This piece isn’t just wall art - it’s a feeling. A reminder of the vastness, beauty, and magic of the Australian landscape, captured for you to cherish.
 
🛒 HOW TO BUY & PRINT
Ordering is as easy as throwing a snag on the barbie. Here’s how:
1️⃣ Add to Cart – Just a couple of clicks and this beauty is yours.
2️⃣ Checkout Securely – Easy, safe, and hassle-free.
3️⃣ Download Instantly – Etsy will send you a direct link to your high-resolution file.
4️⃣ Print It Your Way – At home, through an online service, or at a professional print shop.
5️⃣ Display & Enjoy – Frame it, gift it, or just sit back and admire your excellent taste in art.
Need help with printing? Just shoot me a message - I’m here to help!

❤️ Thank You & Stay Connected
Thank you for supporting Aboriginal and Australian art—your purchase helps keep cultural stories alive.

Don’t wait—click Add to Basket now to secure one of just 25 copies.

If my art brings a smile to your space, I’d love to see a photo or hear from you in a review.

Browse my full collection:
https://www.etsy.com/au/shop/RobinCustance

Special request or a question? Message me anytime—always happy to help.

From my family to yours, thank you for sharing the spirit and stories of Australia with the world.
Warm regards,
Robin Custance

Add this artwork to your shopping basket and let a little bit of Aussie warmth and Dreaming brighten your day, every day.
Thanks for keeping art and culture alive!

```

---
## 📄 scripts/analyze_artwork.py

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DreamArtMachine Lite | analyze_artwork.py
===============================================================
Professional, production-ready, fully sectioned and sub-sectioned
Robbie Mode™ script for analyzing artworks with OpenAI.

This revision adds comprehensive logging, robust error handling and
optional feedback injection for AI analysis. All activity is written to
``capitalart/logs/analyze-artwork-YYYY-MM-DD-HHMM.log``.

Output JSON logic, filename conventions and colour detection remain
compatible with the previous version.
"""

# ============================== [ Imports ] ===============================
import argparse
import datetime as _dt
import json
import logging
import os
import random
import re
import shutil
import sys
import traceback
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

Image.MAX_IMAGE_PIXELS = None

load_dotenv()
client = OpenAI()


# ======================= [ 1. CONFIGURATION & PATHS ] =======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTWORKS_DIR = PROJECT_ROOT / "inputs" / "artworks"
MOCKUPS_DIR = PROJECT_ROOT / "inputs" / "mockups"
GENERIC_TEXTS_DIR = PROJECT_ROOT / "generic_texts"
ONBOARDING_PATH = PROJECT_ROOT / "settings" / "Master-Etsy-Listing-Description-Writing-Onboarding.txt"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "artwork_listing_master.json"
OUTPUT_PROCESSED_ROOT = PROJECT_ROOT / "outputs" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
MOCKUPS_PER_LISTING = 9  # 1 thumb + 9 mockups

# --- [ 1.3: Etsy Colour Palette ]
ETSY_COLOURS = {
    'Beige': (222, 202, 173), 'Black': (24, 23, 22), 'Blue': (42, 80, 166), 'Bronze': (140, 120, 83),
    'Brown': (110, 72, 42), 'Clear': (240, 240, 240), 'Copper': (181, 101, 29), 'Gold': (236, 180, 63),
    'Grey': (160, 160, 160), 'Green': (67, 127, 66), 'Orange': (237, 129, 40), 'Pink': (229, 100, 156),
    'Purple': (113, 74, 151), 'Rainbow': (170, 92, 152), 'Red': (181, 32, 42), 'Rose gold': (212, 150, 146),
    'Silver': (170, 174, 179), 'White': (242, 242, 243), 'Yellow': (242, 207, 46)
}


# ====================== [ 2. LOGGING CONFIGURATION ] ========================
START_TS = _dt.datetime.now().strftime("%Y-%m-%d-%H%M")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"analyze-artwork-{START_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
logger = logging.getLogger("analyze_artwork")


class _Tee:
    """Simple tee to duplicate stdout/stderr to the log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file

    def write(self, data):
        self._original.write(data)
        self._original.flush()
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._original.flush()
        self._log.flush()


_log_fp = open(LOG_FILE, "a", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_fp)
sys.stderr = _Tee(sys.stderr, _log_fp)


logger.info("=== DreamArtMachine Lite: OpenAI Analyzer Started ===")


# ======================== [ 3. UTILITY FUNCTIONS ] ==========================

def get_aspect_ratio(image_path: Path) -> str:
    """Return closest aspect ratio label for given image."""
    with Image.open(image_path) as img:
        w, h = img.size
    aspect_map = [
        ("1x1", 1 / 1), ("2x3", 2 / 3), ("3x2", 3 / 2), ("3x4", 3 / 4), ("4x3", 4 / 3),
        ("4x5", 4 / 5), ("5x4", 5 / 4), ("5x7", 5 / 7), ("7x5", 7 / 5), ("9x16", 9 / 16),
        ("16x9", 16 / 9), ("A-Series-Horizontal", 1.414 / 1), ("A-Series-Vertical", 1 / 1.414),
    ]
    ar = round(w / h, 4)
    best = min(aspect_map, key=lambda tup: abs(ar - tup[1]))
    logger.info(f"Aspect ratio for {image_path.name}: {best[0]}")
    return best[0]


def pick_mockups(aspect: str, max_count: int = 8) -> list:
    """Select mockup images for the given aspect with graceful fallbacks."""
    primary_dir = MOCKUPS_DIR / f"{aspect}-categorised"
    fallback_dir = MOCKUPS_DIR / aspect

    if primary_dir.exists():
        use_dir = primary_dir
    elif fallback_dir.exists():
        logger.warning(f"Categorised mockup folder missing for {aspect}; using fallback")
        use_dir = fallback_dir
    else:
        logger.error(f"No mockup folder found for aspect {aspect}")
        return []

    candidates = [f for f in use_dir.glob("**/*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    random.shuffle(candidates)
    selection = [str(f.resolve()) for f in candidates[:max_count]]
    logger.info(f"Selected {len(selection)} mockups from {use_dir.name} for {aspect}")
    return selection


def read_generic_text(aspect: str) -> str:
    txt_path = GENERIC_TEXTS_DIR / f"{aspect}.txt"
    if txt_path.exists():
        logger.info(f"Loaded generic text for {aspect}")
        return txt_path.read_text(encoding="utf-8")
    logger.warning(f"No generic text found for {aspect}")
    return ""


def read_onboarding_prompt() -> str:
    return Path(ONBOARDING_PATH).read_text(encoding="utf-8")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\- ]+", "", text)
    text = text.strip().replace(" ", "-")
    return re.sub("-+", "-", text).lower()


def parse_text_fallback(text: str) -> dict:
    """Extract key fields from a non-JSON AI response."""
    data = {"fallback_text": text}

    tag_match = re.search(r"Tags:\s*(.*)", text, re.IGNORECASE)
    if tag_match:
        data["tags"] = [t.strip() for t in tag_match.group(1).split(",") if t.strip()]
    else:
        data["tags"] = []

    mat_match = re.search(r"Materials:\s*(.*)", text, re.IGNORECASE)
    if mat_match:
        data["materials"] = [m.strip() for m in mat_match.group(1).split(",") if m.strip()]
    else:
        data["materials"] = []

    title_match = re.search(r"(?:Title|Artwork Title|Listing Title)\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if title_match:
        data["title"] = title_match.group(1).strip()

    seo_match = re.search(r"(?:seo[_ ]filename|seo file|filename)\s*[:\-]\s*(.+\.jpe?g)", text, re.IGNORECASE)
    if seo_match:
        data["seo_filename"] = seo_match.group(1).strip()

    prim_match = re.search(r"Primary Colour\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if prim_match:
        data["primary_colour"] = prim_match.group(1).strip()
    sec_match = re.search(r"Secondary Colour\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if sec_match:
        data["secondary_colour"] = sec_match.group(1).strip()

    desc_match = re.search(r"(?:Description|Artwork Description)\s*[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if desc_match:
        data["description"] = desc_match.group(1).strip()

    return data


def extract_seo_filename_from_text(text: str, fallback_base: str) -> tuple[str, str]:
    """Attempt to extract an SEO filename from plain text."""
    patterns = [r"^\s*(?:SEO Filename|SEO_FILENAME|SEO FILE|FILENAME)\s*[:\-]\s*(.+)$"]
    for line in text.splitlines():
        for pat in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                base = re.sub(r"\.jpe?g$", "", m.group(1).strip(), flags=re.IGNORECASE)
                return slugify(base), "regex"

    m = re.search(r"([\w\-]+\.jpe?g)", text, re.IGNORECASE)
    if m:
        base = os.path.splitext(m.group(1))[0]
        return slugify(base), "jpg"

    return slugify(fallback_base), "fallback"


def extract_seo_filename(ai_listing: dict | None, raw_text: str, fallback_base: str) -> tuple[str, bool, str]:
    """Return SEO slug, whether fallback was used, and extraction method."""
    if ai_listing and isinstance(ai_listing, dict) and ai_listing.get("seo_filename"):
        name = os.path.splitext(str(ai_listing["seo_filename"]))[0]
        return slugify(name), False, "json"

    if ai_listing and isinstance(ai_listing, dict) and ai_listing.get("title"):
        slug = slugify(str(ai_listing["title"]))
        return slug, True, "title"

    slug, method = extract_seo_filename_from_text(raw_text, fallback_base)
    return slug, True, method


def make_preview_2000px_max(src_jpg: Path, dest_jpg: Path, target_long_edge: int = 2000, target_kb: int = 700, min_quality: int = 60) -> None:
    with Image.open(src_jpg) as im:
        w, h = im.size
        scale = target_long_edge / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            im = im.copy()

        q = 92
        for _ in range(12):
            im.save(dest_jpg, "JPEG", quality=q, optimize=True)
            kb = os.path.getsize(dest_jpg) / 1024
            if kb <= target_kb or q <= min_quality:
                break
            q -= 7
    logger.info(f"Saved preview {dest_jpg.name} ({kb:.1f} KB, Q={q})")


def save_finalised_artwork(original_path: Path, seo_name: str, output_base_dir: Path):
    target_folder = Path(output_base_dir) / seo_name
    target_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/using folder {target_folder}")

    orig_filename = original_path.name
    seo_main_jpg = target_folder / f"{seo_name}.jpg"
    orig_jpg = target_folder / f"original-{orig_filename}"
    thumb_jpg = target_folder / f"{seo_name}-THUMB.jpg"

    shutil.copy2(original_path, orig_jpg)
    shutil.copy2(original_path, seo_main_jpg)
    logger.info(f"Copied original files for {seo_name}")

    make_preview_2000px_max(seo_main_jpg, thumb_jpg, 2000, 700, 60)
    logger.info(f"Finalised files saved to {target_folder}")

    return str(seo_main_jpg), str(orig_jpg), str(thumb_jpg), str(target_folder)


def add_to_pending_mockups_queue(image_path: str, queue_file: str) -> None:
    try:
        if os.path.exists(queue_file):
            with open(queue_file, "r", encoding="utf-8") as f:
                queue = json.load(f)
            if not isinstance(queue, list):
                queue = []
        else:
            queue = []
    except Exception:
        queue = []
    if image_path not in queue:
        queue.append(image_path)
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)
    logger.info(f"Added to pending mockups queue: {image_path}")


# ===================== [ 4. COLOUR DETECTION & MAPPING ] ====================

def closest_colour(rgb_tuple):
    min_dist = float('inf')
    best_colour = None
    for name, rgb in ETSY_COLOURS.items():
        dist = sum((rgb[i] - rgb_tuple[i]) ** 2 for i in range(3)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_colour = name
    return best_colour


def get_dominant_colours(img_path: Path, n: int = 2):
    from sklearn.cluster import KMeans
    import numpy as np

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            crop = img.crop((max(0, w // 2 - 25), max(0, h // 2 - 25), min(w, w // 2 + 25), min(h, h // 2 + 25)))
            crop_dir = LOGS_DIR / "crops"
            crop_dir.mkdir(exist_ok=True)
            crop_path = crop_dir / f"{img_path.stem}-crop.jpg"
            crop.save(crop_path, "JPEG", quality=80)
            logger.debug(f"Saved crop for colour check: {crop_path}")
            img = img.resize((100, 100))
            arr = np.asarray(img).reshape(-1, 3)

        k = max(3, n + 1)
        kmeans = KMeans(n_clusters=k, n_init='auto' if hasattr(KMeans, 'n_init') else 10)
        labels = kmeans.fit_predict(arr)
        counts = np.bincount(labels)
        sorted_idx = counts.argsort()[::-1]
        seen = set()
        colours = []
        for i in sorted_idx:
            rgb = tuple(int(c) for c in kmeans.cluster_centers_[i])
            name = closest_colour(rgb)
            if name not in seen:
                seen.add(name)
                colours.append(name)
            if len(colours) >= n:
                break
        if len(colours) < 2:
            colours = (colours + ["White", "Black"])[:2]
    except Exception as e:
        logger.error(f"Colour detection failed for {img_path}: {e}")
        logger.error(traceback.format_exc())
        colours = ["White", "Black"]

    logger.info(f"Colours for {img_path.name}: {colours}")
    return colours


# ========================= [ 5. OPENAI HANDLER ] ===========================

def generate_ai_listing(system_prompt: str, image_filename: str, aspect: str, feedback: str | None = None) -> tuple[dict, bool, str]:
    user_message = (
        f"Artwork filename: {image_filename}\n"
        f"Aspect ratio: {aspect}\n"
        "Describe and analyze the artwork visually, then generate the listing as per the instructions above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    if feedback:
        messages.append({"role": "user", "content": feedback})

    logger.info(f"OpenAI API call for {image_filename} [{aspect}]")
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4.1"),
        messages=messages,
        max_tokens=2100,
        temperature=0.92,
    )
    content = response.choices[0].message.content.strip()
    logger.debug(f"Raw OpenAI response: {content}")
    try:
        parsed = json.loads(content)
        logger.info("Parsed JSON response successfully")
        return parsed, True, content
    except Exception:
        logger.warning("OpenAI response not valid JSON; applying fallback parser")
        logger.warning(f"Raw response sample: {content[:200]}")
        fallback = parse_text_fallback(content)
        logger.warning(f"Fallback extracted keys: {list(fallback.keys())}")
        return fallback, False, content


# ======================== [ 6. MAIN ANALYSIS LOGIC ] ========================

def analyze_single(image_path: Path, system_prompt: str, feedback_text: str | None, statuses: list):
    """Analyze and process a single image path."""

    status = {"file": str(image_path), "success": False, "error": ""}
    try:
        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))

        aspect = get_aspect_ratio(image_path)
        mockups = pick_mockups(aspect, MOCKUPS_PER_LISTING)
        generic_text = read_generic_text(aspect)
        fallback_base = image_path.stem

        try:
            ai_listing, was_json, raw_response = generate_ai_listing(system_prompt, image_path.name, aspect, feedback_text)
        except Exception as e:
            logger.error(f"OpenAI call failed for {image_path.name}: {e}")
            logger.error(traceback.format_exc())
            raise

        if not was_json:
            logger.warning(f"AI listing for {image_path.name} is not valid JSON")

        seo_name, used_fallback_naming, naming_method = extract_seo_filename(ai_listing if was_json else None, raw_response, fallback_base)
        if used_fallback_naming:
            logger.warning(f"SEO filename derived by {naming_method} for {image_path.name}: {seo_name}")
        else:
            logger.info(f"SEO filename from JSON for {image_path.name}: {seo_name}")

        main_jpg, orig_jpg, thumb_jpg, folder_path = save_finalised_artwork(image_path, seo_name, OUTPUT_PROCESSED_ROOT)
        
        primary_colour, secondary_colour = get_dominant_colours(Path(main_jpg), 2)

        tags = ai_listing.get("tags", []) if isinstance(ai_listing, dict) else []
        materials = ai_listing.get("materials", []) if isinstance(ai_listing, dict) else []
        if not tags:
            logger.warning(f"No tags extracted for {image_path.name}")
        if not materials:
            logger.warning(f"No materials extracted for {image_path.name}")

        per_artwork_json = Path(folder_path) / f"{seo_name}-listing.json"
        artwork_entry = {
            "filename": image_path.name,
            "aspect_ratio": aspect,
            "mockups": mockups,
            "generic_text": generic_text,
            "ai_listing": ai_listing,
            "seo_name": seo_name,
            "used_fallback_naming": used_fallback_naming,
            "main_jpg_path": main_jpg,
            "orig_jpg_path": orig_jpg,
            "thumb_jpg_path": thumb_jpg,
            "processed_folder": str(folder_path),
            "primary_colour": primary_colour,
            "secondary_colour": secondary_colour,
            "tags": tags,
            "materials": materials,
        }
        with open(per_artwork_json, "w", encoding="utf-8") as af:
            json.dump(artwork_entry, af, indent=2, ensure_ascii=False)
        logger.info(f"Wrote listing JSON to {per_artwork_json}")
        queue_file = OUTPUT_PROCESSED_ROOT / "pending_mockups.json"
        add_to_pending_mockups_queue(main_jpg, str(queue_file))

        status["success"] = True
        logger.info(f"Completed analysis for {image_path.name}")
        return artwork_entry

    except Exception as e:  # noqa: BLE001
        status["error"] = str(e)
        logger.error(f"Failed processing {image_path}: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        statuses.append(status)


# ============================ [ 7. MAIN ENTRY ] ============================

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze artwork(s) with OpenAI")
    parser.add_argument("image", nargs="?", help="Single image path to process")
    parser.add_argument("--feedback", help="Optional feedback text file")
    return parser.parse_args()


def main() -> None:
    print("\n===== DreamArtMachine Lite: OpenAI Analyzer =====\n")
    try:
        system_prompt = read_onboarding_prompt()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not read onboarding prompt: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: Could not read onboarding prompt. See log at {LOG_FILE}")
        sys.exit(1)

    args = parse_args()
    feedback_text = None
    if args.feedback:
        try:
            feedback_text = Path(args.feedback).read_text(encoding="utf-8")
            logger.info(f"Loaded feedback from {args.feedback}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to read feedback file {args.feedback}: {e}")
            logger.error(traceback.format_exc())

    single_path = Path(args.image) if args.image else None

    results = []
    statuses: list = []

    if single_path:
        logger.info(f"Single-image mode: {single_path}")
        entry = analyze_single(single_path, system_prompt, feedback_text, statuses)
        if entry:
            results.append(entry)
    else:
        all_images = [f for f in ARTWORKS_DIR.rglob("*.jpg") if f.is_file()]
        logger.info(f"Batch mode: {len(all_images)} images found")
        for idx, img_path in enumerate(sorted(all_images), 1):
            print(f"[{idx}/{len(all_images)}] {img_path.relative_to(ARTWORKS_DIR)}")
            entry = analyze_single(img_path, system_prompt, feedback_text, statuses)
            if entry:
                results.append(entry)

    if results:
        OUTPUT_JSON.parent.mkdir(exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote master JSON to {OUTPUT_JSON}")

    success_count = len([s for s in statuses if s["success"]])
    fail_count = len(statuses) - success_count
    logger.info(f"Analysis complete. Success: {success_count}, Failures: {fail_count}")

    print(f"\nListings processed! See logs at: {LOG_FILE}")
    if fail_count:
        print(f"{fail_count} file(s) failed. Check the log for details.")


if __name__ == "__main__":
    main()


```

---
## 📄 scripts/generate_composites.py

```py
import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
DEBUG_MODE = False

# ======================= [ 1. CONFIG & PATHS ] =======================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSSIBLE_ASPECT_RATIOS = [
    "1x1", "2x3", "3x2", "3x4", "4x3", "4x5", "5x4", "5x7", "7x5",
    "9x16", "16x9", "A-Series-Horizontal", "A-Series-Vertical"
]

ARTWORKS_ROOT = PROJECT_ROOT / "outputs" / "processed"
COORDS_ROOT = PROJECT_ROOT / "inputs" / "Coordinates"
MOCKUPS_ROOT = PROJECT_ROOT / "inputs" / "mockups"
QUEUE_FILE = ARTWORKS_ROOT / "pending_mockups.json"

# ======================= [ 2. UTILITIES ] =========================

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

def clean_base_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
    name = "-".join(filter(None, name.split('-')))
    return name

def remove_from_queue(processed_img_path, queue_file):
    """Remove the processed image from the pending queue."""
    if os.path.exists(queue_file):
        with open(queue_file, "r", encoding="utf-8") as f:
            queue = json.load(f)
        new_queue = [p for p in queue if p != processed_img_path]
        with open(queue_file, "w", encoding="utf-8") as f:
            json.dump(new_queue, f, indent=2)

# =============== [ 3. MAIN WORKFLOW: QUEUE-BASED PROCESSING ] ================

def main():
    print("\n===== CapitalArt Lite: Composite Generator (Queue Mode) =====\n")

    if not QUEUE_FILE.exists():
        print(f"⚠️ No pending mockups queue found at {QUEUE_FILE}")
        return

    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        queue = json.load(f)
    if not queue:
        print(f"✅ No pending artworks in queue. All done!")
        return

    print(f"🎨 {len(queue)} artworks in the pending queue.\n")

    processed_count = 0
    for img_path in queue[:]:  # Copy list, in case we mutate queue
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"❌ File not found (skipped): {img_path}")
            remove_from_queue(str(img_path), QUEUE_FILE)
            continue

        # Detect aspect ratio from folder or from JSON listing if available
        folder = img_path.parent
        seo_name = clean_base_name(img_path.stem)
        aspect = None
        json_listing = next(folder.glob(f"{seo_name}-listing.json"), None)
        if json_listing:
            with open(json_listing, "r", encoding="utf-8") as jf:
                entry = json.load(jf)
                aspect = entry.get("aspect_ratio")
        if not aspect:
            # Try to extract from grandparent dir (may be folder name if deeply nested)
            aspect = folder.parent.name
            print(f"  [WARNING] Could not read aspect from JSON, using folder: {aspect}")

        print(f"\n[{processed_count+1}/{len(queue)}] Processing: {img_path.name} [{aspect}]")

        # ----- Find categorised mockups for aspect -----
        mockups_cat_dir = MOCKUPS_ROOT / f"{aspect}-categorised"
        coords_dir = COORDS_ROOT / aspect
        if not mockups_cat_dir.exists() or not coords_dir.exists():
            print(f"⚠️ Missing mockups or coordinates for aspect: {aspect}")
            remove_from_queue(str(img_path), QUEUE_FILE)
            continue

        # ----- Load artwork image (resize for composites) -----
        art_img = Image.open(img_path).convert("RGBA")
        art_img_for_composite = resize_image_for_long_edge(art_img, target_long_edge=2000)
        mockup_seq = 1

        # ----- For each category: one random PNG mockup -----
        categories = sorted([
            d for d in os.listdir(mockups_cat_dir)
            if (mockups_cat_dir / d).is_dir()
        ])
        for category in categories:
            category_dir = mockups_cat_dir / category
            png_mockups = [f for f in os.listdir(category_dir) if f.lower().endswith(".png")]
            if not png_mockups:
                continue
            selected_mockup = random.choice(png_mockups)
            mockup_file = category_dir / selected_mockup
            coord_path = coords_dir / f"{os.path.splitext(selected_mockup)[0]}.json"
            if not coord_path.exists():
                print(f"⚠️ Missing coordinates for {selected_mockup} ({aspect}/{category})")
                continue

            with open(coord_path, "r", encoding="utf-8") as f:
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

            mockup_img = Image.open(mockup_file).convert("RGBA")
            composite = apply_perspective_transform(art_img_for_composite, mockup_img, dst_coords)

            # ---- [Naming and Saving] ----
            output_filename = f"{seo_name}-MU-{mockup_seq:02d}.jpg"
            output_path = folder / output_filename
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"   - Mockup {mockup_seq}: {output_filename} ({category})")
            mockup_seq += 1

        print(f"🎯 Finished all mockups for {img_path.name}.")
        processed_count += 1

        # ----- Remove this artwork from queue after processing -----
        remove_from_queue(str(img_path), QUEUE_FILE)

    print(f"\n✅ Done. {processed_count} artwork(s) processed and removed from queue.\n")

if __name__ == "__main__":
    main()

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
        <!-- DEBUG: Options for slot {{ loop.index0 }}: {{ options|join(", ") }} -->
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
  <button class="composite-btn" type="submit">✅ Generate Composites</button>
</form>
<div style="text-align:center;margin-top:1em;">
  {% if session.latest_seo_folder %}
    <a href="{{ url_for('composites_specific', seo_folder=session.latest_seo_folder) }}" class="composite-btn" style="background:#666;">👁️ Preview Composites</a>
  {% else %}
    <a href="{{ url_for('composites_preview') }}" class="composite-btn" style="background:#666;">👁️ Preview Composites</a>
  {% endif %}
</div>
{% endblock %}

```

---
## 📄 templates/review.html

```html
{% extends "main.html" %}
{% block title %}Review | CapitalArt{% endblock %}
{% block content %}
<h1>Review &amp; Approve Listing</h1>
<section class="review-artwork">
  <h2>{{ artwork.title }}</h2>
  <div class="artwork-images">
    <img src="{{ url_for('static', filename='outputs/processed/' ~ artwork.seo_name ~ '/' ~ artwork.main_image) }}"
         alt="Main artwork" class="main-art-img" style="max-width:360px;">
    <img src="{{ url_for('static', filename='outputs/processed/' ~ artwork.seo_name ~ '/' ~ artwork.thumb) }}"
         alt="Thumbnail" class="thumb-img" style="max-width:120px;">
  </div>
  <h3>Description</h3>
  <div class="art-description" style="max-width:431px;">
    <pre style="white-space: pre-wrap; font-family:inherit;">{{ artwork.description }}</pre>
  </div>
  <h3>Mockups</h3>
  <div class="grid">
    {% for slot in slots %}
    <div class="item">
      <img src="{{ url_for('mockup_img', category=slot.category, filename=slot.image) }}" alt="{{ slot.category }}">
      <strong>{{ slot.category }}</strong>
    </div>
    {% endfor %}
  </div>
</section>
<form method="post" action="{{ url_for('reset') }}">
  <button class="composite-btn" type="submit">Start Over</button>
</form>
<div style="text-align:center;margin-top:1.5em;">
  <a href="{{ url_for('composites_specific', seo_folder=artwork.seo_name) }}" class="composite-btn" style="background:#666;">Preview Composites</a>
</div>
{% endblock %}

```

---
## 📄 templates/artworks.html

```html
{% extends "main.html" %}
{% block title %}Artwork Gallery | CapitalArt{% endblock %}
{% block content %}
<h2 style="margin-top: 16px;">🖼️ Artwork Gallery</h2>
<div class="artwork-grid" style="display: flex; gap: 2rem; flex-wrap: wrap;">
  {% for art in artworks %}
  <div class="artwork-card" style="border:1px solid #eee;border-radius:8px;background:#fff;box-shadow:0 2px 8px #0001;padding:12px;width:340px;display:inline-block;">
    <img src="{{ url_for('artwork_image', aspect=art.aspect, filename=art.filename) }}"
         alt="{{ art.title }}"
         style="width:100%;height:auto;border-radius:8px;margin-bottom:8px;" />
    <div style="font-weight:bold;font-size:1.1em;line-height:1.3;margin-bottom:4px;">
      {{ art.title }}
    </div>
    <div style="font-size:0.95em;color:#888;">{{ art.aspect }}</div>
    <form method="post"
          action="{{ url_for('analyze_artwork', aspect=art.aspect, filename=art.filename) }}"
          style="margin-top:12px;">
      <button type="submit" class="analyze-btn" style="padding:8px 14px;font-size:1em;border-radius:5px;background:#1976d2;color:#fff;border:none;">
        🔍 Analyse
      </button>
    </form>
  </div>
  {% endfor %}
</div>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.analyze-btn').forEach(btn => {
      btn.closest('form').onsubmit = function() {
        let overlay = document.createElement('div');
        overlay.innerHTML = '<div style="position:fixed;top:0;left:0;right:0;bottom:0;z-index:9999;background:rgba(255,255,255,0.8);display:flex;align-items:center;justify-content:center;"><div style="font-size:2em;color:#333;">🔄 Analyzing... Please wait.</div></div>';
        document.body.appendChild(overlay);
      }
    });
  });
</script>
{% endblock %}

```

---
## 📄 templates/composites_preview.html

```html
{% extends "main.html" %}
{% block title %}Composite Preview | CapitalArt{% endblock %}
{% block content %}
<h1 style="text-align:center;">Composite Preview: {{ seo_folder }}</h1>
{% if listing %}
  <div style="text-align:center;margin-bottom:1.5em;">
    <img src="{{ url_for('processed_image', seo_folder=seo_folder, filename=seo_folder+'.jpg') }}" alt="artwork" style="max-width:260px;border-radius:8px;box-shadow:0 2px 6px #0002;">
  </div>
{% endif %}
{% if images %}
<div class="grid">
  {% for img in images %}
  <div class="item">
    <img src="{{ url_for('processed_image', seo_folder=seo_folder, filename=img.filename) }}" alt="{{ img.filename }}">
    <div style="font-size:0.9em;color:#555;word-break:break-all;">{{ img.filename }}</div>
    {% if img.category %}<div style="color:#888;font-size:0.9em;">{{ img.category }}</div>{% endif %}
    <form method="post" action="{{ url_for('regenerate_composite', seo_folder=seo_folder, slot_index=img.index) }}">
      <button type="submit" class="btn btn-reject">Regenerate</button>
    </form>
  </div>
  {% endfor %}
</div>
<form method="post" action="{{ url_for('approve_composites', seo_folder=seo_folder) }}" style="text-align:center;margin-top:2em;">
  <button type="submit" class="composite-btn">Finalize &amp; Approve</button>
</form>
{% else %}
<p style="text-align:center;margin:2em 0;">No composites found.</p>
{% endif %}
<div style="text-align:center;margin-top:2em;">
  <a href="{{ url_for('select') }}" class="composite-btn" style="background:#666;">Back to Selector</a>
</div>
{% endblock %}

```

---
## 📄 templates/gallery.html

```html

```

---
## 📄 templates/main.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{% block title %}CapitalArt{% endblock %}</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav>
      <a href="{{ url_for('home') }}">Home</a>
      <a href="{{ url_for('select') }}">Mockup Selector</a>
      <a href="{{ url_for('artworks') }}">Artwork Gallery</a>
      <a href="{{ url_for('review') }}">Review Listing</a>
      <!-- "Review Artwork" not shown directly in nav, but linked from gallery/analysis -->
    </nav>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="flash">{% for m in messages %}{{ m }}{% endfor %}</div>
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
  </main>
</body>
</html>

```

---
## 📄 templates/review_artwork.html

```html
{% extends "main.html" %}
{% block title %}Review Artwork | CapitalArt{% endblock %}
{% block content %}
<div style="max-width:660px;margin:2.5em auto;">
  <h1 style="font-size:2em;line-height:1.3;text-align:center;">{{ artwork.title }}</h1>
  <div style="text-align:center;margin:1.5em 0;">
    <a href="{{ url_for('processed_image', seo_folder=artwork.seo_name, filename=artwork.main_image.split('/')[-1]) }}" target="_blank">
      <img src="{{ url_for('processed_image', seo_folder=artwork.seo_name, filename=artwork.thumb.split('/')[-1]) }}"
        alt="{{ artwork.title }} thumbnail"
        style="max-width:320px;border-radius:10px;box-shadow:0 2px 8px #0002;cursor:pointer;">
    </a>
    <div style="font-size:0.98em;color:#888;margin-top:0.4em;">Click thumbnail for full size</div>
  </div>

  <div class="desc-panel" style="margin-bottom:2em;">
    <h2 style="margin-bottom:10px;">📝 AI-Generated Listing Description</h2>
    <pre style="white-space: pre-wrap;background:#fafbfc;border-radius:8px;box-shadow:0 1px 4px #0001;padding:16px;min-height:110px;max-height:350px;overflow-y:auto;font-size:1.05em;">{{ ai_description }}</pre>
    {% if tags %}
      <div style="margin-top:0.6em;"><strong>Tags:</strong> {{ tags|join(', ') }}</div>
    {% endif %}
    {% if materials %}
      <div><strong>Materials:</strong> {{ materials|join(', ') }}</div>
    {% endif %}
    {% if used_fallback_naming %}
      <div style="color: orange;">⚠️ SEO Filename used fallback extraction.</div>
    {% endif %}
  </div>

  <h3>About the Artist</h3>
  <div style="background:#fafbfc;border-radius:8px;box-shadow:0 1px 4px #0001;padding:16px;white-space:pre-line;font-size:1.05em;margin-bottom:2em;">{{ generic_text|safe }}</div>

  {% if raw_ai_output %}
  <details style="margin-bottom:2em;">
    <summary>Debug: Raw AI Output</summary>
    <pre style="white-space: pre-wrap;background:#f0f0f0;padding:10px;">{{ raw_ai_output }}</pre>
  </details>
  {% endif %}

  <div style="display:flex;gap:22px;justify-content:center;margin-bottom:2.2em;">
    <!-- COLOUR FIELDS BELOW -->
    <div style="flex:1 1 0;min-width:120px;max-width:160px;">
      <div style="font-weight:600;font-size:1.02em;margin-bottom:5px;">Primary Colour</div>
      <div style="background:#fcfcfc;border-radius:7px;border:1px solid #e2e4e8;min-height:38px;padding:7px 12px;font-size:0.97em;white-space:pre-line;overflow-x:auto;">
        {{ artwork.primary_colour if artwork.primary_colour else "&mdash;" }}
      </div>
    </div>
    <div style="flex:1 1 0;min-width:120px;max-width:160px;">
      <div style="font-weight:600;font-size:1.02em;margin-bottom:5px;">Secondary Colour</div>
      <div style="background:#fcfcfc;border-radius:7px;border:1px solid #e2e4e8;min-height:38px;padding:7px 12px;font-size:0.97em;white-space:pre-line;overflow-x:auto;">
        {{ artwork.secondary_colour if artwork.secondary_colour else "&mdash;" }}
      </div>
    </div>
  </div>

  <form method="POST" action="{{ url_for('analyze_artwork', aspect=artwork.aspect, filename=artwork.seo_name + '.jpg') }}" style="max-width:420px;margin:0 auto;text-align:center;">
    <div style="margin-bottom:10px;font-size:1.07em;font-weight:500;">Re-analyse Artwork</div>
    <textarea name="feedback" placeholder="Add feedback or changes for next analysis..." rows="3" style="width:100%;max-width:400px;padding:10px;border-radius:7px;border:1px solid #ccd2da;resize:vertical;font-size:1em;margin-bottom:9px;"></textarea>
    <br>
    <button type="submit" class="btn btn-primary" style="font-size:1.04em;padding:7px 28px;border-radius:7px;">Re-analyse</button>
  </form>
</div>
{% endblock %}

```

---
## 📄 assets/style.css

```css
body {
  font-family: system-ui, sans-serif;
  margin: 0;
  background: #f9f9f9;
  color: #222;
}
header, footer {
  background: #333;
  color: #fff;
  padding: 1em;
  text-align: center;
}
main {
  display: flex;
  flex-wrap: wrap;
  padding: 2em;
  gap: 1em;
  justify-content: center;
}
.artwork {
  border: 1px solid #ccc;
  background: #fff;
  padding: 1em;
  max-width: 300px;
}
.artwork img {
  max-width: 100%;
  height: auto;
  display: block;
}

```

---
## 📄 older-scripts/compatible-image-generator.jsx

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
## 📄 older-scripts/backup_mockup_structure.sh

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
## 📄 older-scripts/export-artwork-layers.jsx

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
## 📄 older-scripts/fix_srgb_profiles.py

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
## 📄 older-scripts/backup_mockup_generator.sh

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
## 📄 older-scripts/rename-4x5-mockup-layers.jsx

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
## 📄 older-scripts/find-matching-png-files.py

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
## 📄 older-scripts/openai_vision_test.py

```py
import os
from dotenv import load_dotenv
from openai import OpenAI

if os.getenv("SKIP_OPENAI_TESTS", "1") == "1":
    import pytest
    pytest.skip("Skipping OpenAI vision test during automated runs", allow_module_level=True)

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
## 📄 older-scripts/generate_all_coordinates.py

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
## 📄 older-scripts/generate_composites.py

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
## 📄 older-scripts/utils.py

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
## 📄 older-scripts/dupe-png-files-and-rename.py

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
## 📄 older-scripts/google_vision_basic_analyzer.py

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
## 📄 older-scripts/generate_folder_structure.py

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
## 📄 older-scripts/organise-jpg-and-png-files.py

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
## 📄 older-scripts/ai-jpeg-reset.sh

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
## 📄 older-scripts/write_all_composite_generators.py

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
## 📄 older-scripts/gather_mockup_code_to_text.sh

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

---
## 📄 reports/REPORTS-25-JUN-2025-04-29PM/pip_check.txt

```txt
No broken requirements found.

```

---
## 📄 reports/REPORTS-25-JUN-2025-04-29PM/pip_outdated.txt

```txt
Package       Version Latest Type
------------- ------- ------ -----
pydantic_core 2.33.2  2.35.1 wheel

```

---
## 📄 reports/REPORTS-29-JUN-2025-01-25PM/git_snapshot.txt

```txt
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    capitalart-total-nuclear.py
	modified:   folder_structure.txt

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	capitalart-total-nuclear-v2.py
	package-lock.json
	reports/REPORTS-29-JUN-2025-01-16PM/
	reports/REPORTS-29-JUN-2025-01-25PM/

no changes added to commit (use "git add" and/or "git commit -a")
commit 6edaf66411065e4d62ab41d051609230ca4a329d
Author: Robin Custance <your@email.com>
Date:   Sat Jun 28 18:40:55 2025 +0930

    feat: Add composite preview, per-slot regeneration, approval/finalisation, and improved workflow UI
    
    - Introduced per-artwork composite preview route (/composites/<seo_folder>) showing all generated composites with category and slot info.
    - Added per-slot composite regeneration: allows selective re-generation of any mockup composite in an artwork’s set, preserving the rest.
    - Implemented composite approval/finalisation workflow:
        - Copies all approved composites, main image, and JSON to /outputs/finalised-artwork/<seo_folder>/
        - Generates zip archive for export or Sellbrite integration.
        - Marks approval in listing JSON.
    - Updated UI navigation:
        - Selector, review, and workflow pages now link directly to the latest artwork’s composite preview.
        - Users can easily preview, regenerate, and finalise composites from anywhere in the workflow.
    - Added robust logging for composite generation, regeneration, approval, and error events (logs/composites-workflow.log).
    - Improved error handling and user feedback for missing files, invalid slots, and script exceptions.
    - All actions tested: pytest -q ✅
    
    This enables a seamless, pro-grade review and approval workflow for mockup composites and sets the stage for CSV/API automation.
    
    Ref #capitalart
 capitalart-total-nuclear.py | 118 --------------------------------------------
 folder_structure.txt        |  80 +++++++++++++++++++++++++-----
 2 files changed, 68 insertions(+), 130 deletions(-)
🔧 Git Status:

🔁 Last Commit:

🧾 Diff Summary:

```

---
## 📄 reports/REPORTS-29-JUN-2025-01-25PM/env_metadata.txt

```txt
Python 3.13.4
Darwin Robins-iMac.local 24.5.0 Darwin Kernel Version 24.5.0: Tue Apr 22 19:53:26 PDT 2025; root:xnu-11417.121.6~2/RELEASE_X86_64 x86_64
annotated-types==0.7.0
anyio==4.9.0
blinker==1.9.0
certifi==2025.6.15
charset-normalizer==3.4.2
click==8.2.1
distro==1.9.0
Flask==3.1.1
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.10.0
joblib==1.5.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
numpy==2.3.1
openai==1.92.2
opencv-python==4.11.0.86
pandas==2.3.0
pillow==11.2.1
pydantic==2.11.7
pydantic_core==2.33.2
Pygments==2.19.2
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
pytz==2025.2
requests==2.32.4
rich==14.0.0
scikit-learn==1.7.0
scipy==1.16.0
six==1.17.0
sniffio==1.3.1
threadpoolctl==3.6.0
tqdm==4.67.1
typing-inspection==0.4.1
typing_extensions==4.14.0
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
🐍 Python Version:

🖥️ Platform Info:

📦 Installed Packages:

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

/* ----- Home styling ----- */
.home-hero {
  text-align: center;
  margin: 2em auto 1em auto;
}
.home-actions {
  margin: 2em auto 3em auto;
}

/* ----- Artworks Gallery styling ----- */
.analyze-btn {
    margin-top: 10px;
    background: #007bff;
    color: #fff;
    border: none;
    padding: 6px 16px;
    border-radius: 4px;
    font-weight: bold;
    cursor: pointer;
    transition: background 0.2s;
}
.analyze-btn:hover { background: #0056b3; }


/* ----- End CapitalArt Approval UI Stylesheet — Robbie Mode™ ----- */

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
