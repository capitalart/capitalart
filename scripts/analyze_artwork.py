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
    attempt, max_attempts = 1, 3
    error_text = ""
    while attempt <= max_attempts:
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4.1"),
                messages=messages,
                max_tokens=2100,
                temperature=0.92,
                timeout=60,  # Timeout after 60 seconds
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
        except Exception as e:
            error_text = str(e)
            logger.error(f"OpenAI API error on attempt {attempt}: {e}")
            logger.error(traceback.format_exc())
            attempt += 1
            if attempt <= max_attempts:
                logger.info(f"Retrying OpenAI call (attempt {attempt})...")
                import time; time.sleep(2)
    # All attempts failed
    logger.error(f"OpenAI API failed after {max_attempts} attempts: {error_text}")
    return {"fallback_text": "OpenAI API failed, no listing generated.", "error": error_text}, False, ""

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

