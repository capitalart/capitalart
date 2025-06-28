#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================== [ DreamArtMachine Lite | analyze_artwork.py ] ========================
# Professional, production-ready, fully sectioned and sub-sectioned “Robbie Mode™” script.
# - Analyzes art using OpenAI, onboarding prompt, and generic text per aspect.
# - Moves/copies files to /outputs/processed/{SEO_NAME}/ for easy uploading.
# - Outputs: original JPG, SEO-named JPG, 2000px preview (<700KB), ready for mockups.
# - Detects primary and secondary Etsy colours per artwork.
# - Full paths and all metadata are saved to artwork_listing_master.json.
# - Supports single-image CLI (from Flask) or batch mode (manual).
# ==============================================================================================

import os
import sys
import json
import random
import shutil
import re
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ======================= [ 1. CONFIGURATION & PATHS ] =======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTWORKS_DIR = PROJECT_ROOT / "inputs" / "artworks"
MOCKUPS_DIR = PROJECT_ROOT / "inputs" / "mockups"
GENERIC_TEXTS_DIR = PROJECT_ROOT / "generic_texts"
ONBOARDING_PATH = PROJECT_ROOT / "settings" / "Master-Etsy-Listing-Description-Writing-Onboarding.txt"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "artwork_listing_master.json"
OUTPUT_PROCESSED_ROOT = PROJECT_ROOT / "outputs" / "processed"
MOCKUPS_PER_LISTING = 9  # 1 thumb + 9 mockups = 10 Etsy images total

# --- [ 1.3: Etsy Colour Palette ]
ETSY_COLOURS = {
    'Beige': (222, 202, 173), 'Black': (24, 23, 22), 'Blue': (42, 80, 166), 'Bronze': (140, 120, 83),
    'Brown': (110, 72, 42), 'Clear': (240, 240, 240), 'Copper': (181, 101, 29), 'Gold': (236, 180, 63),
    'Grey': (160, 160, 160), 'Green': (67, 127, 66), 'Orange': (237, 129, 40), 'Pink': (229, 100, 156),
    'Purple': (113, 74, 151), 'Rainbow': (170, 92, 152), 'Red': (181, 32, 42), 'Rose gold': (212, 150, 146),
    'Silver': (170, 174, 179), 'White': (242, 242, 243), 'Yellow': (242, 207, 46)
}

# ======================= [ 2. UTILITY FUNCTIONS ] ==========================

def get_aspect_ratio(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
    aspect_map = [
        ("1x1", 1/1), ("2x3", 2/3), ("3x2", 3/2), ("3x4", 3/4), ("4x3", 4/3),
        ("4x5", 4/5), ("5x4", 5/4), ("5x7", 5/7), ("7x5", 7/5), ("9x16", 9/16),
        ("16x9", 16/9), ("A-Series-Horizontal", 1.414/1), ("A-Series-Vertical", 1/1.414)
    ]
    ar = round(w / h, 4)
    best = min(aspect_map, key=lambda tup: abs(ar - tup[1]))
    return best[0]

def pick_mockups(aspect, max_count=8):
    aspect_dir = MOCKUPS_DIR / aspect
    if not aspect_dir.exists():
        return []
    candidates = [f for f in aspect_dir.glob("**/*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    random.shuffle(candidates)
    return [str(f.resolve()) for f in candidates[:max_count]]

def read_generic_text(aspect):
    txt_path = GENERIC_TEXTS_DIR / f"{aspect}.txt"
    return txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

def read_onboarding_prompt():
    return Path(ONBOARDING_PATH).read_text(encoding="utf-8")

def slugify(text):
    text = re.sub(r"[^\w\- ]+", '', text)
    text = text.strip().replace(' ', '-')
    return re.sub('-+', '-', text).lower()

def extract_seo_filename(ai_listing, fallback_base):
    if isinstance(ai_listing, dict):
        if "seo_filename" in ai_listing:
            name = ai_listing["seo_filename"]
            name = os.path.splitext(name)[0]
            return slugify(name)
        if "title" in ai_listing:
            return slugify(ai_listing["title"])
    elif isinstance(ai_listing, str):
        match = re.search(r"(SEO_FILENAME|SEO FILE|FILENAME)\s*[:\-]\s*(.+)", ai_listing, re.IGNORECASE)
        if match:
            base = os.path.splitext(match.group(2).strip())[0]
            return slugify(base)
        title_match = re.search(r"(Title|Listing Title)\s*[:\-]\s*(.+)", ai_listing, re.IGNORECASE)
        if title_match:
            return slugify(title_match.group(2).strip())
    return slugify(fallback_base)

def make_preview_2000px_max(src_jpg, dest_jpg, target_long_edge=2000, target_kb=700, min_quality=60):
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
        print(f"      → Saved 2000px preview as {os.path.basename(dest_jpg)} ({kb:.1f} KB, Q={q})")

def save_finalised_artwork(original_path, seo_name, output_base_dir):
    target_folder = Path(output_base_dir) / seo_name
    target_folder.mkdir(parents=True, exist_ok=True)
    orig_filename = Path(original_path).name
    seo_main_jpg = target_folder / f"{seo_name}.jpg"
    orig_jpg = target_folder / f"original-{orig_filename}"
    thumb_jpg = target_folder / f"{seo_name}-THUMB.jpg"
    shutil.copy2(original_path, orig_jpg)
    shutil.copy2(original_path, seo_main_jpg)
    make_preview_2000px_max(seo_main_jpg, thumb_jpg, 2000, 700, 60)
    return str(seo_main_jpg), str(orig_jpg), str(thumb_jpg), str(target_folder)

def is_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

def generate_ai_listing(system_prompt, image_filename, aspect):
    user_message = (
        f"Artwork filename: {image_filename}\n"
        f"Aspect ratio: {aspect}\n"
        "Describe and analyze the artwork visually, then generate the listing as per the instructions above."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    print(f"  → OpenAI: {image_filename} [{aspect}] ...")
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4.1"),
        messages=messages,
        max_tokens=2100,
        temperature=0.92,
    )
    content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
        return parsed
    except Exception:
        return content

def add_to_pending_mockups_queue(image_path, queue_file):
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

# === [ 2.1: Colour Detection & Mapping ] ===

def closest_colour(rgb_tuple):
    min_dist = float('inf')
    best_colour = None
    for name, rgb in ETSY_COLOURS.items():
        dist = sum((rgb[i] - rgb_tuple[i]) ** 2 for i in range(3)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_colour = name
    return best_colour

def get_dominant_colours(img_path, n=2):
    from sklearn.cluster import KMeans
    import numpy as np
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((100, 100))
        arr = np.asarray(img).reshape(-1, 3)
    k = max(3, n+1)
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
    return colours

# ==================== [ 3. MAIN ANALYSIS LOGIC ] ==========================

def analyze_single(image_path, system_prompt):
    """Analyze and process a single image path (Flask or batch mode). Always runs (no skip)."""
    if not Path(image_path).is_file():
        print(f"❌ Not found: {image_path}")
        return None
    # Always process, even if already processed (for forced re-analyze)
    aspect = get_aspect_ratio(image_path)
    mockups = pick_mockups(aspect, MOCKUPS_PER_LISTING)
    generic_text = read_generic_text(aspect)
    fallback_base = os.path.splitext(Path(image_path).name)[0]
    try:
        ai_listing = generate_ai_listing(system_prompt, Path(image_path).name, aspect)
    except Exception as e:
        print(f"  [OpenAI ERROR] Skipping {Path(image_path).name}: {e}")
        return None
    seo_name = extract_seo_filename(ai_listing, fallback_base)
    if not seo_name:
        print(f"  [SEO Name ERROR] Could not extract SEO name. Using fallback.")
        seo_name = fallback_base
    main_jpg_path, orig_jpg_path, thumb_jpg_path, folder_path = save_finalised_artwork(
        str(image_path), seo_name, OUTPUT_PROCESSED_ROOT
    )

    # [COLOUR DETECTION STEP]
    primary_colour, secondary_colour = get_dominant_colours(main_jpg_path, 2)

    per_artwork_json = os.path.join(folder_path, f"{seo_name}-listing.json")
    artwork_entry = {
        "filename": Path(image_path).name,
        "aspect_ratio": aspect,
        "mockups": mockups,
        "generic_text": generic_text,
        "ai_listing": ai_listing,
        "seo_name": seo_name,
        "main_jpg_path": main_jpg_path,
        "orig_jpg_path": orig_jpg_path,
        "thumb_jpg_path": thumb_jpg_path,
        "processed_folder": str(folder_path),
        "primary_colour": primary_colour,
        "secondary_colour": secondary_colour
    }
    with open(per_artwork_json, "w", encoding="utf-8") as af:
        json.dump(artwork_entry, af, indent=2, ensure_ascii=False)
    queue_file = OUTPUT_PROCESSED_ROOT / "pending_mockups.json"
    add_to_pending_mockups_queue(main_jpg_path, str(queue_file))
    return artwork_entry

# ======================== [ 4. MAIN ENTRY POINT ] ==========================

def main():
    print("\n===== DreamArtMachine Lite: OpenAI Analyzer =====\n")
    try:
        system_prompt = read_onboarding_prompt()
    except Exception as e:
        print(f"❌ Error: Could not read onboarding prompt: {e}")
        sys.exit(1)
    single_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    if single_path:
        print(f"→ Single-image mode: {single_path}")
        entry = analyze_single(single_path, system_prompt)
        if entry:
            results.append(entry)
    else:
        all_images = [f for f in ARTWORKS_DIR.rglob("*.jpg") if f.is_file()]
        for idx, img_path in enumerate(sorted(all_images), 1):
            print(f"[{idx}/{len(all_images)}] Processing: {img_path.relative_to(ARTWORKS_DIR)}")
            entry = analyze_single(str(img_path), system_prompt)
            if entry:
                results.append(entry)
    if results:
        OUTPUT_JSON.parent.mkdir(exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=2, ensure_ascii=False)
        print(f"\n✅ Listings processed! Output saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
