# ======================== [ DreamArtMachine Lite | analyze_artwork.py ] ========================
# Professional, production-ready, fully sectioned and sub-sectioned “Robbie Mode™” script.
# - Analyzes art using OpenAI, onboarding prompt, and generic text per aspect.
# - Moves/copies files to /Users/robin/Desktop/Listing/{SEO_NAME}/ for easy uploading.
# - Outputs: original JPG, SEO-named JPG, 2000px preview (<700KB), ready for mockups.
# - Full paths and all metadata are saved to artwork_listing_master.json.
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
# --- [ 1.1: Project Root and Inputs ]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTWORKS_DIR = PROJECT_ROOT / "inputs" / "artworks"
MOCKUPS_DIR = PROJECT_ROOT / "inputs" / "mockups"
GENERIC_TEXTS_DIR = PROJECT_ROOT / "generic_texts"
ONBOARDING_PATH = PROJECT_ROOT / "settings" / "Master-Etsy-Listing-Description-Writing-Onboarding.txt"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "artwork_listing_master.json"
OUTPUT_PROCESSED_ROOT = PROJECT_ROOT / "outputs" / "processed"
MOCKUPS_PER_LISTING = 9  # 1 thumb + 9 mockups = 10 Etsy images total

# --- [ 1.2: Load OpenAI Config from .env ]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_PRIMARY_MODEL", "gpt-4.1")
FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4-turbo")
client = OpenAI(api_key=OPENAI_API_KEY)

# ======================= [ 2. UTILITY FUNCTIONS ] ==========================

# --- [ 2.1: Aspect Ratio Detection ]
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

# --- [ 2.2: Pick Random Mockups (Optional, Not Used for Now) ]
def pick_mockups(aspect, max_count=8):
    aspect_dir = MOCKUPS_DIR / aspect
    if not aspect_dir.exists():
        return []
    candidates = [f for f in aspect_dir.glob("**/*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    random.shuffle(candidates)
    return [str(f.resolve()) for f in candidates[:max_count]]

# --- [ 2.3: Read Generic Text for Aspect ]
def read_generic_text(aspect):
    txt_path = GENERIC_TEXTS_DIR / f"{aspect}.txt"
    return txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

# --- [ 2.4: Read Onboarding Prompt ]
def read_onboarding_prompt():
    return Path(ONBOARDING_PATH).read_text(encoding="utf-8")

# --- [ 2.5: Slugify for SEO File Naming ]
def slugify(text):
    text = re.sub(r"[^\w\- ]+", '', text)
    text = text.strip().replace(' ', '-')
    return re.sub('-+', '-', text).lower()

# --- [ 2.6: Extract SEO Filename from AI Output ]
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

# --- [ 2.7: Make 2000px Preview, <700KB ]
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

# --- [ 2.8: Save Artwork Set (Original, SEO, Preview) ]
def save_finalised_artwork(original_path, seo_name, output_base_dir):
    target_folder = Path(output_base_dir) / seo_name
    target_folder.mkdir(parents=True, exist_ok=True)

    orig_filename = Path(original_path).name
    seo_main_jpg = target_folder / f"{seo_name}.jpg"
    orig_jpg = target_folder / f"original-{orig_filename}"
    thumb_jpg = target_folder / f"{seo_name}-THUMB.jpg"

    shutil.copy2(original_path, orig_jpg)
    shutil.copy2(original_path, seo_main_jpg)
    print(f"   - Original and SEO-named file saved to {target_folder}")

    make_preview_2000px_max(seo_main_jpg, thumb_jpg, 2000, 700, 60)

    return str(seo_main_jpg), str(orig_jpg), str(thumb_jpg), str(target_folder)

# --- [ 2.9: File Type Filter ]
def is_image(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))

# --- [ 2.10: AI Listing Generation ]
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
        model=OPENAI_MODEL,
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

# ======================= [ 3. MAIN EXECUTION ] ==============================
def add_to_pending_mockups_queue(image_path, queue_file):
    """Appends the artwork image path to the pending queue JSON file."""
    import json
    if os.path.exists(queue_file):
        with open(queue_file, "r", encoding="utf-8") as f:
            queue = json.load(f)
    else:
        queue = []
    if image_path not in queue:
        queue.append(image_path)
        with open(queue_file, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)

def main():
    print("\n===== DreamArtMachine Lite: OpenAI Bulk Analyzer =====\n")
    # --- [ 3.1: Load Onboarding Prompt ]
    try:
        system_prompt = read_onboarding_prompt()
    except Exception as e:
        print(f"❌ Error: Could not read onboarding prompt: {e}")
        sys.exit(1)

    # --- [ 3.2: Recursively Find Artworks ]
    artworks = [f for f in ARTWORKS_DIR.rglob("*.jpg") if f.is_file()]
    if not artworks:
        print(f"❌ No artworks found in {ARTWORKS_DIR} or any subfolders.")
        sys.exit(1)
    print(f"Found {len(artworks)} artworks to analyze.\n")

    output_data = []
    for idx, art_path in enumerate(artworks, 1):
        print(f"[{idx}/{len(artworks)}] Processing: {art_path.relative_to(ARTWORKS_DIR)}")
        aspect = get_aspect_ratio(art_path)
        mockups = pick_mockups(aspect, MOCKUPS_PER_LISTING)
        generic_text = read_generic_text(aspect)
        fallback_base = os.path.splitext(art_path.name)[0]

        # --- [ 3.3: AI Analysis ]
        try:
            ai_listing = generate_ai_listing(system_prompt, art_path.name, aspect)
        except Exception as e:
            print(f"  [OpenAI ERROR] Skipping {art_path.name}: {e}")
            continue

        # --- [ 3.4: Extract SEO Name ]
        seo_name = extract_seo_filename(ai_listing, fallback_base)
        if not seo_name:
            print(f"  [SEO Name ERROR] Could not extract SEO name. Using fallback.")
            seo_name = fallback_base

        # --- [ 3.5: Save Artwork Set ]
        main_jpg_path, orig_jpg_path, thumb_jpg_path, folder_path = save_finalised_artwork(
            str(art_path), seo_name, OUTPUT_PROCESSED_ROOT
        )

        # --- [ 3.6: Per-Artwork JSON Output ]
        per_artwork_json = os.path.join(folder_path, f"{seo_name}-listing.json")
        artwork_entry = {
            "filename": art_path.name,
            "aspect_ratio": aspect,
            "mockups": mockups,
            "generic_text": generic_text,
            "ai_listing": ai_listing,
            "seo_name": seo_name,
            "main_jpg_path": main_jpg_path,
            "orig_jpg_path": orig_jpg_path,
            "thumb_jpg_path": thumb_jpg_path,
            "processed_folder": str(folder_path)
        }
        with open(per_artwork_json, "w", encoding="utf-8") as af:
            json.dump(artwork_entry, af, indent=2, ensure_ascii=False)

        # --- [ 3.7: Update Pending Mockups Queue ]
        queue_file = OUTPUT_PROCESSED_ROOT / "pending_mockups.json"
        add_to_pending_mockups_queue(main_jpg_path, str(queue_file))

        output_data.append(artwork_entry)

    # --- [ 3.8: Output Master JSON ]
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=2, ensure_ascii=False)
    print(f"\n✅ All listings processed! Output saved to: {OUTPUT_JSON}")

# ======================= [ 4. ENTRY POINT ] ==================================
if __name__ == "__main__":
    main()

# ======================== [ END OF SCRIPT ] ==================================
