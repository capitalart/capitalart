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
