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

