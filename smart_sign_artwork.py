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