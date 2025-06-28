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