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
