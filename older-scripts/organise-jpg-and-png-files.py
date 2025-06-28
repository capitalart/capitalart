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
