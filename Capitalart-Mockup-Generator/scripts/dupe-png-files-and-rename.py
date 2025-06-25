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
