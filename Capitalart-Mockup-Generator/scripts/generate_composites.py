import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

# =========================== [CONFIGURABLES] ===========================
possible_aspect_ratios = [
    "1x1",
    "2x3",
    "3x2",
    "3x4",
    "4x3",
    "4x5",
    "5x4",
    "5x7",
    "7x5",
    "9x16",
    "16x9",
    "A-Series-Horizontal",
    "A-Series-Vertical"
]
# =======================================================================

overall_artworks_found = False

for aspect_ratio in possible_aspect_ratios:
    # --- [1. Directory Setup] ---
    input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
    input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
    input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
    output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
    os.makedirs(output_root_dir, exist_ok=True)

    # --- [2. Helper Functions] ---
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

    # --- [3. Find Artworks] ---
    if not os.path.exists(input_artworks_dir):
        print(f"⚠️ Artwork folder not found for {aspect_ratio}: {input_artworks_dir}")
        continue

    artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not artwork_files:
        print(f"⚠️ No artwork files found in {aspect_ratio} folder: {input_artworks_dir}")
        continue
    else:
        overall_artworks_found = True

    print(f"\n✨ Processing {len(artwork_files)} artworks for {aspect_ratio} aspect ratio...")

    # === [4. Process Each Artwork] ===
    for index, artwork_file in enumerate(artwork_files):
        # --- [4.1 Clean Base Name] ---
        cleaned_base_name = os.path.splitext(artwork_file)[0]
        cleaned_base_name = cleaned_base_name.replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
        cleaned_base_name = "-".join(filter(None, cleaned_base_name.split('-')))

        # --- [4.2 Output Set Directory (Artwork-Based Naming)] ---
        set_dir = os.path.join(output_root_dir, f"{cleaned_base_name}-Mockups")
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")

        # --- [4.3 Generate Preview Image & Save to Output Folder] ---
        preview_img = resize_image_for_long_edge(art_img.copy(), target_long_edge=2000)
        preview_filename = f"{cleaned_base_name}-THUMB-01.jpg"
        preview_path = os.path.join(set_dir, preview_filename)  # <-- NOW in the set_dir (output, not input)
        target_file_size_kb = 700
        initial_quality = 85
        min_quality = 50
        current_quality = initial_quality
        file_size_bytes = float('inf')

        print(f"Attempting to save preview for {artwork_file} under {target_file_size_kb}KB...")

        while file_size_bytes > (target_file_size_kb * 1024) and current_quality >= min_quality:
            preview_img.convert("RGB").save(preview_path, "JPEG", quality=current_quality, optimize=True)
            file_size_bytes = os.path.getsize(preview_path)
            file_size_kb = file_size_bytes / 1024

            if file_size_kb > target_file_size_kb:
                print(f"  Current size: {file_size_kb:.2f}KB (Quality: {current_quality}). Reducing quality...")
                current_quality -= 5
            else:
                print(f"  Achieved target! Final size: {file_size_kb:.2f}KB (Quality: {current_quality}).")
                break

            if current_quality < min_quality:
                print(f"  Warning: Minimum quality ({min_quality}) reached. Final size: {file_size_kb:.2f}KB. Target may not be met.")

        print(f"✅ Saved Preview: {preview_filename} (Long edge 2000px, placed in {set_dir})")

        # --- [4.4 Copy Original and TXT if exists] ---
        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, os.path.splitext(artwork_file)[0] + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, os.path.splitext(artwork_file)[0] + ".txt"))

        # --- [4.5 Prepare Mockup Processing] ---
        if not os.path.exists(input_mockups_dir):
            print(f"⚠️ Mockup folder not found for {aspect_ratio}: {input_mockups_dir}")
            continue

        art_img_for_composite = resize_image_for_long_edge(art_img, target_long_edge=2000)

        # -------- [4.6 MOCKUP NUMBERING RESET FOR THIS ARTWORK] --------
        mockup_seq = 1

        # --- [4.7 Process Each Mockup for this Artwork] ---
        for mockup_file in sorted(os.listdir(input_mockups_dir)):
            if "-THUMB-01.jpg" in mockup_file:
                continue
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")
            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file} in {aspect_ratio} folder.")
                continue

            with open(coord_path, "r") as f:
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

            mockup_img = Image.open(mockup_path).convert("RGBA")
            composite = apply_perspective_transform(art_img_for_composite, mockup_img, dst_coords)

            # --- [4.8 Sequential Naming of Mockups] ---
            output_filename = f"{cleaned_base_name}-MU-{mockup_seq:02d}.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved Composite: {output_filename}")
            mockup_seq += 1

    print(f"🎯 Finished processing for {aspect_ratio} aspect ratio.")

# === [5. Summary Message] ===
if not overall_artworks_found:
    print("\n⚠️ No artwork files found in any of the specified aspect ratio folders.")
else:
    print("\n🎉 All composite sets and previews completed for all found aspect ratios.")
