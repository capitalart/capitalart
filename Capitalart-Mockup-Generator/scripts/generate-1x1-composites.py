import os
import json
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

Image.MAX_IMAGE_PIXELS = None
aspect_ratio = "1x1"
DEBUG_MODE = False

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

input_artworks_dir = os.path.join(project_root, f"Input/Artworks/{aspect_ratio}")
input_mockups_dir = os.path.join(project_root, f"Input/Mockups/{aspect_ratio}")
input_coords_dir = os.path.join(project_root, f"Input/Coordinates/{aspect_ratio}")
output_root_dir = os.path.join(project_root, f"Output/Composites/{aspect_ratio}")
os.makedirs(output_root_dir, exist_ok=True)

def resize_image(image: Image.Image, target_size=2000) -> Image.Image:
    return image.resize((target_size, target_size), Image.LANCZOS)

def draw_debug_overlay(img: Image.Image, points: list) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red', outline='white')
    return img

def apply_perspective_transform(art_img, mockup_img, dst_coords):
    h, w = art_img.size
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(dst_coords)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    art_np = np.array(art_img)
    # Original logic: warp artwork and then composite it on top of the mockup using a mask from the artwork
    warped = cv2.warpPerspective(art_np, matrix, (mockup_img.width, mockup_img.height))
    mask = np.any(warped > 0, axis=-1).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    composite = Image.composite(Image.fromarray(warped), mockup_img, mask) # Artwork is placed in front
    if DEBUG_MODE:
        composite = draw_debug_overlay(composite, dst_coords)
    return composite

artwork_files = [f for f in os.listdir(input_artworks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not artwork_files:
    print("⚠️ No artwork files found.")
else:
    # Removed rjc_sku_counter as the SKU is assumed to be part of the artwork's base filename
    for index, artwork_file in enumerate(artwork_files):
        # Clean the artwork filename to be used in the composite name
        # This assumes the artwork's base name (e.g., 'flinders-ranges-eucalyptus-RJC-0003')
        # already contains the SKU part.
        cleaned_base_name = os.path.splitext(artwork_file)[0].lower()
        cleaned_base_name = cleaned_base_name.replace(" ", "-").replace("_", "-").replace("(", "").replace(")", "")
        cleaned_base_name = "-".join(filter(None, cleaned_base_name.split('-')))

        set_name = f"set-{index+1:02d}"
        set_dir = os.path.join(output_root_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        art_path = os.path.join(input_artworks_dir, artwork_file)
        art_img = Image.open(art_path).convert("RGBA")
        art_img = resize_image(art_img)

        # Copy original artwork and any associated text file to the output set directory
        shutil.copy2(art_path, os.path.join(set_dir, artwork_file))
        txt_path = os.path.join(input_artworks_dir, os.path.splitext(artwork_file)[0] + ".txt")
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(set_dir, os.path.splitext(artwork_file)[0] + ".txt"))

        for mockup_file in os.listdir(input_mockups_dir):
            if not mockup_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            mockup_path = os.path.join(input_mockups_dir, mockup_file)
            coord_path = os.path.join(input_coords_dir, os.path.splitext(mockup_file)[0] + ".json")

            if not os.path.exists(coord_path):
                print(f"⚠️ Missing coordinates for {mockup_file}")
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
            composite = apply_perspective_transform(art_img, mockup_img, dst_coords)

            # Extract mockup number from mockup filename (e.g., "1x1-Mockup-1.jpg" -> "01")
            mockup_base_name = os.path.splitext(mockup_file)[0]
            mockup_number_str = ""
            if 'mockup-' in mockup_base_name.lower():
                parts = mockup_base_name.lower().split('mockup-')
                if len(parts) > 1:
                    mockup_number_str = parts[-1].strip()
            
            try:
                mockup_num_formatted = f"{int(mockup_number_str):02d}"
            except (ValueError, IndexError):
                print(f"⚠️ Could not extract valid mockup number from '{mockup_file}'. Using fallback.")
                mockup_num_formatted = "XX"

            # Construct the new output filename with mockup number, assuming SKU is in cleaned_base_name
            output_filename = f"{cleaned_base_name}-MU-{mockup_num_formatted}.jpg"
            output_path = os.path.join(set_dir, output_filename)
            composite.convert("RGB").save(output_path, "JPEG", quality=85)
            print(f"✅ Saved: {output_filename}")

    print(f"\n🎯 All {aspect_ratio} composite sets completed.")