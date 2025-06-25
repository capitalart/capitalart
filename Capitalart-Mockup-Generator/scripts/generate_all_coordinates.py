#!/usr/bin/env python3
# =============================================================================
# 🧠 Script: generate_all_coordinates.py
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts/
# 🎯 Purpose:
#     Scans all PNG mockup images inside Input/Mockups/[aspect-ratio] folders.
#     Detects transparent artwork zones and outputs a JSON file with 4 corner
#     coordinates into Input/Coordinates/[aspect-ratio] folders.
# ▶️ Run with:
#     python3 scripts/generate_all_coordinates.py
# =============================================================================

import os
import cv2
import json

# ----------------------------------------
# 📁 Folder Paths
# ----------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MOCKUP_DIR = os.path.join(BASE_DIR, 'Input', 'Mockups')
COORDINATE_DIR = os.path.join(BASE_DIR, 'Input', 'Coordinates')

# ----------------------------------------
# 🔧 Ensure output folders exist
# ----------------------------------------
def ensure_folder(path):
    """Ensure a folder exists; create if missing."""
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
# 📐 Corner Sorting
# ----------------------------------------
def sort_corners(pts):
    """
    Sorts 4 corner points to a consistent order:
    top-left, top-right, bottom-left, bottom-right
    """
    pts = sorted(pts, key=lambda p: (p["y"], p["x"]))  # Primary sort by Y, secondary by X
    top = sorted(pts[:2], key=lambda p: p["x"])
    bottom = sorted(pts[2:], key=lambda p: p["x"])
    return [*top, *bottom]

# ----------------------------------------
# 🔍 Transparent Region Detector
# ----------------------------------------
def detect_corner_points(image):
    """
    Detects 4 corner points of a transparent region in a PNG mockup.
    Returns a list of 4 dict points or None if detection fails.
    """
    if image is None or image.shape[2] != 4:
        return None

    # Use alpha channel to find transparent regions
    alpha = image[:, :, 3]
    _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    thresh_inv = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get largest contour and approximate shape
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:
        return None

    corners = [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in approx]
    return sort_corners(corners)

# ----------------------------------------
# 🚀 Coordinate Generation Runner
# ----------------------------------------
def generate_all_coordinates():
    """
    Loops through all subfolders in Input/Mockups/,
    detects artwork areas in .png files, and outputs JSON
    coordinate templates to Input/Coordinates/[aspect-ratio]/
    """
    print(f"\n📁 Scanning mockup source: {MOCKUP_DIR}\n")

    if not os.path.exists(MOCKUP_DIR):
        print(f"❌ Error: Mockup directory not found: {MOCKUP_DIR}")
        return

    for folder in sorted(os.listdir(MOCKUP_DIR)):
        mockup_folder = os.path.join(MOCKUP_DIR, folder)
        if not os.path.isdir(mockup_folder):
            continue

        print(f"🔍 Processing folder: {folder}")
        output_folder = os.path.join(COORDINATE_DIR, folder)
        ensure_folder(output_folder)

        for filename in os.listdir(mockup_folder):
            if not filename.lower().endswith('.png'):
                continue

            mockup_path = os.path.join(mockup_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.png', '.json'))

            try:
                image = cv2.imread(mockup_path, cv2.IMREAD_UNCHANGED)
                corners = detect_corner_points(image)

                if corners:
                    data = {
                        "template": filename,
                        "corners": corners
                    }
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"✅ Saved: {output_path}")
                else:
                    print(f"⚠️ Skipped (no valid corners): {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

    print("\n🏁 All coordinate templates generated.\n")

# ----------------------------------------
# 🔧 Script Entrypoint
# ----------------------------------------
if __name__ == "__main__":
    generate_all_coordinates()
