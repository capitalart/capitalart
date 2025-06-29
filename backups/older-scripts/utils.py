#!/usr/bin/env python3
# =============================================================================
# 🧠 Module: utils.py
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts
# ▶️ Imported by:
#     generate_composites.py
# 🎯 Purpose:
#     1. Load JSON coordinate data for mockups.
#     2. Apply a perspective warp to map artwork into mockup templates.
# 🔗 Dependencies: OpenCV, NumPy, JSON, pathlib
# 🕒 Last Updated: May 9, 2025
# =============================================================================

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_corner_data(templates_dir: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Loads all JSON files from the given coordinates folder.

    Args:
        templates_dir (str): Path to the folder containing JSON corner files.

    Returns:
        dict: A mapping from lowercase template filename (e.g., 'mockup-01.png')
              to its 4-corner data for warping.
    """
    corner_data = {}
    templates_path = Path(templates_dir)

    for json_file in templates_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                template_name = data.get("template", "").lower().strip()
                corners = data.get("corners")

                if template_name and corners and len(corners) == 4:
                    corner_data[template_name] = corners
        except Exception as e:
            print(f"❌ Failed to read {json_file.name}: {e}")

    return corner_data


def perspective_transform(
    artwork_img: np.ndarray,
    dst_points: List[Dict[str, int]],
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Applies a 4-point perspective transformation to map an artwork into a mockup zone.

    Args:
        artwork_img (np.ndarray): The original artwork image.
        dst_points (list): List of 4 destination corner dicts (with "x" and "y").
        output_shape (tuple): Shape (height, width) of the mockup image.

    Returns:
        np.ndarray: The warped artwork image sized to fit inside the mockup.
    """
    h, w = artwork_img.shape[:2]

    # Source points are the corners of the artwork image
    src_points = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype="float32"
    )

    # Destination points are the mockup's 4 corners
    dst_points_array = np.array(
        [[pt["x"], pt["y"]] for pt in dst_points], dtype="float32"
    )

    # Create the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points_array)

    # Warp the artwork using the matrix and match mockup's resolution
    warped = cv2.warpPerspective(
        artwork_img,
        matrix,
        (output_shape[1], output_shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return warped
