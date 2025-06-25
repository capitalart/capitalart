#!/usr/bin/env python3
# =========================================================
# 🧠 Script: generate_folder_structure.py
# 📍 Local Mockup Generator Tool
# 📅 Timestamped with Australia/Adelaide timezone
# ▶️ Run with:
#     python3 scripts/generate_folder_structure.py
# =========================================================

import os
from datetime import datetime
from pathlib import Path
import pytz

BASE_DIR = Path("/Users/robin/Documents/01-ezygallery-MockupWorkShop")
IGNORE_FOLDERS = {"venv", "__pycache__", ".git", ".idea", ".vscode", "node_modules"}
IGNORE_FILES = {".DS_Store", "Thumbs.db"}
IGNORE_EXTS = {".pyc", ".log", ".tmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".psd"}

timestamp = datetime.now(pytz.timezone("Australia/Adelaide")).strftime("%a-%d-%b-%Y_%I-%M%p")
output_file = BASE_DIR / f"folder-structure-no-images_{timestamp}.txt"

def generate_tree(path, prefix=""):
    lines = []
    entries = sorted(os.listdir(path))
    for idx, name in enumerate(entries):
        full_path = path / name
        connector = "└── " if idx == len(entries) - 1 else "├── "
        if full_path.is_dir() and name not in IGNORE_FOLDERS:
            lines.append(f"{prefix}{connector}{name}/")
            sub_prefix = "    " if idx == len(entries) - 1 else "│   "
            lines.extend(generate_tree(full_path, prefix + sub_prefix))
        elif full_path.is_file():
            if name in IGNORE_FILES or full_path.suffix.lower() in IGNORE_EXTS:
                continue
            lines.append(f"{prefix}{connector}{name}")
    return lines

if __name__ == "__main__":
    print("📂 Generating folder structure (excluding image files)...")
    lines = [f"{BASE_DIR.name}/"] + generate_tree(BASE_DIR)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Folder structure saved to: {output_file}")