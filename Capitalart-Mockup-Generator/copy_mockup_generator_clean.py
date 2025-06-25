import shutil
from pathlib import Path

# Paths: set your actual mockup workshop path here
SOURCE_BASE = Path("/Users/robin/Documents/01-ezygallery-MockupWorkShop")
DEST_BASE = Path("/Users/robin/Documents/mockup-generator-cleaned")

# Required folders and scripts
REQUIRED_SUBDIRS = [
    "Input/Artworks", "Input/Mockups", "Input/Coordinates",
    "Output/Composites", "scripts"
]

REQUIRED_SCRIPTS = [
    "generate_composites.py",
    "generate_folder_structure.py",
    "generate_all_coordinates.py",
    "utils.py",
    *[f"generate-{r}-composites.py" for r in ["1x1", "2x3", "3x2", "3x4", "4x3", "4x5", "5x4", "9x16", "16x9"]],
]

def copy_mockup_generator():
    # Create destination base
    DEST_BASE.mkdir(parents=True, exist_ok=True)

    # Copy scripts
    src_scripts = SOURCE_BASE / "scripts"
    dst_scripts = DEST_BASE / "scripts"
    dst_scripts.mkdir(parents=True, exist_ok=True)

    for script_name in REQUIRED_SCRIPTS:
        src = src_scripts / script_name
        dst = dst_scripts / script_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✅ Copied: {script_name}")
        else:
            print(f"⚠️ Missing script: {script_name}")

    # Copy required folders
    for subdir in REQUIRED_SUBDIRS:
        src_dir = SOURCE_BASE / subdir
        dst_dir = DEST_BASE / subdir
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f"📁 Copied folder: {subdir}")
        else:
            print(f"⚠️ Missing folder: {subdir}")

    print(f"\n🎉 All essential files copied to: {DEST_BASE}")

if __name__ == "__main__":
    copy_mockup_generator()
