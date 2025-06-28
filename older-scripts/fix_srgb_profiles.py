import os
from PIL import Image

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MOCKUPS_DIR = os.path.join(BASE_DIR, 'Input', 'Mockups')

def strip_broken_icc_profiles():
    print("🧼 Stripping broken sRGB profiles from PNGs...")
    for filename in os.listdir(MOCKUPS_DIR):
        if filename.lower().endswith('.png'):
            path = os.path.join(MOCKUPS_DIR, filename)
            try:
                img = Image.open(path)
                img = img.convert("RGBA")  # ensure consistent mode
                img.save(path, format="PNG", icc_profile=None)  # strip any broken profiles
                print(f"✅ Cleaned: {filename}")
            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")
    print("🏁 All done, no more sRGB warnings 🎉")

if __name__ == "__main__":
    strip_broken_icc_profiles()
