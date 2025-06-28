#!/bin/bash
# =========================================================================
# 🧠 Script Name: gather_mockup_code_to_text.sh
# 📍 Location: /Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts
# ▶️ Run with:
#     bash scripts/gather_mockup_code_to_text.sh
# =========================================================================

# --- Configuration ---
SNAPSHOT_BASENAME="mockup_code_snapshot"
OUTPUT_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop/backups"
ADELAIDE_TIMEZONE="Australia/Adelaide"

FILENAME_TIMESTAMP=$(TZ="$ADELAIDE_TIMEZONE" date +"%a-%d-%b-%Y_%I.%M%p_%Z")
OUTPUT_FILE="${OUTPUT_DIR}/${SNAPSHOT_BASENAME}_${FILENAME_TIMESTAMP}.md"
GENERATED_TIMESTAMP=$(TZ="$ADELAIDE_TIMEZONE" date +"%A, %d %B %Y, %I:%M:%S %p %Z (%z)")

INCLUDE_EXTENSIONS=("*.py" "*.sh" "*.txt" "*.md" "*.json")
EXCLUDE_PATHS=(
  "*/venv/*"
  "*/__pycache__/*"
  "*/.git/*"
  "*/.vscode/*"
  "*/.DS_Store"
  "*/backups/*"
  "*/__MACOSX/*"
  "*/Artworks/*"
  "*/Upscaled-Art/*"
  "*/Optimised-Art/*"
  "*/Signed-Art/*"
)

PROJECT_DIRECTORIES=(
  "/Users/robin/Documents/01-ezygallery-MockupWorkShop/scripts"
  "/Users/robin/Documents/01-ezygallery-MockupWorkShop/Coordinates"
  "/Users/robin/Documents/01-ezygallery-MockupWorkShop/Output"
)

echo "📦 Gathering mockup generator source files into Markdown snapshot..."
mkdir -p "$OUTPUT_DIR"
echo -e "# Mockup Generator Code Snapshot\n\n**Generated:** ${GENERATED_TIMESTAMP}\n\n---" > "$OUTPUT_FILE"

# Process and append all files
for base_path in "${PROJECT_DIRECTORIES[@]}"; do
  if [ -d "$base_path" ]; then
    for ext in "${INCLUDE_EXTENSIONS[@]}"; do
      while IFS= read -r file; do
        # Check against excluded paths
        skip=false
        for exclude in "${EXCLUDE_PATHS[@]}"; do
          if [[ "$file" == $exclude ]]; then
            skip=true; break
          fi
        done
        if [ "$skip" = false ]; then
          relative_path="${file#/Users/robin/Documents/01-ezygallery-MockupWorkShop/}"
          {
            echo ""
            echo "## 📄 FILE: \`$relative_path\`"
            echo '```'
            cat "$file"
            echo '```'
            echo "---"
          } >> "$OUTPUT_FILE"
        fi
      done < <(find "$base_path" -type f -name "$ext" 2>/dev/null)
    done
  fi
done

echo "✅ Markdown snapshot saved to: $OUTPUT_FILE"
