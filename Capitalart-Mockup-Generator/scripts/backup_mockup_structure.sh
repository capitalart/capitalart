#!/bin/bash
# ======================================================
# 🧠 Local Backup Script: Mockup Generator (Structure Only)
# 📍 Saves project structure excluding large image files
# 🕒 Timestamp set using Australia/Adelaide timezone
# ▶️ Run with:
#     bash scripts/backup_mockup_structure.sh
# ======================================================

set -e

echo "🔄 Starting Mockup Generator STRUCTURE-ONLY backup..."
BACKUP_BASE_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop/backups"
PROJECT_SOURCE_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop"
LOG_DIR="$BACKUP_BASE_DIR/logs"

mkdir -p "$BACKUP_BASE_DIR"
mkdir -p "$LOG_DIR"

# Set time/date stamp (Adelaide time)
TIMESTAMP=$(TZ="Australia/Adelaide" date +"%a-%d-%b-%Y_%I-%M%p") # e.g. Wed-08-May-2025_06-24PM

BACKUP_FILENAME="mockup_structure_backup_${TIMESTAMP}.tar.gz"
BACKUP_PATH="$BACKUP_BASE_DIR/$BACKUP_FILENAME"
LOG_FILE="$LOG_DIR/backup_log.txt"

echo "----------------------------------------------------" | tee -a "$LOG_FILE"
echo "Backup started at $TIMESTAMP" | tee -a "$LOG_FILE"

cd /Users/robin/Documents || exit 1

tar -czf "$BACKUP_PATH" \
    --exclude=backups \
    --exclude=*.jpg \
    --exclude=*.jpeg \
    --exclude=*.png \
    --exclude=*.webp \
    --exclude=*.tif \
    --exclude=*.tiff \
    --exclude=*.psd \
    --exclude=venv \
    --exclude=__pycache__ \
    --exclude=.DS_Store \
    --exclude=.Spotlight-V100 \
    --exclude=.TemporaryItems \
    --exclude=.Trashes \
    --exclude=.DocumentRevisions-V100 \
    --exclude=.fseventsd \
    --exclude=.VolumeIcon.icns \
    --exclude=.AppleDouble \
    --exclude=.apdisk \
    "01-ezygallery-MockupWorkShop"

echo "✅ STRUCTURE backup created at $BACKUP_PATH" | tee -a "$LOG_FILE"
echo "----------------------------------------------------"