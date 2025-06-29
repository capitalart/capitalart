#!/bin/bash
# ======================================================
# 🧠 Local Backup Script: Mockup Generator (Mac Dev)
# 📍 Saves project archive excluding venv/macOS metadata
# 🕒 Timestamp set using Australia/Adelaide timezone
# ▶️ Run with:
#     bash backup_mockup_generator.sh
# ======================================================

set -e

echo "🔄 Starting Mockup Generator backup..."
BACKUP_BASE_DIR="/Users/robin/mockup-generator-backups"
PROJECT_SOURCE_DIR="/Users/robin/Documents/01-ezygallery-MockupWorkShop"
LOG_DIR="/Users/robin/mockup-generator-backups/logs"

mkdir -p "$BACKUP_BASE_DIR"
mkdir -p "$LOG_DIR"

# Set time/date stamp (Adelaide time)
TIMESTAMP=$(TZ="Australia/Adelaide" date +"%Y-%m-%d_%H%M%S_%Z") # More sortable timestamp
READABLE_TIMESTAMP=$(TZ="Australia/Adelaide" date +"%a-%d-%b-%Y_%I-%M%p")
BACKUP_PATH="$BACKUP_BASE_DIR/$BACKUP_FILENAME"
LOG_FILE="$LOG_DIR/backup_log.txt"

echo "----------------------------------------------------" | tee -a "$LOG_FILE"
echo "Backup started at $(TZ="Australia/Adelaide" date)" | tee -a "$LOG_FILE"

cd /Users/robin/Documents || exit 1

tar -czf "$BACKUP_PATH" \
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

echo "✅ Backup created at $BACKUP_PATH" | tee -a "$LOG_FILE"
echo "----------------------------------------------------"