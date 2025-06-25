#!/bin/bash

#####################################################################################
#  ai-jpeg-reset.sh — Purify AI JPEGs for Fujifilm Imagine/Harvey Norman Print Labs #
# --------------------------------------------------------------------------------- #
#  Author: Robin “Robbie” Custance + ChatGPT Assist                                #
#  Last updated: 18-May-2025                                                        #
#                                                                                   #
#  DESCRIPTION:                                                                     #
#    - Converts every .jpg/.jpeg in AI-JPEG-Originals into an 8-bit TIFF, then      #
#      exports a fully standard, lab-safe JPEG into AI-JPEG-Reset.                  #
#    - Strips all metadata, forces sRGB, 8-bit, baseline, 4:2:0, 300 DPI.           #
#    - Removes all traces of Midjourney/AI segment oddities by using TIFF as a      #
#      “reset” intermediary.                                                        #
#    - Should resolve “invalid JPEG” errors on Fujifilm Imagine, Harvey Norman,     #
#      and BigW photo kiosks.                                                       #
#                                                                                   #
#  HOW TO USE:                                                                      #
#    1. Place original images in .../AI-JPEG-Originals                              #
#    2. Run this script (see below)                                                 #
#    3. Retrieve “purified” JPEGs from .../AI-JPEG-Reset                            #
#                                                                                   #
#  REQUIREMENTS: ImageMagick installed (`brew install imagemagick`)                 #
#                                                                                   #
#  USAGE:                                                                           #
#    chmod +x ai-jpeg-reset.sh                                                      #
#    ./ai-jpeg-reset.sh                                                             #
#####################################################################################

in_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/AI-JPEG-Originals"
tmp_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/AI-JPEG-TIFFS"
out_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/AI-JPEG-Reset"

mkdir -p "$tmp_dir" "$out_dir"

icc_profile="/Library/ColorSync/Profiles/sRGB2014.icc" # Use IEC61966-2.1.icc if sRGB2014 causes issues

for f in "$in_dir"/*.jpg "$in_dir"/*.jpeg; do
  [ -e "$f" ] || continue
  base=$(basename "$f" | sed 's/ /_/g')
  name="${base%.*}"

  tiff_file="$tmp_dir/${name}.tiff"
  out_file="$out_dir/${name}-RESET.jpg"

  echo "Converting $base to TIFF..."
  magick "$f" -strip -depth 8 -colorspace sRGB "$tiff_file"

  echo "Exporting $name as fresh JPEG..."
  magick "$tiff_file" \
    -strip \
    -depth 8 \
    -colorspace sRGB \
    -profile "$icc_profile" \
    -units PixelsPerInch -density 300 \
    -sampling-factor 2x2,1x1,1x1 \
    -interlace none \
    -quality 85 \
    "$out_file"

  # Optional: Remove intermediate TIFF to save space
  rm "$tiff_file"

  # Validate the output
  final_size=$(stat -f%z "$out_file")
  echo "✅ $out_file created: $((final_size/1024/1024)) MB"

done

echo "Batch conversion complete. All 'RESET' JPEGs are ready in $out_dir."
