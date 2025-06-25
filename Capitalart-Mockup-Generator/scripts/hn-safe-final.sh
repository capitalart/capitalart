#!/bin/bash

##############################################################################################
#  hn-safe-final.sh — Adaptive, Validating, and Logging Harvey Norman JPEG Batch Script      #
# ------------------------------------------------------------------------------------------ #
#  Author: Robin “Robbie” Custance + AI/print lab deep-dive                                  #
#  Last updated: 18-May-2025 (jpegtran fix, dependency info, improved logging)               #
#                                                                                            #
#  DESCRIPTION:                                                                              #
#    Processes JPEGs for Harvey Norman print services, aiming for maximum compatibility.     #
#    Key features: metadata stripping, sRGB conversion, JFIF enforcement, configurable      #
#    chroma subsampling & DPI, adaptive quality/resizing, detailed logging, optional         #
#    aggressive final metadata strip.                                                        #
#                                                                                            #
#  CRITICAL FOR USER:                                                                        #
#    1. SET 'target_sampling_factor' and 'target_density' below based on a                   #
#       JPEG file KNOWN TO BE ACCEPTED by Harvey Norman.                                     #
#       Use: exiftool -YCbCrSubSampling your_accepted_image.jpg                              #
#            exiftool -XResolution -YResolution your_accepted_image.jpg                      #
#    2. VERIFY 'icc_profile' path.                                                           #
#    3. CONSIDER enabling the 'FINAL OPTIONAL EXIFTOOL STRIP/REBUILD' section for stubborn   #
#       files AFTER providing diagnostic exiftool logs for comparison.                       #
#                                                                                            #
#  DEPENDENCIES: imagemagick, jpegtran, exiftool, bc                                         #
#    macOS (Homebrew): brew install imagemagick libjpeg exiftool bc                          #
#    Debian/Ubuntu: sudo apt update && sudo apt install imagemagick libjpeg-turbo-progs \    #
#                   libimage-exiftool-perl bc                                                #
#    Fedora/CentOS: sudo dnf install ImageMagick libjpeg-turbo-utils \                       #
#                   perl-Image-ExifTool bc                                                   #
#                                                                                            #
##############################################################################################

# === Script Configuration ===
in_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/Originals"
out_dir="/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/HN-Safe-Final"
icc_profile="/Library/ColorSync/Profiles/sRGB2014.icc" # Standard Mac sRGB v2 profile.
# If issues persist with sRGB2014.icc, consider a generic sRGB IEC61966-2.1 profile:
# icc_profile="/path/to/your/sRGB_IEC61966-2.1.icc"

log_file_timestamp=$(date +%Y%m%d-%H%M%S)
log_file="hn-safe-final-LOG-${log_file_timestamp}.txt"

# --- USER: SET THESE BASED ON A KNOWN GOOD, ACCEPTED IMAGE ---
# YCbCrSubSampling examples:
#   '1x1,1x1,1x1' for 4:4:4 (no subsampling, highest quality, largest file component)
#   '2x1,1x1,1x1' for 4:2:2 (horizontal subsampling)
#   '2x2,1x1,1x1' for 4:2:0 (horizontal and vertical subsampling, common for compatibility & size)
target_sampling_factor='2x2,1x1,1x1' # Defaulting to 4:2:0 as a strong candidate
target_density=300                   # Common print DPI (e.g., 300, 254, or 72 if uploader recalculates)

# === File Size Limits ===
max_size=$((20800 * 1024))   # 20.8 MB (target maximum)
plus5MB=$((max_size + 5*1024*1024))
plus10MB=$((max_size + 10*1024*1024))
# Thresholds for starting more aggressive resizing, relative to max_size
med_size_threshold=$((max_size + 2*1024*1024)) # e.g., 22.8MB, start 5% resizes
low_size_threshold=$((max_size + 4*1024*1024)) # e.g., 24.8MB, start 10% resizes

# === Initial Setup ===
mkdir -p "$out_dir"
declare -a temp_files_to_clean_for_current_file

# Function to clean up temporary files for the current image being processed
cleanup_current_file_temps() {
    for tmp_file in "${temp_files_to_clean_for_current_file[@]}"; do
        if [ -f "$tmp_file" ]; then
             rm "$tmp_file" && echo "    Removed temp file: $tmp_file" >> "$log_file"
        fi
    done
    temp_files_to_clean_for_current_file=() # Clear the array
}
# Trap EXIT, INT, TERM signals to run a broader cleanup if script is interrupted
main_cleanup_on_exit() {
    echo "Script interrupted. Attempting to clean any remaining temp files..." | tee -a "$log_file"
    # This could try to find all _tmp*.jpg files in out_dir if needed, but per-file is safer.
    echo "Main cleanup finished." | tee -a "$log_file"
}
trap main_cleanup_on_exit EXIT INT TERM


# Check for bc dependency (for MB calculation in logs)
if ! command -v bc &> /dev/null; then
    echo "CRITICAL WARNING: 'bc' command is not installed. MB size calculation in logs will be skipped. Please install 'bc'." | tee -a "$log_file"
    BC_INSTALLED=false
else
    BC_INSTALLED=true
fi

# === Logging Header ===
echo "==== Harvey Norman JPEG Batch Log: $log_file_timestamp ====" > "$log_file"
echo "Script Version Date: 18-May-2025 (jpegtran fix, dependency info, improved logging)" | tee -a "$log_file"
echo "Processing parameters:" | tee -a "$log_file"
echo "  Input Directory: $in_dir" | tee -a "$log_file"
echo "  Output Directory: $out_dir" | tee -a "$log_file"
echo "  Target Chroma Subsampling: $target_sampling_factor" | tee -a "$log_file"
echo "  Target Density: ${target_density} DPI" | tee -a "$log_file"
echo "  ICC Profile for Embedding: $icc_profile" | tee -a "$log_file"
if [ ! -f "$icc_profile" ]; then
    echo "  CRITICAL WARNING: ICC Profile at '$icc_profile' not found. Profile embedding will fail or use fallbacks. This is likely to cause issues." | tee -a "$log_file"
fi
echo "  Max File Size Target: $((max_size / 1024 / 1024)) MB" | tee -a "$log_file"
echo "-----------------------------------------------------" | tee -a "$log_file"

# === Main Processing Loop ===
# Using find for robustness with filenames containing spaces or special characters
find "$in_dir" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | while IFS= read -r -d $'\0' f; do
  # `f` is the full path to the original file
  if [ ! -f "$f" ]; then # Double check it's a file and exists
    echo "Warning: '$f' found by find is not a file or does not exist. Skipping." | tee -a "$log_file"
    continue
  fi

  original_basename=$(basename "$f")
  # Sanitize base name: lowercase, replace non-alphanumeric/hyphen/underscore with underscore, squeeze multiple underscores, cut to 58 chars
  safe_basename=$(echo "${original_basename%.*}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9_-' '_' | sed 's/__\{1,\}/_/g' | cut -c1-58)
  output_filename="${safe_basename}.jpg" # Always .jpg extension
  output_filepath="$out_dir/$output_filename"
  
  file_action_log="Processing Original: $original_basename  ->  Output: $output_filename\n"
  temp_files_to_clean_for_current_file=() # Reset for each file

  # Skip if already processed and meets size criteria
  if [ -f "$output_filepath" ]; then
    # stat -f%z is for macOS, stat -c%s for Linux. Error output suppressed.
    processed_file_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
    if [ -n "$processed_file_size" ] && [ "$processed_file_size" -le "$max_size" ]; then
      echo "🔷 Skipping $original_basename (already processed as $output_filename and under target size)" | tee -a "$log_file"
      file_action_log+="Skipped: $output_filename exists ($((processed_file_size/1024))KB) and is under target size.\n"
      echo -e "\n==== $output_filename - SKIPPED ====\n$file_action_log" >> "$log_file"
      continue
    else
      file_action_log+="Note: $output_filename exists but is over target size ($((processed_file_size/1024))KB) or size unknown. Reprocessing.\n"
    fi
  fi

  echo "🔧 Processing $original_basename -> $output_filename" | tee -a "$log_file"

  current_quality=92 # Starting quality; consider 90-95 based on source material
  # Define temporary file paths for this iteration
  temp_magick_conversion="$out_dir/${safe_basename}_tmp_magick.jpg"; temp_files_to_clean_for_current_file+=("$temp_magick_conversion")
  temp_icc_applied="$out_dir/${safe_basename}_tmp_icc.jpg";       temp_files_to_clean_for_current_file+=("$temp_icc_applied")

  # --- Step 1: Initial Convert, Strip, Set Colorspace/Density (Source: Original $f) ---
  file_action_log+="Step 1: Initial ImageMagick conversion (Quality: ${current_quality}, Chroma: ${target_sampling_factor}, Density: ${target_density}dpi).\n"
  if magick "$f" \
     -strip \
     -depth 8 \
     -quality "$current_quality" \
     -interlace none \
     -sampling-factor "$target_sampling_factor" \
     -colorspace sRGB \
     -units PixelsPerInch -density "$target_density" \
     "$temp_magick_conversion"; then
    file_action_log+="  ImageMagick initial conversion to $temp_magick_conversion: Success.\n"
  else
    file_action_log+="  CRITICAL: ImageMagick initial conversion to $temp_magick_conversion FAILED for $original_basename.\n"
    echo "❌ ERROR: Initial ImageMagick conversion FAILED for $original_basename." | tee -a "$log_file"
    echo -e "\n==== $output_filename - FAILED PREPARATION ====\n$file_action_log" >> "$log_file"
    cleanup_current_file_temps
    continue # Skip to next file
  fi

  # --- Step 2: Embed ICC Profile (Source: $temp_magick_conversion) ---
  file_action_log+="Step 2: Embedding ICC Profile '$icc_profile' into $temp_icc_applied.\n"
  if [ -f "$icc_profile" ]; then
    if magick "$temp_magick_conversion" -profile "$icc_profile" "$temp_icc_applied"; then
      file_action_log+="  ICC profile embedded successfully.\n"
    else
      file_action_log+="  WARNING: Embedding ICC profile '$icc_profile' using ImageMagick FAILED. Copying $temp_magick_conversion to $temp_icc_applied as fallback.\n"
      echo "⚠️ WARNING: Failed to embed ICC profile '$icc_profile' for $original_basename via ImageMagick. $temp_icc_applied will be a copy of $temp_magick_conversion." | tee -a "$log_file"
      cp "$temp_magick_conversion" "$temp_icc_applied"
    fi
  else
    file_action_log+="  CRITICAL WARNING: ICC profile file '$icc_profile' not found. Copying $temp_magick_conversion to $temp_icc_applied without embedding. THIS WILL LIKELY CAUSE ISSUES.\n"
    echo "⚠️ CRITICAL WARNING: ICC Profile '$icc_profile' not found. $temp_icc_applied will be a copy of $temp_magick_conversion for $original_basename." | tee -a "$log_file"
    cp "$temp_magick_conversion" "$temp_icc_applied"
  fi
  
  # --- Step 3: jpegtran Optimization & Final Output Generation (Source: $temp_icc_applied) ---
  file_action_log+="Step 3: Preparing for jpegtran. Input: $temp_icc_applied, Output: $output_filepath.\n"
  # DEBUG: Check the input file for jpegtran
  if [ ! -f "$temp_icc_applied" ]; then
    file_action_log+="  CRITICAL DEBUG: Input file $temp_icc_applied for jpegtran does NOT exist. Skipping jpegtran operation.\n"
    echo "❌ DEBUG ERROR: $temp_icc_applied does not exist before jpegtran call for $original_basename. Copying $temp_icc_applied (which is likely $temp_magick_conversion) to $output_filepath as a last resort." | tee -a "$log_file"
    cp "$temp_magick_conversion" "$output_filepath" # Fallback if $temp_icc_applied is missing
  elif [ ! -s "$temp_icc_applied" ]; then
    file_action_log+="  CRITICAL DEBUG: Input file $temp_icc_applied for jpegtran is EMPTY. Skipping jpegtran operation.\n"
    echo "❌ DEBUG ERROR: $temp_icc_applied is EMPTY before jpegtran call for $original_basename. Copying $temp_magick_conversion to $output_filepath as a last resort." | tee -a "$log_file"
    cp "$temp_magick_conversion" "$output_filepath" # Fallback if $temp_icc_applied is empty
  else
    temp_icc_applied_size_debug=$(stat -f%z "$temp_icc_applied" 2>/dev/null || stat -c%s "$temp_icc_applied" 2>/dev/null)
    file_type_check_debug=$(file "$temp_icc_applied" 2>/dev/null || echo "file command failed")
    file_action_log+="  DEBUG: Input file $temp_icc_applied exists. Size: ${temp_icc_applied_size_debug:-N/A} bytes. Type: $file_type_check_debug\n"

    # Try jpegtran without -baseline first
    file_action_log+="  Attempting jpegtran command (1st try, no -baseline): command jpegtran -copy none -optimize -outfile \"$output_filepath\" \"$temp_icc_applied\"\n"
    if command jpegtran -copy none -optimize -outfile "$output_filepath" "$temp_icc_applied"; then
      file_action_log+="  jpegtran processing (1st try, no -baseline): Success. Output is $output_filepath.\n"
    else
      file_action_log+="  WARNING: jpegtran processing FAILED for $original_basename (1st try, no -baseline).\n"
      file_action_log+="    Attempting jpegtran command (2nd try, WITH -baseline): command jpegtran -copy none -optimize -baseline -outfile \"$output_filepath\" \"$temp_icc_applied\"\n"
      echo "⚠️ WARNING: First jpegtran attempt FAILED for $original_basename. Retrying with -baseline..." | tee -a "$log_file"
      
      if command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied"; then
          file_action_log+="  jpegtran processing (2nd try, WITH -baseline fallback): Success. Output is $output_filepath.\n"
          echo "✅ INFO: Second jpegtran attempt (with -baseline) SUCCEEDED for $original_basename." | tee -a "$log_file"
      else
          file_action_log+="  CRITICAL: jpegtran processing FAILED AGAIN for $original_basename (2nd try, WITH -baseline).\n"
          # Capture error output from jpegtran if possible
          jpegtran_error_output=$(command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied" 2>&1)
          file_action_log+="    jpegtran error output (if any from 2nd try): $jpegtran_error_output\n"
          echo "❌ ERROR: Both jpegtran attempts FAILED for $original_basename. Check log for details. The file $temp_icc_applied might be problematic." | tee -a "$log_file"
          # As a last resort, if jpegtran fails completely, copy the ImageMagick output directly
          # This means the jpegtran benefits (like perfect optimization or strict baseline) are lost.
          file_action_log+="    FALLBACK: Copying $temp_icc_applied to $output_filepath as jpegtran failed.\n"
          cp "$temp_icc_applied" "$output_filepath"
          echo "    FALLBACK: Copied $temp_icc_applied to $output_filepath due to jpegtran failure." | tee -a "$log_file"
      fi
    fi
  fi
  
  current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
  actions_summary_log="Initial Q$current_quality, Chroma ${target_sampling_factor}, ${target_density}dpi. Size: $((current_output_size/1024))KB."

  # --- Step 4: Adaptive Quality Reduction (if oversized) ---
  file_action_log+="Step 4: Adaptive quality reduction if needed (Min Q70 for this phase).\n"
  while [ "$current_output_size" -gt "$max_size" ] && [ "$current_quality" -ge 70 ]; do
    previous_loop_quality=$current_quality
    if [ "$current_output_size" -gt "$plus10MB" ]; then current_quality=$((current_quality - 12));
    elif [ "$current_output_size" -gt "$plus5MB" ]; then current_quality=$((current_quality - 8));
    else current_quality=$((current_quality - 4));
    fi
    [ "$current_quality" -lt 70 ] && current_quality=70
    
    file_action_log+="  Size $((current_output_size/1024))KB > $((max_size/1024))KB. Reducing Q from $previous_loop_quality to $current_quality (Source: Original $original_basename).\n"
    magick "$f" -strip -depth 8 -quality "$current_quality" -interlace none -sampling-factor "$target_sampling_factor" -colorspace sRGB -units PixelsPerInch -density "$target_density" "$temp_magick_conversion"
    if [ -f "$icc_profile" ]; then magick "$temp_magick_conversion" -profile "$icc_profile" "$temp_icc_applied" 2>/dev/null || cp "$temp_magick_conversion" "$temp_icc_applied"; else cp "$temp_magick_conversion" "$temp_icc_applied"; fi
    # Re-run jpegtran after quality change
    if ! command jpegtran -copy none -optimize -outfile "$output_filepath" "$temp_icc_applied"; then
        command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied" || cp "$temp_icc_applied" "$output_filepath" # Fallback to copy if jpegtran still fails
    fi
    current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
    actions_summary_log+=" | QDrop $previous_loop_quality→$current_quality. Size: $((current_output_size/1024))KB."
    file_action_log+="    New size: $((current_output_size/1024))KB.\n"
    if [ "$previous_loop_quality" -eq "$current_quality" ]; then file_action_log+="    Quality unchanged at $current_quality, breaking coarse QDrop loop.\n"; break; fi
  done

  # --- Step 5: Fine-tune Quality by 1-point drops (if still slightly oversized) ---
  file_action_log+="Step 5: Fine-tune quality reduction if needed (Min Q60 for this phase).\n"
  while [ "$current_output_size" -gt "$max_size" ] && [ "$current_quality" -gt 60 ]; do
    previous_loop_quality=$current_quality
    current_quality=$((current_quality - 1))

    file_action_log+="  Size $((current_output_size/1024))KB > $((max_size/1024))KB. Fine-tuning Q from $previous_loop_quality to $current_quality (Source: Original $original_basename).\n"
    magick "$f" -strip -depth 8 -quality "$current_quality" -interlace none -sampling-factor "$target_sampling_factor" -colorspace sRGB -units PixelsPerInch -density "$target_density" "$temp_magick_conversion"
    if [ -f "$icc_profile" ]; then magick "$temp_magick_conversion" -profile "$icc_profile" "$temp_icc_applied" 2>/dev/null || cp "$temp_magick_conversion" "$temp_icc_applied"; else cp "$temp_magick_conversion" "$temp_icc_applied"; fi
    if ! command jpegtran -copy none -optimize -outfile "$output_filepath" "$temp_icc_applied"; then
        command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied" || cp "$temp_icc_applied" "$output_filepath"
    fi
    current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
    actions_summary_log+=" | QFineTune $previous_loop_quality→$current_quality. Size: $((current_output_size/1024))KB."
    file_action_log+="    New size: $((current_output_size/1024))KB.\n"
    if [ "$previous_loop_quality" -eq "$current_quality" ]; then file_action_log+="    Quality unchanged at $current_quality, breaking fine-tune QDrop loop.\n"; break; fi
  done

  # --- Step 6: Resize Loops (if still oversized after quality adjustments) ---
  input_for_resize="$output_filepath" # Start with the current output file
  current_width_for_resize=$(magick identify -format "%w" "$input_for_resize" 2>/dev/null)
  current_height_for_resize=$(magick identify -format "%h" "$input_for_resize" 2>/dev/null)
  file_action_log+="Step 6: Resize loops if needed. Initial dims for resize: ${current_width_for_resize}x${current_height_for_resize} @ Q${current_quality}.\n"

  # Resize 10% steps
  while [ "$current_output_size" -gt "$low_size_threshold" ] && [ "$current_width_for_resize" -gt 200 ] && [ "$current_height_for_resize" -gt 200 ]; do
    previous_w_resize=$current_width_for_resize; previous_h_resize=$current_height_for_resize
    current_width_for_resize=$((current_width_for_resize * 90 / 100))
    current_height_for_resize=$((current_height_for_resize * 90 / 100))
    [ "$current_width_for_resize" -lt 100 ] || [ "$current_height_for_resize" -lt 100 ] && { file_action_log+="  Resize (10% step) halted: dimensions would become too small.\n"; break; }

    file_action_log+="  Resize 10%: Size $((current_output_size/1024))KB > $((low_size_threshold/1024))KB. From ${previous_w_resize}x${previous_h_resize} to ${current_width_for_resize}x${current_height_for_resize}.\n"
    magick "$input_for_resize" -resize "${current_width_for_resize}x${current_height_for_resize}" \
           -quality "$current_quality" -strip -depth 8 -interlace none \
           -sampling-factor "$target_sampling_factor" -colorspace sRGB \
           -units PixelsPerInch -density "$target_density" "$temp_magick_conversion"
    if [ -f "$icc_profile" ]; then magick "$temp_magick_conversion" -profile "$icc_profile" "$temp_icc_applied" 2>/dev/null || cp "$temp_magick_conversion" "$temp_icc_applied"; else cp "$temp_magick_conversion" "$temp_icc_applied"; fi
    if ! command jpegtran -copy none -optimize -outfile "$output_filepath" "$temp_icc_applied"; then
        command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied" || cp "$temp_icc_applied" "$output_filepath"
    fi
    input_for_resize="$output_filepath" 
    current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
    current_width_for_resize=$(magick identify -format "%w" "$input_for_resize" 2>/dev/null) 
    current_height_for_resize=$(magick identify -format "%h" "$input_for_resize" 2>/dev/null)
    actions_summary_log+=" | Resize10% ${previous_w_resize}x${previous_h_resize}→${current_width_for_resize}x${current_height_for_resize}. Size: $((current_output_size/1024))KB."
    file_action_log+="    New size: $((current_output_size/1024))KB. New Dims: ${current_width_for_resize}x${current_height_for_resize}.\n"
  done
  
  # Resize 5% steps
  while [ "$current_output_size" -gt "$med_size_threshold" ] && [ "$current_width_for_resize" -gt 200 ] && [ "$current_height_for_resize" -gt 200 ]; do
    previous_w_resize=$current_width_for_resize; previous_h_resize=$current_height_for_resize
    current_width_for_resize=$((current_width_for_resize * 95 / 100))
    current_height_for_resize=$((current_height_for_resize * 95 / 100))
    [ "$current_width_for_resize" -lt 100 ] || [ "$current_height_for_resize" -lt 100 ] && { file_action_log+="  Resize (5% step) halted: dimensions would become too small.\n"; break; }

    file_action_log+="  Resize 5%: Size $((current_output_size/1024))KB > $((med_size_threshold/1024))KB. From ${previous_w_resize}x${previous_h_resize} to ${current_width_for_resize}x${current_height_for_resize}.\n"
    magick "$input_for_resize" -resize "${current_width_for_resize}x${current_height_for_resize}" \
           -quality "$current_quality" -strip -depth 8 -interlace none \
           -sampling-factor "$target_sampling_factor" -colorspace sRGB \
           -units PixelsPerInch -density "$target_density" "$temp_magick_conversion"
    if [ -f "$icc_profile" ]; then magick "$temp_magick_conversion" -profile "$icc_profile" "$temp_icc_applied" 2>/dev/null || cp "$temp_magick_conversion" "$temp_icc_applied"; else cp "$temp_magick_conversion" "$temp_icc_applied"; fi
    if ! command jpegtran -copy none -optimize -outfile "$output_filepath" "$temp_icc_applied"; then
        command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied" || cp "$temp_icc_applied" "$output_filepath"
    fi
    input_for_resize="$output_filepath"
    current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
    current_width_for_resize=$(magick identify -format "%w" "$input_for_resize" 2>/dev/null)
    current_height_for_resize=$(magick identify -format "%h" "$input_for_resize" 2>/dev/null)
    actions_summary_log+=" | Resize5% ${previous_w_resize}x${previous_h_resize}→${current_width_for_resize}x${current_height_for_resize}. Size: $((current_output_size/1024))KB."
    file_action_log+="    New size: $((current_output_size/1024))KB. New Dims: ${current_width_for_resize}x${current_height_for_resize}.\n"
  done

  # Resize 1% steps (fine-tuning resize)
  while [ "$current_output_size" -gt "$max_size" ] && [ "$current_width_for_resize" -gt 200 ] && [ "$current_height_for_resize" -gt 200 ]; do
    previous_w_resize=$current_width_for_resize; previous_h_resize=$current_height_for_resize
    current_width_for_resize=$((current_width_for_resize * 99 / 100))
    current_height_for_resize=$((current_height_for_resize * 99 / 100))
    [ "$current_width_for_resize" -lt 100 ] || [ "$current_height_for_resize" -lt 100 ] && { file_action_log+="  Resize (1% step) halted: dimensions would become too small.\n"; break; }
    if [ "$previous_w_resize" -eq "$current_width_for_resize" ] && [ "$previous_h_resize" -eq "$current_height_for_resize" ]; then
        file_action_log+="  Resize (1% step) halted: dimensions not changing significantly (${current_width_for_resize}x${current_height_for_resize}).\n"; break;
    fi

    file_action_log+="  Resize 1%: Size $((current_output_size/1024))KB > $((max_size/1024))KB. From ${previous_w_resize}x${previous_h_resize} to ${current_width_for_resize}x${current_height_for_resize}.\n"
    magick "$input_for_resize" -resize "${current_width_for_resize}x${current_height_for_resize}" \
           -quality "$current_quality" -strip -depth 8 -interlace none \
           -sampling-factor "$target_sampling_factor" -colorspace sRGB \
           -units PixelsPerInch -density "$target_density" "$temp_magick_conversion"
    if [ -f "$icc_profile" ]; then magick "$temp_magick_conversion" -profile "$icc_profile" "$temp_icc_applied" 2>/dev/null || cp "$temp_magick_conversion" "$temp_icc_applied"; else cp "$temp_magick_conversion" "$temp_icc_applied"; fi
    if ! command jpegtran -copy none -optimize -outfile "$output_filepath" "$temp_icc_applied"; then
        command jpegtran -copy none -optimize -baseline -outfile "$output_filepath" "$temp_icc_applied" || cp "$temp_icc_applied" "$output_filepath"
    fi
    input_for_resize="$output_filepath"
    current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null)
    current_width_for_resize=$(magick identify -format "%w" "$input_for_resize" 2>/dev/null)
    current_height_for_resize=$(magick identify -format "%h" "$input_for_resize" 2>/dev/null)
    actions_summary_log+=" | Resize1% ${previous_w_resize}x${previous_h_resize}→${current_width_for_resize}x${current_height_for_resize}. Size: $((current_output_size/1024))KB."
    file_action_log+="    New size: $((current_output_size/1024))KB. New Dims: ${current_width_for_resize}x${current_height_for_resize}.\n"
  done

  # --- Step 7: FINAL OPTIONAL EXIFTOOL STRIP/REBUILD (Uncomment to enable) ---
  # enable_final_exiftool_strip=false # Set to true to enable this section
  # if [ "$enable_final_exiftool_strip" = true ] && command -v exiftool &> /dev/null; then
  #   file_action_log+="Step 7: Applying FINAL AGGRESSIVE EXIFTOOL STRIP to $output_filepath.\n"
  #   echo "    Applying final ExifTool strip for $output_filename..." | tee -a "$log_file"
  #   cp "$output_filepath" "${output_filepath}.bak_before_exiftool_strip"; temp_files_to_clean_for_current_file+=("${output_filepath}.bak_before_exiftool_strip")
  #   if exiftool -all= -overwrite_original "$output_filepath"; then
  #     file_action_log+="  ExifTool -all= (strip all): Success.\n"
  #     if [ -f "$icc_profile" ]; then
  #       if exiftool "-icc_profile<=$icc_profile" -overwrite_original "$output_filepath"; then
  #         file_action_log+="  ExifTool re-embedded ICC profile '$icc_profile': Success.\n"
  #       else
  #         file_action_log+="  WARNING: ExifTool FAILED to re-embed ICC profile '$icc_profile' after stripping.\n"
  #         echo "    ⚠️ WARNING: ExifTool FAILED to re-embed ICC profile for $output_filename after stripping." | tee -a "$log_file"
  #       fi
  #     else
  #       file_action_log+="  WARNING: ICC profile '$icc_profile' not found, cannot re-embed after ExifTool strip.\n"
  #     fi
  #   else
  #     file_action_log+="  WARNING: ExifTool -all= (strip all) FAILED. Restoring from backup.\n"
  #     echo "    ⚠️ WARNING: ExifTool -all= FAILED for $output_filename. Restoring from backup." | tee -a "$log_file"
  #     mv "${output_filepath}.bak_before_exiftool_strip" "$output_filepath" 
  #   fi
  #   # Ensure backup is removed if it exists and wasn't used for restore
  #   if [ -f "${output_filepath}.bak_before_exiftool_strip" ]; then rm "${output_filepath}.bak_before_exiftool_strip"; fi
  #   current_output_size=$(stat -f%z "$output_filepath" 2>/dev/null || stat -c%s "$output_filepath" 2>/dev/null) 
  #   actions_summary_log+=" | FinalExiftoolStrip. Size: $((current_output_size/1024))KB."
  # elif [ "$enable_final_exiftool_strip" = true ]; then
  #    file_action_log+="Step 7: FINAL AGGRESSIVE EXIFTOOL STRIP SKIPPED - exiftool command not found.\n"
  #    echo "    Skipping ExifTool strip for $output_filename - exiftool not found." | tee -a "$log_file"
  # fi
  # --- End of Optional ExifTool Strip section ---

  cleanup_current_file_temps # Clean up _tmp_magick.jpg and _tmp_icc.jpg

  # === DETAILED LOGGING AND VALIDATION FOR THIS FILE ===
  echo -e "\n==== $output_filename - Processing Details Summary ====" >> "$log_file"
  echo -e "$file_action_log" >> "$log_file"
  echo "--- Summary of Transformations Applied ---" >> "$log_file"
  echo "$actions_summary_log" >> "$log_file"
  
  final_width=$(magick identify -format "%w" "$output_filepath" 2>/dev/null)
  final_height=$(magick identify -format "%h" "$output_filepath" 2>/dev/null)
  final_size_kb=$((current_output_size/1024))
  final_size_mb_calc=""
  if [ "$BC_INSTALLED" = true ]; then
    final_size_mb_calc=$(printf "%.2f" "$(echo "$current_output_size / (1024*1024)" | bc -l)")
  else
    final_size_mb_calc="N/A (bc not installed)"
  fi

  echo "--- Final File Validation ($output_filepath) ---" >> "$log_file"
  echo "Final Specs: Quality Used ~Q$current_quality, Dimensions ${final_width}x${final_height}px, Size ${final_size_kb}KB (${final_size_mb_calc} MB)" >> "$log_file"
  
  if command -v exiftool &> /dev/null; then
    echo "ExifTool Validation Data:" >> "$log_file"
    profile_name_final=$(exiftool -s3 -icc_profile_name "$output_filepath" 2>/dev/null)
    jfif_final=$(exiftool -s3 -JFIFVersion "$output_filepath" 2>/dev/null)
    color_space_final=$(exiftool -s3 -ColorSpace "$output_filepath" 2>/dev/null) 
    bits_final=$(exiftool -s3 -BitsPerSample "$output_filepath" 2>/dev/null)    
    compression_type_final=$(exiftool -s3 -Compression "$output_filepath" 2>/dev/null)
    jpeg_process_final=$(exiftool -s3 -JPEGProcess "$output_filepath" 2>/dev/null) 
    chroma_final=$(exiftool -s3 -YCbCrSubSampling "$output_filepath" 2>/dev/null)
    xres_final=$(exiftool -s3 -XResolution "$output_filepath" 2>/dev/null)
    yres_final=$(exiftool -s3 -YResolution "$output_filepath" 2>/dev/null)
    resunit_final=$(exiftool -s3 -ResolutionUnit "$output_filepath" 2>/dev/null) 
    app14_check=$(exiftool -s3 -APP14Flags "$output_filepath" 2>/dev/null) 
    any_exif_tags=$(exiftool -G1 -s -if '$EXIF:all' -EXIF:all "$output_filepath" 2>/dev/null)

    echo "  ICC Profile Name: ${profile_name_final:-Not set or unreadable by exiftool}" >> "$log_file"
    echo "  JFIF Version: ${jfif_final:-Not Present (IMPORTANT - usually expected)}" >> "$log_file"
    echo "  Colorspace (reported by ExifTool): ${color_space_final:-Not specified}" >> "$log_file"
    echo "  Bits Per Sample: ${bits_final:-Not specified (expected 8)}" >> "$log_file"
    echo "  Compression Type (ExifTool): ${compression_type_final:-Not specified (expected JPEG)}" >> "$log_file"
    echo "  JPEG Process (ExifTool): ${jpeg_process_final:-Not specified (expected 0 for Baseline or 'Baseline')}" >> "$log_file"
    echo "  Chroma Subsampling: ${chroma_final:-Not Present or N/A (check against known good)}" >> "$log_file"
    echo "  Resolution: ${xres_final:-N/A}x${yres_final:-N/A} ${resunit_final:-Unit N/A (DPI)}" >> "$log_file"
    echo "  Adobe APP14 Flags: ${app14_check:-Not Present (Good - indicates no Adobe APP14 marker)}" >> "$log_file"
    if [ -n "$any_exif_tags" ]; then
      echo "  EXIF Data: PRESENT (CRITICAL - This is UNEXPECTED after -strip and jpegtran -copy none. Review.)" >> "$log_file"
    else
      echo "  EXIF Data: Not Present (Correctly Stripped)" >> "$log_file"
    fi
  else
    echo "ExifTool not found. Skipping detailed ExifTool validation." >> "$log_file"
  fi
  
  file_type_final_output=$(file "$output_filepath" 2>/dev/null || echo "file command failed")
  echo "  'file' command output: ${file_type_final_output:-'file command failed or no output'}" >> "$log_file"
  echo "-----------------------------------" >> "$log_file"

  # Final result (to screen and summary log)
  if [ "$current_output_size" -le "$max_size" ]; then
    echo "✅ $output_filename: Q~$current_quality, ${final_width}x${final_height}, ${final_size_mb_calc}MB. Chroma: ${chroma_final:-N/A}. JFIF: ${jfif_final:-N/A}." | tee -a "$log_file"
  else
    echo "❌ $output_filename: STILL OVER TARGET! Final ${final_size_mb_calc}MB at Q~$current_quality, ${final_width}x${final_height}. Chroma: ${chroma_final:-N/A}." | tee -a "$log_file"
  fi

done

echo "" | tee -a "$log_file"
echo "==== Batch finished: $(date) ====" | tee -a "$log_file"
echo "Batch log written to $log_file"
echo "Review log for any WARNINGS or ERRORS, especially regarding jpegtran or ExifTool."
if ! "$BC_INSTALLED"; then
    echo "Note: 'bc' command was not found, so MB file sizes in logs may show as 'N/A'."
fi
