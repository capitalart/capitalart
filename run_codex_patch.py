#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Run: python3 run_codex_patch.py

"""
run_codex_patch.py
---------------------------------------------------
Apply Codex-generated patch stored in copy-git-apply.txt
and log all activity. Handles patch file generation, logs,
and cleanup. No prompts.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
import shutil
import sys

# === [1. PATHS & SETUP] ===

BASE_DIR = Path(__file__).resolve().parent
PATCHES_DIR = BASE_DIR / "patches"
LOGS_DIR = BASE_DIR / "logs"
PATCH_INPUT = BASE_DIR / "copy-git-apply.txt"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
patch_file = PATCHES_DIR / f"codex_patch_{timestamp}.patch"
log_file = LOGS_DIR / f"patch-log-{timestamp}.txt"

PATCHES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# === [2. CLEAN PATCH TEXT] ===

def extract_clean_patch(text: str) -> str:
    """Extract valid patch lines from heredoc if present."""
    if "<<'EOF'" in text:
        text = text.split("<<'EOF'")[-1].split("EOF")[0]
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("diff --git"):
            return "\n".join(lines[i:]).strip()
    return ""

# === [3. READ PATCH FILE] ===

if not PATCH_INPUT.exists():
    print(f"❌ Error: Patch file not found: {PATCH_INPUT}")
    sys.exit(1)

with PATCH_INPUT.open("r", encoding="utf-8") as f:
    raw_patch = f.read()

clean_patch = extract_clean_patch(raw_patch)

if not clean_patch:
    print("❌ Error: No valid patch content found.")
    with log_file.open("w", encoding="utf-8") as log:
        log.write("❌ No valid patch content found.\n")
        log.write("=== Raw Input ===\n")
        log.write(raw_patch)
    sys.exit(1)

patch_file.write_text(clean_patch, encoding="utf-8")

# === [4. APPLY PATCH] ===

try:
    print("📦 Stashing current changes...")
    subprocess.run(["git", "stash", "push", "-m", "pre Codex task updates"], cwd=BASE_DIR, check=True)

    print(f"🧩 Applying patch: {patch_file.name}")
    result = subprocess.run(
        ["git", "apply", "--3way", str(patch_file)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    with log_file.open("w", encoding="utf-8") as log:
        log.write("=== GIT PATCH APPLY LOG ===\n")
        log.write(f"Patch file: {patch_file}\n")
        log.write(f"Return Code: {result.returncode}\n\n")
        log.write("=== STDOUT ===\n")
        log.write(result.stdout or "(no output)\n")
        log.write("\n=== STDERR ===\n")
        log.write(result.stderr or "(no errors)\n")

        if result.returncode != 0:
            log.write("\n=== GIT STATUS ===\n")
            status = subprocess.run(["git", "status"], cwd=BASE_DIR, capture_output=True, text=True)
            log.write(status.stdout)

    if result.returncode == 0:
        print(f"✅ Patch applied successfully.")
    else:
        print(f"❌ Patch failed! See log: {log_file}")

except Exception as e:
    print(f"💥 Unexpected error: {e}")
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n\n=== Exception ===\n{str(e)}")
