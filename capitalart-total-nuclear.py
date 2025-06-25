#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 📦 CapitalArt Project Utility Toolkit
# 🔧 FILE: capitalart-total-nuclear.py
# 🧠 RULESET: Robbie's Rules™ — No scaffolding, no placeholders, full functionality
# ===========================================

# --- [ 1a: Standard Library Imports | total-nuclear-1a ] ---
import os
import sys
import datetime
import subprocess
from pathlib import Path

# --- [ 1b: Typing Imports | total-nuclear-1b ] ---
from typing import List, Generator

# ===========================================
# 2. ⏰ Timestamp Utility
# ===========================================

def get_timestamp() -> str:
    """Returns formatted timestamp for folder and filenames."""
    return datetime.datetime.now().strftime("REPORTS-%d-%b-%Y-%I-%M%p").upper()

# ===========================================
# 3. 📁 Directory Management
# ===========================================

def create_reports_folder() -> Path:
    """Creates the timestamped reports folder."""
    timestamp = get_timestamp()
    folder_path = Path("reports") / timestamp
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created report folder: {folder_path}")
    return folder_path

# ===========================================
# 4. 🧽 Safe File Collector
# ===========================================

def get_filtered_py_files(base_path: Path = Path(".")) -> Generator[Path, None, None]:
    """
    Yields `.py` files while excluding virtual environments, git folders, caches, and junk files.
    """
    exclude_dirs = {"venv", "__pycache__", ".git", ".vscode", ".idea", "node_modules"}
    for path in base_path.rglob("*.py"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        yield path

# ===========================================
# 5. 📜 Code Snapshot Generator
# ===========================================

def gather_code_snapshot(folder: Path) -> Path:
    """Gathers filtered `.py` files and writes a combined snapshot .md file."""
    md_path = folder / f"report_code_snapshot_{folder.name.lower()}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# 🧠 Code Snapshot — {folder.name}\n\n")
        for pyfile in get_filtered_py_files():
            rel_path = pyfile.relative_to(Path("."))
            md_file.write(f"\n---\n## 📄 {rel_path}\n\n```python\n")
            with open(pyfile, "r", encoding="utf-8") as source:
                md_file.write(source.read())
            md_file.write("\n```\n")
    print(f"✅ Snapshot saved to: {md_path}")
    return md_path

# ===========================================
# 6. 🧪 Dependency Check Utilities (Terminal Only)
# ===========================================

def show_dependency_issues():
    """Prints pip check and pip list --outdated results to terminal."""
    print("\n🔍 Running pip check...")
    subprocess.run(["pip", "check"])

    print("\n📦 Outdated packages:")
    subprocess.run(["pip", "list", "--outdated"])

# ===========================================
# 7. 🚀 Main Entry Point
# ===========================================

def main():
    print("🧠 Starting CapitalArt Report Generator...")
    reports_folder = create_reports_folder()
    gather_code_snapshot(reports_folder)
    show_dependency_issues()
    print("🎉 All done! Snapshot generated and pip health shown.")

# ===========================================
# 🔚 Run Script
# ===========================================

if __name__ == "__main__":
    main()
