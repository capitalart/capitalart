#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 📦 CapitalArt Project Utility Toolkit
# 🔧 FILE: capitalart-total-nuclear.py
# 🧠 RULESET: Robbie's Rules™ — No fluff, no filler, just what matters
# ===========================================

# --- [ 1a: Standard Library Imports | nuclear-1a ] ---
import os
import sys
import datetime
import subprocess
from pathlib import Path

# --- [ 1b: Typing Imports | nuclear-1b ] ---
from typing import Generator

# ===========================================
# 2. ⏰ Timestamp Utility
# ===========================================

def get_timestamp() -> str:
    """Returns formatted timestamp for folder and filenames."""
    return datetime.datetime.now().strftime("REPORTS-%d-%b-%Y-%I-%M%p").upper()

# ===========================================
# 3. 📁 Report Directory Setup
# ===========================================

def create_reports_folder() -> Path:
    """Creates the timestamped reports folder inside /reports."""
    timestamp = get_timestamp()
    folder_path = Path("reports") / timestamp
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created report folder: {folder_path}")
    return folder_path

# ===========================================
# 4. 📦 Included File Targets
# ===========================================

# 🔒 Allow only specific extensions
ALLOWED_EXTENSIONS = [".py", ".sh", ".jsx", ".txt", ".html", ".js", ".css"]

# 🎯 Paths to include in the report
INCLUDE_PATHS = [
    Path("mockup_selector_ui.py"),
    Path("mockup_categoriser.py"),
    Path("main.py"),
    Path("app.py"),
    Path("generate_folder_tree.py"),
    Path("requirements.txt"),
    Path("templates/index.html"),
    Path("templates/mockup_selector.html"),
    Path("static/js/main.js"),
    Path("static/css/style.css"),
    Path("Capitalart-Mockup-Generator/scripts"),  # 🔥 All bash, jsx, py files
]

def get_included_files() -> Generator[Path, None, None]:
    """Yields only files with allowed extensions from INCLUDE_PATHS."""
    for path in INCLUDE_PATHS:
        if path.is_file() and path.suffix in ALLOWED_EXTENSIONS:
            yield path
        elif path.is_dir():
            for file in path.rglob("*"):
                if file.is_file() and file.suffix in ALLOWED_EXTENSIONS:
                    yield file

# ===========================================
# 5. 🧠 Code Snapshot Generator
# ===========================================

def gather_code_snapshot(folder: Path) -> Path:
    """Creates markdown snapshot of selected source files."""
    md_path = folder / f"report_code_snapshot_{folder.name.lower()}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# 🧠 CapitalArt Snapshot — {folder.name}\n\n")
        for file in get_included_files():
            rel_path = file.relative_to(Path("."))
            print(f"📄 Including file: {rel_path}")
            md_file.write(f"\n---\n## 📄 {rel_path}\n\n```{file.suffix[1:]}\n")
            with open(file, "r", encoding="utf-8") as source:
                md_file.write(source.read())
            md_file.write("\n```\n")
    print(f"✅ Snapshot saved to: {md_path}")
    return md_path

# ===========================================
# 6. 🧪 Environment Dependency Report
# ===========================================

def show_dependency_issues():
    """Outputs pip health checks and outdated packages."""
    print("\n🔍 Running pip check...")
    subprocess.run(["pip", "check"])
    print("\n📦 Outdated packages:")
    subprocess.run(["pip", "list", "--outdated"])

# ===========================================
# 7. 🚀 Main Execution Logic
# ===========================================

def main():
    print("🎨 Generating CapitalArt Dev Snapshot...")
    reports_folder = create_reports_folder()
    gather_code_snapshot(reports_folder)
    show_dependency_issues()
    print("🎉 Done! Snapshot complete. All systems go.")

# ===========================================
# 🔚 Entry Point
# ===========================================

if __name__ == "__main__":
    main()
