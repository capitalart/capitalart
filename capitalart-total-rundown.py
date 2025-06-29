#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 🔥 CapitalArt Total Nuclear Snapshot v2
# 🚀 Ultimate Dev Toolkit by Robbie Mode™
#
# USAGE EXAMPLES:
#   python3 capitalart-total-rundown.py
#   python3 capitalart-total-rundown.py --no-zip
#   python3 capitalart-total-rundown.py --skip-git --skip-env
# ===========================================

# --- [ 1a: Standard Library Imports | rundown-1a ] ---
import os
import sys
import datetime
import subprocess
import zipfile
import py_compile
from pathlib import Path
from typing import Generator
import argparse

# --- [ 1b: Snapshot Configuration | rundown-1b ] ---
ALLOWED_EXTENSIONS = {".py", ".sh", ".jsx", ".txt", ".html", ".js", ".css"}
EXCLUDED_EXTENSIONS = {".json"}
EXCLUDED_FOLDERS = {"venv", ".venv", "__MACOSX", ".git", ".vscode", "reports", "backups", "node_modules", ".idea"}
EXCLUDED_FILES = {".DS_Store"}

# ===========================================
# 2. 🧭 Timestamp + Report Folder
# ===========================================

def get_timestamp() -> str:
    return datetime.datetime.now().strftime("REPORTS-%d-%b-%Y-%I-%M%p").upper()

def create_reports_folder() -> Path:
    timestamp = get_timestamp()
    folder_path = Path("reports") / timestamp
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created report folder: {folder_path}")
    return folder_path

# ===========================================
# 3. 🔍 Collect Valid Files
# ===========================================

def get_included_files() -> Generator[Path, None, None]:
    for path in Path(".").rglob("*"):
        if (
            path.is_file()
            and path.suffix in ALLOWED_EXTENSIONS
            and path.suffix not in EXCLUDED_EXTENSIONS
            and path.name not in EXCLUDED_FILES
            and not any(str(part).lower() in EXCLUDED_FOLDERS for part in path.parts)
        ):
            yield path

# ===========================================
# 4. 📄 Markdown Code Snapshot
# ===========================================

def gather_code_snapshot(folder: Path) -> Path:
    md_path = folder / f"report_code_snapshot_{folder.name.lower()}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# 🧠 CapitalArt Code Snapshot — {folder.name}\n\n")
        for file in get_included_files():
            rel_path = file.relative_to(Path("."))
            print(f"📄 Including file: {rel_path}")
            md_file.write(f"\n---\n## 📄 {rel_path}\n\n```{file.suffix[1:]}\n")
            try:
                with open(file, "r", encoding="utf-8") as f:
                    md_file.write(f.read())
            except Exception as e:
                md_file.write(f"[ERROR READING FILE: {e}]")
            md_file.write("\n```\n")
    print(f"✅ Code snapshot saved to: {md_path}")
    return md_path

# ===========================================
# 5. 📊 File Summary Table
# ===========================================

def generate_file_summary(folder: Path) -> None:
    summary_path = folder / "file_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 📊 File Summary\n\n")
        f.write("| File | Size (KB) | Last Modified |\n")
        f.write("|------|------------|----------------|\n")
        for file in get_included_files():
            size_kb = round(file.stat().st_size / 1024, 1)
            mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            rel = file.relative_to(Path("."))
            f.write(f"| `{rel}` | {size_kb} KB | {mtime:%Y-%m-%d %H:%M} |\n")
    print(f"📋 File summary written to: {summary_path}")

# ===========================================
# 6. 🧪 Python Syntax Validation
# ===========================================

def validate_python_files() -> None:
    print("\n🧪 Validating Python syntax...")
    for file in get_included_files():
        if file.suffix == ".py":
            try:
                py_compile.compile(file, doraise=True)
                print(f"✅ {file}")
            except py_compile.PyCompileError as e:
                print(f"❌ {file} → {e.msg}")

# ===========================================
# 7. 🧬 Git Status + Commit Info
# ===========================================

def log_git_status(folder: Path) -> None:
    git_path = folder / "git_snapshot.txt"
    with open(git_path, "w", encoding="utf-8") as f:
        f.write("🔧 Git Status:\n")
        subprocess.run(["git", "status"], stdout=f, stderr=subprocess.DEVNULL)
        f.write("\n🔁 Last Commit:\n")
        subprocess.run(["git", "log", "-1"], stdout=f, stderr=subprocess.DEVNULL)
        f.write("\n🧾 Diff Summary:\n")
        subprocess.run(["git", "diff", "--stat"], stdout=f, stderr=subprocess.DEVNULL)
    print(f"📘 Git snapshot written to: {git_path}")

# ===========================================
# 8. 🌐 Environment Metadata
# ===========================================

def log_environment_details(folder: Path) -> None:
    env_path = folder / "env_metadata.txt"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("🐍 Python Version:\n")
        subprocess.run(["python3", "--version"], stdout=f)
        f.write("\n🖥️ Platform Info:\n")
        subprocess.run(["uname", "-a"], stdout=f)
        f.write("\n📦 Installed Packages:\n")
        subprocess.run(["pip", "freeze"], stdout=f)
    print(f"📚 Environment metadata saved to: {env_path}")

# ===========================================
# 9. 🧪 Environment Dependency Report
# ===========================================

def show_dependency_issues() -> None:
    print("\n🔍 Running pip check...")
    subprocess.run(["pip", "check"])
    print("\n📦 Outdated packages:")
    subprocess.run(["pip", "list", "--outdated"])

# ===========================================
# 10. 📦 Zip the Report Folder
# ===========================================

def zip_report_folder(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in folder.rglob("*"):
            zipf.write(file, file.relative_to(folder.parent))
    print(f"📦 Report zipped to: {zip_path}")
    return zip_path

# ===========================================
# 11. 🧰 CLI Args
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(description="CapitalArt Dev Snapshot Generator")
    parser.add_argument("--no-zip", action="store_true", help="Skip ZIP archive creation")
    parser.add_argument("--skip-env", action="store_true", help="Skip environment metadata logging")
    parser.add_argument("--skip-validate", action="store_true", help="Skip Python syntax validation")
    parser.add_argument("--skip-git", action="store_true", help="Skip Git snapshot logging")
    parser.add_argument("--skip-pip-check", action="store_true", help="Skip pip dependency checks")
    return parser.parse_args()

# ===========================================
# 12. 🚀 Main Execution
# ===========================================

def main():
    args = parse_args()
    print("🎨 Generating CapitalArt Total Nuclear Snapshot (v2)...")

    report_folder = create_reports_folder()
    gather_code_snapshot(report_folder)
    generate_file_summary(report_folder)

    if not args.skip_validate:
        validate_python_files()
    if not args.skip_env:
        log_environment_details(report_folder)
    if not args.skip_git:
        log_git_status(report_folder)
    if not args.skip_pip_check:
        show_dependency_issues()
    if not args.no_zip:
        zip_report_folder(report_folder)

    print("✅ Snapshot complete. All systems green, Robbie! 💚")

# ===========================================
# 🔚 Entry Point
# ===========================================

if __name__ == "__main__":
    main()
