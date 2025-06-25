#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================
# 🚀 CapitalArt App Entry Point
# 🔧 FILE: main.py
# 🧠 RULESET: Robbie's Rules™ — No fluff, just fire
# ===========================================

# --- [ 1a: Standard Imports | main-1a ] ---
import os
import sys

# --- [ 1b: Third-Party Imports | main-1b ] ---
from dotenv import load_dotenv

# --- [ 1c: Flask App Import | main-1c ] ---
from mockup_selector_ui import app  # 🔄 import the initialized Flask app

# ===========================================
# 2. 🌱 Environment Setup
# ===========================================

def init_environment():
    """Loads .env variables and validates critical keys."""
    load_dotenv()
    secret_key = os.getenv("FLASK_SECRET_KEY", "")
    if not secret_key or secret_key == "mockup-secret":
        print("⚠️  WARNING: Using default or missing FLASK_SECRET_KEY. Set a strong one in your .env.")
    app.secret_key = secret_key


# ===========================================
# 3. 🚀 Run the App
# ===========================================

def run_server():
    """Launch the Flask development server."""
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)


# ===========================================
# 🔚 Main Bootstrap
# ===========================================

if __name__ == "__main__":
    print("🎨 Launching CapitalArt Mockup Selector UI...")
    init_environment()
    run_server()
