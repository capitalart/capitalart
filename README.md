# 🎨 CapitalArt Mockup Generator

Welcome to **CapitalArt**, a lightweight yet powerful mockup generation system designed to categorise, preview, and finalise high-quality mockups for digital artworks — all from the comfort of your local environment or server.

This system helps artists like me (Robin Custance — Aboriginal Aussie artist and part-time Kangaroo whisperer 🦘🎨) bulk-organise, intelligently analyse, and preview professional product mockups for marketplaces like Etsy.

---

## 🔧 Project Features

- ✅ **Mockup Categorisation** using OpenAI Vision (GPT-4.1 / GPT-4o)
- ✅ **Automatic Folder Sorting** based on AI-detected room types
- ✅ **Flask UI** to preview randomly selected mockups (1 per category)
- ✅ **Swap / Regenerate** functionality for better aesthetic control
- ✅ **Ready for Composite Generation** and final publishing
- ✅ Designed to support multiple **aspect ratios** like 4:5, 1:1, etc.

---

## 📁 Folder Structure

```bash
Capitalart-Mockup-Generator/
├── Input/
│   └── Mockups/
│       ├── 4x5/
│       └── 4x5-categorised/
│           ├── Living Room/
│           ├── Bedroom/
│           ├── Nursery/
│           └── ...
├── Output/
│   └── Composites/
└── mockup_selector_ui.py

pip install -r requirements.txt



Flask
openai
python-dotenv
Pillow
requests


🧩 In Development
🖼 Composite Generator (overlay artwork onto mockups)

🧼 Finalisation Script (move print files, create web preview)

📦 Sellbrite/Nembol CSV Exporter

🖼 Aspect Ratio Selector Support

🇦🇺 About the Artist
Hi, I’m Robin Custance — proud Aboriginal Aussie artist and storyteller through colour and dots. I live on Kaurna Country in Adelaide, with ancestral ties to the Boandik people of Naracoorte.

This project supports my mission to share stories through art while helping my family thrive. ❤️

⚡ Contact
💌 rob@asbcreative.com.au

🌐 robincustance.etsy.com

📷 Insta coming soon...