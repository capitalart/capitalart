#!/bin/bash

# ========================================================
# CapitalArt GitHub Pages Setup Script
# By: Robin Custance — Aboriginal Artist & Kangaroo Whisperer 🦘
# Purpose: Bootstraps a lightweight website to preview art mockups
# ========================================================

echo "🛠️  Setting up CapitalArt GitHub Pages project..."

# -------------------------------
# 1. Create Basic Project Structure
# -------------------------------
mkdir -p assets/mockups descriptions scripts

touch index.html
touch assets/style.css
touch main.js
touch scripts/analyze.js
touch .nojekyll   # prevent GitHub from ignoring folders starting with "_"

# -------------------------------
# 2. Write index.html Skeleton
# -------------------------------
cat > index.html <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CapitalArt Mockup Gallery</title>
  <link rel="stylesheet" href="assets/style.css" />
</head>
<body>
  <header>
    <h1>🎨 CapitalArt Mockup Gallery</h1>
    <p>Browse artworks, categories, and AI-generated Etsy listings</p>
  </header>

  <main id="gallery">
    <!-- Artworks will be injected here by main.js -->
  </main>

  <footer>
    <p>© Robin Custance • Proudly on Kaurna Country • GitHub Pages powered</p>
  </footer>

  <script src="main.js"></script>
</body>
</html>
EOF

# -------------------------------
# 3. Write style.css Boilerplate
# -------------------------------
cat > assets/style.css <<EOF
body {
  font-family: system-ui, sans-serif;
  margin: 0;
  background: #f9f9f9;
  color: #222;
}
header, footer {
  background: #333;
  color: #fff;
  padding: 1em;
  text-align: center;
}
main {
  display: flex;
  flex-wrap: wrap;
  padding: 2em;
  gap: 1em;
  justify-content: center;
}
.artwork {
  border: 1px solid #ccc;
  background: #fff;
  padding: 1em;
  max-width: 300px;
}
.artwork img {
  max-width: 100%;
  height: auto;
  display: block;
}
EOF

# -------------------------------
# 4. Add JavaScript Loader
# -------------------------------
cat > main.js <<EOF
document.addEventListener("DOMContentLoaded", () => {
  fetch("./descriptions/artworks.json")
    .then((res) => res.json())
    .then((artworks) => {
      const gallery = document.getElementById("gallery");
      artworks.forEach(({ title, image, description }) => {
        const el = document.createElement("div");
        el.className = "artwork";
        el.innerHTML = \`
          <img src="\${image}" alt="\${title}" />
          <h2>\${title}</h2>
          <p>\${description}</p>
        \`;
        gallery.appendChild(el);
      });
    })
    .catch(() => {
      document.getElementById("gallery").innerHTML =
        "<p>⚠️ No artwork found yet. Add some to descriptions/artworks.json</p>";
    });
});
EOF

# -------------------------------
# 5. Add Sample artworks.json
# -------------------------------
cat > descriptions/artworks.json <<EOF
[
  {
    "title": "Red Dreaming Hills",
    "image": "assets/mockups/red-dreaming-001.jpg",
    "description": "A fiery abstract Aboriginal-style landscape, inspired by Kaurna Country."
  },
  {
    "title": "Cosmic Possum Flow",
    "image": "assets/mockups/cosmic-possum-002.jpg",
    "description": "A spiritual journey of the flying possum through the night sky."
  }
]
EOF

# -------------------------------
# 6. Git Commit & Push
# -------------------------------
git add .
git commit -m "🚀 Initial GitHub Pages setup for CapitalArt gallery"
git push origin master

echo "✅ Setup complete. Visit your site at:"
echo "   https://capitalart.github.io/capitalart"
