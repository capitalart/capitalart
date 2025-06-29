#!/bin/bash

# ===========================================
# 🔁 CapitalArt Git Update + Commit + Push
# 💥 Robbie Mode™ Gitflow Commander
# Run: ./git-update-push.sh
# ===========================================

# 1. Show current status
echo "📂 Checking git status..."
git status

# 2. Stage everything
echo "➕ Adding all changes..."
git add .

# 3. Prompt for commit message
read -rp "📝 Enter commit message: " commit_msg

# 4. Commit
echo "✅ Committing changes..."
git commit -m "$commit_msg"

# 5. Pull latest from origin/main first (safety first)
echo "🔄 Pulling latest from origin/main..."
git pull origin main --rebase

# 6. Push to main
echo "🚀 Pushing to origin/main..."
git push origin main

echo "✅ All done, Robbie! Git repo is updated and synced. 💚"
