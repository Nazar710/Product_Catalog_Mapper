#!/bin/bash
# Quick Git Update Script for Product Catalog Mapper
# Usage: ./git_update.sh "Your commit message"

echo "🔍 Checking git status..."
git status

echo ""
echo "📦 Adding all changes..."
git add .

if [ -z "$1" ]; then
    echo "⚠️  No commit message provided. Using default message."
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M')"
else
    COMMIT_MSG="$1"
fi

echo "💾 Committing changes: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo "🚀 Pushing to GitHub..."
git push

echo "✅ Done! Check your repository at: https://github.com/Nazar710/Product_Catalog_Mapper"