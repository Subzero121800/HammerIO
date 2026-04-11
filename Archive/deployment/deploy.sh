#!/usr/bin/env bash
# HammerIO GitHub Deployment Script
# Copyright 2026 ResilientMind AI | Joseph C McGinty Jr
#
# Usage: ./deploy/deploy.sh [github-username/repo]
# Example: ./deploy/deploy.sh resilientmindai/hammerio

set -euo pipefail

REPO="${1:-resilientmindai/hammerio}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║      HammerIO → GitHub Deployment            ║"
echo "║  ResilientMind AI | Joseph C McGinty Jr     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Repository: $REPO"
echo "  Source:     $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Check prerequisites
command -v git >/dev/null 2>&1 || { echo "Error: git not found"; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "Warning: gh (GitHub CLI) not found — manual steps needed"; }

# Verify tests pass
echo "Running tests..."
python3 -m pytest tests/ -q --tb=line 2>&1 | tail -3
echo ""

# Check git status
if [ -n "$(git status --porcelain)" ]; then
    echo "Warning: Uncommitted changes detected."
    git status --short
    read -p "Commit all changes before deploying? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        git commit -m "Pre-deployment cleanup"
    fi
fi

# Rename branch to main if needed
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "Renaming branch '$BRANCH' → 'main'..."
    git branch -M main
fi

# Create remote if not exists
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "Adding remote origin..."

    if command -v gh >/dev/null 2>&1; then
        echo "Creating GitHub repo via gh CLI..."
        gh repo create "$REPO" \
            --public \
            --description "GPU where it matters. CPU where it doesn't. Zero configuration." \
            --homepage "https://resilientmindai.com" \
            --source . \
            --remote origin 2>/dev/null || {
                echo "Repo may already exist. Adding remote manually..."
                git remote add origin "https://github.com/$REPO.git"
            }
    else
        echo "Create the repo at: https://github.com/new"
        echo "  Name: $(echo $REPO | cut -d'/' -f2)"
        echo "  Public, no README, Apache 2.0"
        echo ""
        read -p "Press Enter when ready..."
        git remote add origin "https://github.com/$REPO.git"
    fi
fi

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin main

# Set topics
if command -v gh >/dev/null 2>&1; then
    echo "Setting repository topics..."
    gh repo edit "$REPO" \
        --add-topic cuda \
        --add-topic compression \
        --add-topic nvidia-jetson \
        --add-topic gpu \
        --add-topic nvenc \
        --add-topic edge-ai \
        --add-topic media-processing \
        --add-topic python \
        --add-topic zstd \
        --add-topic gstreamer 2>/dev/null || true
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║      Deployment Complete!                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Repository: https://github.com/$REPO"
echo ""
echo "  Next steps:"
echo "    1. Visit https://github.com/$REPO"
echo "    2. Create release: gh release create v0.1.0 --title 'HammerIO v0.1.0'"
echo "    3. Set repo topics and social preview"
echo ""
