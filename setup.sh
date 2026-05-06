#!/bin/bash
# ============================================================
# ResumeIQ ATS Analyzer — Setup Script
# Run this once to install all dependencies
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ResumeIQ ATS Analyzer — Setup          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python
python3.12 --version || { echo "Python 3.12 is required"; exit 1; }

# Create virtual environment
echo "→ Creating virtual environment..."
python3.12 -m venv venv

# Activate
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null

# Upgrade pip
pip install --upgrade pip -q

# Install dependencies
echo "→ Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q

# Download spaCy model
echo "→ Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo ""
echo "✓ Setup complete!"
echo ""
echo "To start the app:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open: http://localhost:5000"
echo ""
