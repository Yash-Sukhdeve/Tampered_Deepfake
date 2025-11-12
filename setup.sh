#!/bin/bash
# Production Setup Script for Tampered Deepfake Pipeline
# Author: Tampered Deepfake Research Project

set -e  # Exit on error

echo "========================================="
echo "Tampered Deepfake Pipeline Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python version: $python_version"
echo ""

# Check for GPU
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. Whisper will run on CPU (slower)."
fi
echo ""

# Check for ffmpeg
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg found: $(ffmpeg -version | head -1)"
else
    echo "❌ Error: ffmpeg not found!"
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    exit 1
fi
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Python packages installed"
echo ""

# Install spaCy model
echo "Installing spaCy English model..."
python -m spacy download en_core_web_sm
echo "✓ spaCy model installed"
echo ""

# Check MFA
echo "Checking for Montreal Forced Aligner..."
if command -v mfa &> /dev/null; then
    echo "✓ MFA found: $(mfa version)"
    echo ""
    echo "Downloading MFA models..."
    mfa model download acoustic english || echo "⚠️  Could not download MFA acoustic model"
    mfa model download dictionary english_us_arpa || echo "⚠️  Could not download MFA dictionary"
else
    echo "⚠️  MFA not found (optional but recommended)"
    echo "Install with: conda install -c conda-forge montreal-forced-aligner"
fi
echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p output/{preprocessed,transcripts,tampered,metadata}
echo "✓ Directories created"
echo ""

# Test imports
echo "Testing Python imports..."
python -c "
import whisper
import librosa
import pydub
import spacy
import numpy as np
import tqdm
print('✓ All Python modules imported successfully')
"
echo ""

# Make scripts executable
echo "Making scripts executable..."
chmod +x src/*.py run_pipeline.py
echo "✓ Scripts are executable"
echo ""

echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Place your audio files in data/sub*/original/"
echo "2. Run the pipeline: python run_pipeline.py --all"
echo ""
echo "For more information, see README.md"
