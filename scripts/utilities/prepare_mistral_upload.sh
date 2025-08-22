#!/bin/bash
# Prepare Mistral model for Colab upload

echo "📦 Preparing Mistral model for upload..."

# Create a clean directory with just the model files
mkdir -p ~/Desktop/mistral_for_colab

# Copy the actual model files (following symlinks)
cd ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a/

echo "Copying model files..."
cp -L model-00001-of-00003.safetensors ~/Desktop/mistral_for_colab/
cp -L model-00002-of-00003.safetensors ~/Desktop/mistral_for_colab/
cp -L model-00003-of-00003.safetensors ~/Desktop/mistral_for_colab/

# Also copy consolidated version (alternative format)
cp -L consolidated.safetensors ~/Desktop/mistral_for_colab/

# Copy config files
echo "Copying config files..."
cp *.json ~/Desktop/mistral_for_colab/
cp tokenizer.model ~/Desktop/mistral_for_colab/ 2>/dev/null || true

echo "✅ Files ready in ~/Desktop/mistral_for_colab/"
ls -lah ~/Desktop/mistral_for_colab/

echo ""
echo "📤 Upload instructions:"
echo "1. Go to Google Drive (drive.google.com)"
echo "2. Create folder 'mistral_model'"
echo "3. Upload all files from ~/Desktop/mistral_for_colab/"
echo "4. Files to upload (total ~27GB):"
echo "   - model-00001-of-00003.safetensors (4.6GB)"
echo "   - model-00002-of-00003.safetensors (4.7GB)"  
echo "   - model-00003-of-00003.safetensors (4.2GB)"
echo "   - consolidated.safetensors (14GB) - optional, same weights"
echo "   - All .json files (small)"