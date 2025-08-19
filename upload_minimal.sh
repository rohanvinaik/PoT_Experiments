#!/bin/bash
# Create minimal Mistral model package (just the actual model files)

echo "Creating minimal Mistral package..."

# Create temp directory
mkdir -p ~/Desktop/mistral_minimal

# Find and copy only the actual model files (not symlinks)
cd ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/

# Copy the actual model files from blobs
echo "Copying model files..."
cp blobs/* ~/Desktop/mistral_minimal/ 2>/dev/null || true

# Copy config files from snapshots
echo "Copying config files..."
find snapshots -name "*.json" -exec cp {} ~/Desktop/mistral_minimal/ \; 2>/dev/null || true
find snapshots -name "tokenizer*" -exec cp {} ~/Desktop/mistral_minimal/ \; 2>/dev/null || true

# Create a smaller zip
cd ~/Desktop
zip -r mistral_minimal.zip mistral_minimal/

echo "✅ Created mistral_minimal.zip"
du -sh ~/Desktop/mistral_minimal.zip