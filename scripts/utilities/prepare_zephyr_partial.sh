#!/bin/bash
# Prepare partial Zephyr files for upload

echo "📦 Preparing Zephyr partial files..."
echo "Note: We have 11GB of incomplete downloads (8 partial model files)"

# Create directory
mkdir -p ~/Desktop/zephyr_partial

# Copy incomplete files (they might help resume)
echo "Copying partial model files (11GB)..."
cd ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/

# Copy incomplete files and rename without .incomplete
for file in blobs/*.incomplete; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .incomplete)
        echo "Copying $(basename $file) ..."
        cp "$file" ~/Desktop/zephyr_partial/${filename}.partial
    fi
done

# Copy config files that we do have
echo "Copying config files..."
cd snapshots/892b3d7a7b1cf10c7a701c60881cd93df615734c/
cp *.json ~/Desktop/zephyr_partial/ 2>/dev/null || true
cp tokenizer.model ~/Desktop/zephyr_partial/ 2>/dev/null || true

echo ""
echo "✅ Partial files ready in ~/Desktop/zephyr_partial/"
ls -lah ~/Desktop/zephyr_partial/

echo ""
echo "⚠️ IMPORTANT: These are INCOMPLETE files!"
echo "   - Each .partial file is ~1.8GB but needs to be ~1.9GB"
echo "   - Colab will need to complete the download"
echo ""
echo "📤 Upload options:"
echo "1. Skip uploading these (let Colab download fresh) - RECOMMENDED"
echo "2. Upload to try resuming (might not work with .partial files)"
echo ""
echo "For Colab, it's probably better to just download Zephyr fresh"
echo "since we only have incomplete files."