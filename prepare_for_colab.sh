#!/bin/bash
# Prepare PoT codebase for Colab upload

echo "📦 Preparing PoT codebase for Google Colab..."

# Create a clean package
echo "Creating pot_package.tar.gz..."

# Create tar archive of just the pot folder
tar -czf ~/Desktop/pot_package.tar.gz pot/

echo "✅ Package created at ~/Desktop/pot_package.tar.gz"
echo ""
echo "📤 Upload instructions:"
echo "1. Upload pot_package.tar.gz to your Google Drive"
echo "2. Run the Colab notebook to extract and use your code"
echo ""
echo "Package contains your entire pot/ folder with:"
ls -la pot/ | grep "^d" | wc -l
echo "directories and all your actual PoT code!"