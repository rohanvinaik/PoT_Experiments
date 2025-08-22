#!/bin/bash
# Monitor download and automatically run test when complete

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")/.."

echo "📡 Monitoring Zephyr download progress..."
echo "Will automatically run Mistral vs Zephyr test when complete"
echo "="*60

while true; do
    # Check if all model files exist
    COMPLETE=true
    for i in {1..8}; do
        FILE="$HOME/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/model-0000$i-of-00008.safetensors"
        if [ ! -f "$FILE" ] || [ $(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null) -lt 1000000 ]; then
            COMPLETE=false
            break
        fi
    done
    
    if [ "$COMPLETE" = true ]; then
        echo "✅ Download complete! Running test..."
        python3 "$ROOT_DIR/archived_mistral_tests/run_mistral_zephyr.py"
        break
    fi
    
    # Show progress
    SIZE=$(du -sh ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/ 2>/dev/null | cut -f1)
    echo -ne "\r📦 Progress: $SIZE / ~14GB - $(date '+%H:%M:%S')"

    sleep 30
done

