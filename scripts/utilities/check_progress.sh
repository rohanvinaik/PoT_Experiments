#!/bin/bash
# Simple download progress checker

CACHE_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta"
EXPECTED_SIZE_GB=14

while true; do
    # Get current size in MB
    if [ -d "$CACHE_DIR" ]; then
        SIZE_MB=$(du -sm "$CACHE_DIR" 2>/dev/null | cut -f1)
        SIZE_GB=$(echo "scale=2; $SIZE_MB / 1024" | bc 2>/dev/null || echo "0")
        PROGRESS=$(echo "scale=1; ($SIZE_GB / $EXPECTED_SIZE_GB) * 100" | bc 2>/dev/null || echo "0")
        
        # Check for incomplete files
        INCOMPLETE=$(find "$CACHE_DIR" -name "*.incomplete" 2>/dev/null | wc -l | xargs)
        
        # Clear line and print progress
        echo "Download Progress: ${SIZE_GB}GB / ${EXPECTED_SIZE_GB}GB (${PROGRESS}%) - Incomplete files: $INCOMPLETE"
        
        # Check if download might be complete
        if (( $(echo "$SIZE_GB > 13" | bc -l) )); then
            echo "✅ Download appears complete or nearly complete!"
        fi
    else
        echo "Waiting for download to start..."
    fi
    
    sleep 10
done