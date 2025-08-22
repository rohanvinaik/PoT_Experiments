#!/bin/bash
echo "📊 Monitoring Zephyr download progress..."
echo "Target: 14GB total (11GB already cached)"
echo "=" * 60

LAST_SIZE=0
while true; do
    # Get current cache size
    CURRENT_SIZE=$(du -s ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/ 2>/dev/null | cut -f1)
    CURRENT_MB=$((CURRENT_SIZE / 1024))
    
    # Calculate speed
    if [ $LAST_SIZE -gt 0 ]; then
        DIFF=$((CURRENT_SIZE - LAST_SIZE))
        SPEED_KB=$((DIFF / 30))  # Per second over 30s
        
        if [ $SPEED_KB -gt 0 ]; then
            echo -ne "\r📦 Size: ${CURRENT_MB}MB | Speed: ${SPEED_KB}KB/s | $(date '+%H:%M:%S')  "
        else
            echo -ne "\r📦 Size: ${CURRENT_MB}MB | Speed: Stalled | $(date '+%H:%M:%S')  "
        fi
    else
        echo -ne "\r📦 Size: ${CURRENT_MB}MB | Calculating... | $(date '+%H:%M:%S')  "
    fi
    
    LAST_SIZE=$CURRENT_SIZE
    
    # Check if complete
    INCOMPLETE=$(ls ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/*.incomplete 2>/dev/null | wc -l)
    if [ "$INCOMPLETE" -eq 0 ] && [ "$CURRENT_MB" -gt 14000 ]; then
        echo -e "\n✅ Download complete!"
        break
    fi
    
    sleep 30
done
