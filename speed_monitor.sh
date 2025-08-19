#!/bin/bash
echo "📊 Speed Monitor Started - Tracking download progress"
echo "================================================="

LAST_SIZE=$(du -s ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/ 2>/dev/null | cut -f1)
LAST_TIME=$(date +%s)

while true; do
    sleep 30
    
    CURRENT_SIZE=$(du -s ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/ 2>/dev/null | cut -f1)
    CURRENT_TIME=$(date +%s)
    
    SIZE_DIFF=$((CURRENT_SIZE - LAST_SIZE))
    TIME_DIFF=$((CURRENT_TIME - LAST_TIME))
    
    if [ $TIME_DIFF -gt 0 ]; then
        SPEED_KB=$((SIZE_DIFF / TIME_DIFF))
        SPEED_MB=$((SPEED_KB / 1024))
        
        SIZE_GB=$((CURRENT_SIZE / 1024 / 1024))
        SIZE_MB=$((CURRENT_SIZE / 1024))
        
        echo -ne "\r📦 Size: ${SIZE_GB}.${SIZE_MB:(-3):1}GB | Speed: ${SPEED_MB}MB/s | "
        
        if [ $SPEED_KB -gt 1000 ]; then
            echo -ne "✅ Good speed!"
        elif [ $SPEED_KB -gt 100 ]; then
            echo -ne "⚠️  Slow speed"
        elif [ $SPEED_KB -gt 0 ]; then
            echo -ne "🐌 Very slow"
        else
            echo -ne "❌ Stalled"
        fi
        
        echo -ne " | $(date '+%H:%M:%S')  "
    fi
    
    LAST_SIZE=$CURRENT_SIZE
    LAST_TIME=$CURRENT_TIME
done
