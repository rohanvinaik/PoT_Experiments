#!/bin/bash
# Watchdog to monitor downloads and restart if needed

LOG_FILE="download_watchdog.log"
STALL_COUNT=0
LAST_SIZE=0
MAX_STALLS=6  # Allow 3 minutes of no progress before action

echo "🐕 Watchdog started at $(date)" | tee -a $LOG_FILE

while true; do
    # Check current cache size
    CURRENT_SIZE=$(du -s ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/ 2>/dev/null | cut -f1)
    
    # Check active downloads
    PYTHON_COUNT=$(ps aux | grep -E "python.*zephyr|snapshot_download" | grep -v grep | wc -l)
    WGET_COUNT=$(ps aux | grep "wget.*safetensors" | grep -v grep | wc -l)
    
    # Log status
    echo "[$(date '+%H:%M:%S')] Size: ${CURRENT_SIZE}KB, Python: $PYTHON_COUNT, Wget: $WGET_COUNT" >> $LOG_FILE
    
    # Check for progress
    if [ "$CURRENT_SIZE" -eq "$LAST_SIZE" ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        echo "⚠️  No progress for $STALL_COUNT cycles" | tee -a $LOG_FILE
        
        if [ $STALL_COUNT -ge $MAX_STALLS ]; then
            echo "🔄 Download stalled! Attempting restart..." | tee -a $LOG_FILE
            
            # Try to restart HuggingFace download
            python3 << 'PYEOF' &
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '7200'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from huggingface_hub import snapshot_download
print("Restarting download with watchdog...")

try:
    snapshot_download(
        "HuggingFaceH4/zephyr-7b-beta",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,
    )
    print("✅ Download complete!")
    # Run test immediately
    exec(open("run_mistral_zephyr.py").read())
except Exception as e:
    print(f"Download error: {e}")
PYEOF
            
            STALL_COUNT=0
        fi
    else
        if [ $STALL_COUNT -gt 0 ]; then
            echo "✅ Progress resumed!" | tee -a $LOG_FILE
        fi
        STALL_COUNT=0
        LAST_SIZE=$CURRENT_SIZE
    fi
    
    # Check if downloads completed
    COMPLETE_COUNT=$(ls ~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/blobs/*.incomplete 2>/dev/null | wc -l)
    if [ "$COMPLETE_COUNT" -eq 0 ] && [ "$CURRENT_SIZE" -gt 14000000 ]; then
        echo "🎉 Downloads appear complete! Running test..." | tee -a $LOG_FILE
        python3 run_mistral_zephyr.py
        break
    fi
    
    # Alert if processes died unexpectedly
    if [ "$PYTHON_COUNT" -eq 0 ] && [ "$WGET_COUNT" -eq 0 ]; then
        echo "❌ All download processes died! Restarting..." | tee -a $LOG_FILE
        # Restart main download
        python3 << 'PYEOF' &
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceH4/zephyr-7b-beta", resume_download=True, max_workers=1)
PYEOF
    fi
    
    sleep 30
done
