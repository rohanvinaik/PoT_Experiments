#!/bin/bash
# Simple progress dashboard for PoT pipeline

echo "Starting PoT Pipeline Monitor..."
echo "Press Ctrl+C to exit (pipeline continues running)"
echo ""

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║             POT FRAMEWORK PIPELINE MONITOR - QWEN 72B               ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📅 Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Get latest log file
    LOG_FILE=$(ls -t experimental_results/qwen_pipeline_*.log 2>/dev/null | head -1)
    
    if [ -z "$LOG_FILE" ]; then
        echo "❌ No pipeline log found!"
        sleep 5
        continue
    fi
    
    # Extract progress info
    PROGRESS=$(grep "Progress:" "$LOG_FILE" 2>/dev/null | tail -1)
    if [ ! -z "$PROGRESS" ]; then
        # Extract numbers
        CURRENT=$(echo "$PROGRESS" | sed -n 's/Progress: \([0-9]*\).*/\1/p')
        TOTAL=5000
        
        if [ ! -z "$CURRENT" ]; then
            # Calculate percentage
            PERCENT=$(echo "scale=1; $CURRENT * 100 / $TOTAL" | bc)
            
            # Create progress bar
            BAR_LENGTH=50
            FILLED=$(echo "scale=0; $CURRENT * $BAR_LENGTH / $TOTAL" | bc)
            EMPTY=$((BAR_LENGTH - FILLED))
            
            echo "📊 PROGRESS:"
            echo -n "["
            for i in $(seq 1 $FILLED); do echo -n "█"; done
            for i in $(seq 1 $EMPTY); do echo -n "░"; done
            echo "] $PERCENT% ($CURRENT/$TOTAL)"
            echo ""
        fi
    fi
    
    # Show recent log entries
    echo "📝 RECENT ACTIVITY:"
    echo "────────────────────────────────────────────────────────────────────"
    tail -10 "$LOG_FILE" | grep -E "Progress:|Mean diff|ETA|Avg time" | tail -5
    echo ""
    
    # Show memory usage
    MEM=$(ps aux | grep 'qwen.*pipeline' | grep -v grep | awk '{printf "%.1f", $6/1024/1024}' | head -1)
    if [ ! -z "$MEM" ]; then
        echo "💾 Memory Usage: ${MEM} GB"
    fi
    
    # Show process status
    if pgrep -f "qwen.*pipeline" > /dev/null; then
        echo "✅ Status: Pipeline Running"
        PID=$(pgrep -f "qwen.*pipeline" | head -1)
        echo "📍 Process ID: $PID"
    else
        echo "⚠️  Status: Pipeline Not Running"
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "Refreshing every 5 seconds..."
    
    sleep 5
done