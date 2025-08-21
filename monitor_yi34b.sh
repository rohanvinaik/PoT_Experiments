#!/bin/bash

# Monitor Yi-34B test progress without interrupting it

echo "============================================="
echo "  Yi-34B Test Monitor (NO TIMEOUTS)"
echo "============================================="
echo ""

while true; do
    clear
    echo "Yi-34B Validation Progress Monitor"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================="
    
    # Check if process is running
    if ps aux | grep -q "[r]un_yi34b_no_timeout.py"; then
        echo "Status: 🟢 RUNNING"
        
        # Get process info
        ps aux | grep "[r]un_yi34b_no_timeout.py" | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "Time:", $10}'
    else
        echo "Status: 🔴 NOT RUNNING"
    fi
    
    echo ""
    echo "Recent Log Output:"
    echo "---------------------------------------------"
    
    if [ -f "experimental_results/yi34b_no_timeout.log" ]; then
        tail -20 experimental_results/yi34b_no_timeout.log
    else
        echo "Log file not found yet..."
    fi
    
    echo ""
    echo "Memory Usage:"
    echo "---------------------------------------------"
    vm_stat | grep -E "Pages (free|active|inactive|wired)"
    
    echo ""
    echo "CPU Load:"
    echo "---------------------------------------------"
    uptime
    
    echo ""
    echo "Python Processes:"
    echo "---------------------------------------------"
    ps aux | grep python | head -5
    
    echo ""
    echo "Press Ctrl+C to stop monitoring (test continues running)"
    
    sleep 5
done