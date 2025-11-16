#!/bin/bash
# Monitor neural search training every 10 minutes for 30 minutes total

LOG_FILE="/tmp/neural_intensive_train.log"
CHECK_INTERVAL=600  # 10 minutes in seconds
TOTAL_DURATION=1800  # 30 minutes in seconds

echo "==================================================================="
echo "NEURAL SEARCH TRAINING MONITOR"
echo "==================================================================="
echo "Start time: $(date)"
echo "Will check every 10 minutes for 30 minutes"
echo "Log file: $LOG_FILE"
echo ""

START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $TOTAL_DURATION ]; then
        echo "==================================================================="
        echo "30 MINUTES ELAPSED - Training should be complete"
        echo "==================================================================="
        echo "Final log tail:"
        tail -50 "$LOG_FILE"
        break
    fi

    echo "-------------------------------------------------------------------"
    echo "Check at: $(date) (Elapsed: $((ELAPSED / 60)) min)"
    echo "-------------------------------------------------------------------"

    # Show last 15 lines of training progress
    if [ -f "$LOG_FILE" ]; then
        echo "Latest training status:"
        tail -15 "$LOG_FILE" | grep -E "Epoch|loss=|✓|Building"
        echo ""
    else
        echo "Log file not found yet..."
        echo ""
    fi

    # Sleep for 10 minutes
    sleep $CHECK_INTERVAL
done
