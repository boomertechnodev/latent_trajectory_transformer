#!/bin/bash
# Real-time monitoring of hybrid training progress

LOG_FILE="${1:-/tmp/hybrid_training.log}"

echo "=============================================================="
echo "📊 HYBRID TRAINING MONITOR"
echo "=============================================================="
echo ""
echo "Log file: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Extract key metrics
echo "──────────────────────────────────────────────────────────────"
echo "TRAINING SUMMARY"
echo "──────────────────────────────────────────────────────────────"
echo ""

# Check if manual explanations were loaded
MANUAL_COUNT=$(grep "Loaded.*manual explanations" "$LOG_FILE" | grep -o "[0-9]*" | head -1)
if [ -n "$MANUAL_COUNT" ]; then
    echo "✅ Manual explanations loaded: $MANUAL_COUNT"
else
    echo "⚠️  Manual explanations: NOT FOUND in log"
fi

# Get total chunks
CHUNKS=$(grep "Created.*code chunks" "$LOG_FILE" | grep -o "[0-9]*" | head -1)
if [ -n "$CHUNKS" ]; then
    echo "✅ Total chunks indexed: $CHUNKS"
fi

# Get model size
PARAMS=$(grep "parameters" "$LOG_FILE" | grep -o "[0-9,]*" | head -1)
if [ -n "$PARAMS" ]; then
    echo "✅ Model parameters: $PARAMS"
fi

echo ""

# Current epoch and loss
CURRENT_EPOCH=$(grep -o "Epoch [0-9]*/100" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" | head -1)
if [ -n "$CURRENT_EPOCH" ]; then
    PROGRESS=$((CURRENT_EPOCH * 100 / 100))
    echo "📈 Current progress: Epoch $CURRENT_EPOCH/100 ($PROGRESS%)"
else
    echo "⏳ Waiting for training to start..."
    exit 0
fi

# Recent loss values
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "LOSS CURVE (last 20 values)"
echo "──────────────────────────────────────────────────────────────"
echo ""

grep -o "loss=[0-9.]*" "$LOG_FILE" | tail -20 | nl -w2 -s'. ' | awk '{printf "  %-4s %s\n", $1, $2}'

echo ""

# Calculate statistics
echo "──────────────────────────────────────────────────────────────"
echo "STATISTICS"
echo "──────────────────────────────────────────────────────────────"
echo ""

# Get first and last loss
FIRST_LOSS=$(grep -o "loss=[0-9.]*" "$LOG_FILE" | head -1 | cut -d= -f2)
LAST_LOSS=$(grep -o "loss=[0-9.]*" "$LOG_FILE" | tail -1 | cut -d= -f2)

if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
    REDUCTION=$(python3 -c "print(f'{(1 - $LAST_LOSS / $FIRST_LOSS) * 100:.1f}')" 2>/dev/null || echo "N/A")
    echo "  Initial loss: $FIRST_LOSS"
    echo "  Current loss: $LAST_LOSS"
    echo "  Reduction: $REDUCTION%"
fi

echo ""

# Estimate time remaining
if [ "$CURRENT_EPOCH" -gt 0 ]; then
    # Get timestamps
    START_TIME=$(grep "Epoch 1/100" "$LOG_FILE" | head -1 | grep -o "[0-9][0-9]:[0-9][0-9]" | head -1)
    CURRENT_TIME=$(date +%H:%M)

    if [ -n "$START_TIME" ]; then
        # Simple estimate: assume 15 seconds per epoch
        REMAINING_EPOCHS=$((100 - CURRENT_EPOCH))
        REMAINING_MINS=$((REMAINING_EPOCHS * 15 / 60))

        echo "  Estimated time remaining: ~$REMAINING_MINS minutes"
        echo ""
    fi
fi

echo "──────────────────────────────────────────────────────────────"
echo "RECENT ACTIVITY (last 10 lines)"
echo "──────────────────────────────────────────────────────────────"
echo ""
tail -10 "$LOG_FILE" | sed 's/^/  /'
echo ""

echo "=============================================================="
echo ""
echo "💡 Commands:"
echo "  Watch live: tail -f $LOG_FILE"
echo "  Full stats: cat $LOG_FILE | grep 'Epoch [0-9]*/100' | tail -20"
echo "  After training: ./validate_hybrid_training.sh neural_hybrid.index"
echo ""
