#!/bin/bash
# Quick test script - Run a single neural search query to validate training worked

INDEX="neural_intensive.index"

if [ ! -f "$INDEX" ]; then
    echo "❌ Index not found: $INDEX"
    echo "   Training may still be running..."
    exit 1
fi

echo "=============================================================="
echo "QUICK NEURAL SEARCH TEST"
echo "=============================================================="
echo ""

echo "Query: 'Where is the SDE dynamics implemented?'"
echo ""

python neural_code_search.py query "where is the SDE dynamics implemented?" \
    --index "$INDEX" \
    --top-k 3

echo ""
echo "=============================================================="
echo "If explanations are coherent (not gibberish), training worked!"
echo "If still gibberish, need more epochs or different approach."
echo "=============================================================="
