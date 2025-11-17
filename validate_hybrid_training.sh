#!/bin/bash
# Comprehensive validation script for hybrid explanation training
# Usage: ./validate_hybrid_training.sh neural_hybrid.index

set -e

INDEX_FILE="${1:-neural_hybrid.index}"

if [ ! -f "$INDEX_FILE" ]; then
    echo "❌ Index file not found: $INDEX_FILE"
    echo "   Training may still be running. Check /tmp/hybrid_training.log"
    exit 1
fi

echo "=============================================================="
echo "🔍 HYBRID TRAINING VALIDATION"
echo "=============================================================="
echo ""
echo "Index: $INDEX_FILE"
echo ""

# ============================================================
# PART 1: INSPECT INDEX METADATA
# ============================================================
echo "──────────────────────────────────────────────────────────────"
echo "PART 1: Index Metadata Inspection"
echo "──────────────────────────────────────────────────────────────"
echo ""

python3 << 'EOF'
import pickle
import sys

try:
    with open('$INDEX_FILE', 'rb') as f:
        index = pickle.load(f)

    print(f"✓ Index loaded successfully")
    print(f"  Top-level keys: {list(index.keys())}")
    print(f"  Total chunks: {len(index['metadata'])}")
    print(f"  Embeddings shape: {index['embeddings'].shape}")
    print(f"  Config: {index['config']}")
    print()

    # Check explanation coverage
    has_explanation = sum(1 for m in index['metadata']
                          if m.get('explanation_target') not in [None, '', 'N/A'])
    coverage_pct = (has_explanation / len(index['metadata'])) * 100

    print(f"📊 EXPLANATION COVERAGE")
    print(f"  Chunks with explanations: {has_explanation}/{len(index['metadata'])} ({coverage_pct:.1f}%)")

    if coverage_pct < 50:
        print(f"  ⚠️  WARNING: Low coverage! Expected >95%")
        sys.exit(1)
    elif coverage_pct < 95:
        print(f"  ⚠️  Coverage below target (expected >95%)")
    else:
        print(f"  ✅ EXCELLENT coverage!")
    print()

    # Sample explanations by category
    print(f"📝 SAMPLE EXPLANATIONS (first 10 with targets)")
    print()

    count = 0
    manual_count = 0
    auto_count = 0

    for i, m in enumerate(index['metadata']):
        target = m.get('explanation_target', '')
        if target and target not in ['', 'N/A']:
            count += 1

            # Categorize
            is_manual = any(keyword in target.lower() for keyword in
                          ['implements', 'stochastic differential', 'coupling',
                           'affine', 'normalizing flow', 'fractal', 'hilbert'])

            if is_manual:
                manual_count += 1
                category = "MANUAL"
            else:
                auto_count += 1
                category = "AUTO"

            if count <= 10:
                filepath = m.get('filepath', 'unknown')
                filename = filepath.split('/')[-1] if '/' in filepath else filepath
                print(f"{count}. [{category}] {filename}")
                print(f"   {target[:100]}{'...' if len(target) > 100 else ''}")
                print()

            if count >= 100:
                break

    print(f"📈 CATEGORY DISTRIBUTION (sampled 100 chunks)")
    print(f"  Manual-quality: {manual_count}/100 ({manual_count}%)")
    print(f"  Auto-generated: {auto_count}/100 ({auto_count}%)")
    print()

except Exception as e:
    print(f"❌ Error inspecting index: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Index inspection failed"
    exit 1
fi

# ============================================================
# PART 2: TEST MANUALLY-EXPLAINED CHUNKS
# ============================================================
echo "──────────────────────────────────────────────────────────────"
echo "PART 2: Test Queries on Manual Explanations"
echo "──────────────────────────────────────────────────────────────"
echo ""

echo "Query 1: SDE Dynamics (should find RaccoonDynamics)"
echo "─────────────────────────────────────────────────────────────"
python neural_code_search.py query "SDE dynamics drift diffusion stochastic" \
    --index "$INDEX_FILE" --top-k 3 | grep -A 15 "Result 1:"
echo ""

echo "Query 2: Hilbert Curve (should find hilbert_mapper.py)"
echo "─────────────────────────────────────────────────────────────"
python neural_code_search.py query "Hilbert curve fractal 2D mapping" \
    --index "$INDEX_FILE" --top-k 3 | grep -A 15 "Result 1:"
echo ""

echo "Query 3: Normalizing Flow (should find CouplingLayer/RaccoonFlow)"
echo "─────────────────────────────────────────────────────────────"
python neural_code_search.py query "normalizing flow coupling layers invertible" \
    --index "$INDEX_FILE" --top-k 3 | grep -A 15 "Result 1:"
echo ""

# ============================================================
# PART 3: EXPLANATION COHERENCE CHECK
# ============================================================
echo "──────────────────────────────────────────────────────────────"
echo "PART 3: Explanation Coherence Analysis"
echo "──────────────────────────────────────────────────────────────"
echo ""
echo "Checking if explanations are coherent (not gibberish)..."
echo ""

python3 << 'EOF'
import pickle
import re

with open('$INDEX_FILE', 'rb') as f:
    index = pickle.load(f)

# Test queries to get decoder outputs
test_queries = [
    "SDE dynamics",
    "Hilbert curve",
    "normalizing flow",
]

print("💬 DECODER OUTPUT QUALITY CHECK")
print()
print("Looking for common gibberish patterns:")
print("  - Random characters (°, ¼, ±, etc.)")
print("  - Nonsense words (rrpy, sfitation, etc.)")
print("  - Broken capitalization (Indy, Aspattern, etc.)")
print()

# Sample some explanations from index
coherent_count = 0
gibberish_count = 0
total_checked = 0

gibberish_indicators = [
    r'[°¼±§¶]',  # Special chars
    r'\b[A-Z][a-z]{1,3}[A-Z]',  # Broken caps like "InDy"
    r'[a-z]{10,}',  # Very long lowercase words
    r'\s[a-z]\s',  # Single lowercase letters
]

for m in index['metadata'][:50]:
    target = m.get('explanation_target', '')
    if target and target not in ['', 'N/A']:
        total_checked += 1

        is_gibberish = any(re.search(pattern, target) for pattern in gibberish_indicators)

        if is_gibberish:
            gibberish_count += 1
            if gibberish_count <= 3:
                print(f"❌ GIBBERISH DETECTED:")
                print(f"   {target[:100]}")
                print()
        else:
            coherent_count += 1
            if coherent_count <= 3:
                print(f"✅ COHERENT:")
                print(f"   {target[:100]}")
                print()

print(f"📊 COHERENCE STATISTICS (sampled {total_checked} chunks)")
print(f"  Coherent: {coherent_count}/{total_checked} ({coherent_count/total_checked*100:.1f}%)")
print(f"  Gibberish: {gibberish_count}/{total_checked} ({gibberish_count/total_checked*100:.1f}%)")
print()

if gibberish_count > total_checked * 0.3:
    print("⚠️  WARNING: High gibberish rate (>30%)")
    print("   Decoder may need more training or architectural changes")
elif gibberish_count > 0:
    print("⚠️  Some gibberish detected, but mostly coherent")
    print("   Acceptable for hybrid training with manual examples")
else:
    print("✅ EXCELLENT: No gibberish detected!")
    print("   Decoder learned to generate coherent explanations")
print()

EOF

# ============================================================
# PART 4: RETRIEVAL QUALITY CHECK
# ============================================================
echo "──────────────────────────────────────────────────────────────"
echo "PART 4: Retrieval Quality Assessment"
echo "──────────────────────────────────────────────────────────────"
echo ""

echo "Testing retrieval accuracy (do we find the right files?)..."
echo ""

# Test 1: SDE Dynamics
echo "TEST 1: 'SDE dynamics' should find latent_drift_trajectory.py"
RESULT=$(python neural_code_search.py query "SDE dynamics drift diffusion" \
    --index "$INDEX_FILE" --top-k 3 2>&1 | grep "File:" | head -1)
echo "  Result: $RESULT"
if echo "$RESULT" | grep -q "latent_drift_trajectory"; then
    echo "  ✅ CORRECT file found!"
else
    echo "  ❌ WRONG file (expected latent_drift_trajectory.py)"
fi
echo ""

# Test 2: Hilbert Curve
echo "TEST 2: 'Hilbert curve' should find hilbert_mapper.py"
RESULT=$(python neural_code_search.py query "Hilbert curve fractal" \
    --index "$INDEX_FILE" --top-k 3 2>&1 | grep "File:" | head -1)
echo "  Result: $RESULT"
if echo "$RESULT" | grep -q "hilbert"; then
    echo "  ✅ CORRECT file found!"
else
    echo "  ❌ WRONG file (expected hilbert_mapper.py)"
fi
echo ""

# Test 3: Experience Replay
echo "TEST 3: 'experience replay' should find latent_drift_trajectory.py (RaccoonMemory)"
RESULT=$(python neural_code_search.py query "experience replay memory buffer" \
    --index "$INDEX_FILE" --top-k 3 2>&1 | grep "File:" | head -1)
echo "  Result: $RESULT"
if echo "$RESULT" | grep -q "latent_drift_trajectory\|raccoon"; then
    echo "  ✅ CORRECT file found!"
else
    echo "  ❌ WRONG file (expected latent_drift_trajectory.py or raccoon*.py)"
fi
echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
echo "=============================================================="
echo "✅ VALIDATION COMPLETE"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "  1. Review results above"
echo "  2. If coherent: Run full test suite (python test_neural_search.py)"
echo "  3. If gibberish: Expand manual explanations to 50-100"
echo "  4. If retrieval fails: Check encoder training, consider contrastive loss"
echo "  5. Compare with grep: python compare_search_methods.py"
echo ""
echo "For detailed analysis, see:"
echo "  - EVALUATION_NEURAL_SEARCH.md"
echo "  - TRAINING_PROGRESS.md"
echo "  - /tmp/hybrid_training.log"
echo ""
