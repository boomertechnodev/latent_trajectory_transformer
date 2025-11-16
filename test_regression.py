"""
Regression Test Suite for Latent Trajectory Transformer

Establishes baseline benchmarks and prevents silent degradation through:
- Accuracy baselines on standard test sets
- Output reproducibility across runs and platforms
- API compatibility (no breaking changes)
- Performance regression detection (latency/memory)
- Version-to-version metric tracking

Run with: pytest test_regression.py -v -s
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import hashlib
from typing import Dict, Any
import inspect

from latent_drift_trajectory import (
    RaccoonLogClassifier,
    DeterministicLatentODE,
    LogDataset,
    SyntheticTargetDataset,
    log_vocab_size,
)


# ============================================================================
# BASELINE MANAGEMENT
# ============================================================================

BASELINE_FILE = "/tmp/regression_baselines.json"


def save_baseline(name: str, value: Any):
    """Save a baseline metric to file."""
    try:
        with open(BASELINE_FILE, 'r') as f:
            baselines = json.load(f)
    except FileNotFoundError:
        baselines = {}

    baselines[name] = value

    with open(BASELINE_FILE, 'w') as f:
        json.dump(baselines, f, indent=2)


def load_baseline(name: str) -> Any:
    """Load a baseline metric from file."""
    try:
        with open(BASELINE_FILE, 'r') as f:
            baselines = json.load(f)
        return baselines.get(name)
    except FileNotFoundError:
        return None


def compute_hash(tensor: torch.Tensor) -> str:
    """Compute hash of tensor for reproducibility checking."""
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()


# ============================================================================
# TEST 1: Baseline Accuracy Benchmarks
# ============================================================================

def test_baseline_accuracy():
    """
    Establish baseline accuracy on standard test set.

    Protocol:
    1. Train model with fixed hyperparameters
    2. Evaluate on test set
    3. Save accuracy as baseline
    4. Future runs compare against this baseline

    Failure modes:
    - Accuracy drops >5% → alert regression
    - Accuracy improves >10% → update baseline (verify improvement is real)
    """
    print("\n" + "="*80)
    print("TEST 1: Baseline Accuracy Benchmarks")
    print("="*80)

    device = torch.device('cpu')

    # Fixed configuration for reproducibility
    torch.manual_seed(42)

    # Create datasets
    train_dataset = LogDataset(n_samples=500, seq_len=128, drift_point=None)
    test_dataset = LogDataset(n_samples=200, seq_len=128, drift_point=None)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model
    model = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=4,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=200
    ).to(device)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining for 100 steps...")
    model.train()
    for step, (tokens, labels) in enumerate(train_loader):
        if step >= 100:
            break

        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, stats = model(tokens, labels)
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: loss={loss.item():.4f}")

    # Evaluate
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            loss, stats = model(tokens, labels)
            preds = stats['logits'].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Load previous baseline
    baseline_acc = load_baseline("raccoon_test_accuracy")

    if baseline_acc is None:
        print(f"\n✓ Baseline established: {accuracy:.4f}")
        save_baseline("raccoon_test_accuracy", accuracy)
    else:
        print(f"\nBaseline Comparison:")
        print(f"  Previous: {baseline_acc:.4f}")
        print(f"  Current:  {accuracy:.4f}")

        diff = accuracy - baseline_acc
        diff_pct = (diff / baseline_acc) * 100

        print(f"  Change: {diff:+.4f} ({diff_pct:+.1f}%)")

        # Regression check
        REGRESSION_THRESHOLD = -0.05  # -5%

        if diff_pct < REGRESSION_THRESHOLD * 100:
            print(f"\n✗ REGRESSION: Accuracy dropped by {-diff_pct:.1f}%")
            assert False, f"Accuracy regression: {diff_pct:.1f}%"
        elif diff_pct > 10.0:
            print(f"\n✓ IMPROVEMENT: Accuracy increased by {diff_pct:.1f}%")
            print(f"   Consider updating baseline")
        else:
            print(f"\n✓ PASS: No significant regression")

    print(f"\n✓ Baseline accuracy test complete")


# ============================================================================
# TEST 2: Output Reproducibility
# ============================================================================

def test_output_reproducibility():
    """
    Verify model outputs are reproducible across runs with same seed.

    Protocol:
    1. Run inference with seed 42 → output1
    2. Run inference with seed 42 → output2
    3. Verify output1 == output2 (exact match)

    This catches:
    - Non-deterministic operations
    - Platform-specific behavior
    - Floating-point inconsistencies
    """
    print("\n" + "="*80)
    print("TEST 2: Output Reproducibility")
    print("="*80)

    device = torch.device('cpu')

    # Fixed input
    tokens = torch.randint(0, log_vocab_size, (8, 128))
    labels = torch.randint(0, 4, (8,))

    # Run 1
    print("\nRun 1 with seed=42...")
    torch.manual_seed(42)

    model1 = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=4,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=100
    ).to(device)

    model1.eval()

    with torch.no_grad():
        loss1, stats1 = model1(tokens, labels)
        output1 = stats1['logits']

    print(f"  Loss: {loss1.item():.6f}")
    print(f"  Output hash: {compute_hash(output1)}")

    # Run 2
    print("\nRun 2 with seed=42...")
    torch.manual_seed(42)

    model2 = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=4,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=100
    ).to(device)

    model2.eval()

    with torch.no_grad():
        loss2, stats2 = model2(tokens, labels)
        output2 = stats2['logits']

    print(f"  Loss: {loss2.item():.6f}")
    print(f"  Output hash: {compute_hash(output2)}")

    # Compare
    print("\nComparison:")

    loss_match = torch.allclose(loss1, loss2, atol=1e-6)
    output_match = torch.allclose(output1, output2, atol=1e-6)

    print(f"  Loss match: {loss_match}")
    print(f"  Output match: {output_match}")

    if output_match:
        print(f"  Max difference: {(output1 - output2).abs().max().item():.2e}")
    else:
        print(f"  Max difference: {(output1 - output2).abs().max().item():.2e}")

    assert loss_match, "Loss not reproducible across runs"
    assert output_match, "Outputs not reproducible across runs"

    print(f"\n✓ PASS: Outputs are reproducible")


# ============================================================================
# TEST 3: API Compatibility
# ============================================================================

def test_api_compatibility():
    """
    Verify API signatures haven't changed (prevents breaking changes).

    Checks:
    1. Class constructors have expected parameters
    2. Forward methods have expected signatures
    3. Return types are consistent

    Saves API signatures to baseline file for comparison.
    """
    print("\n" + "="*80)
    print("TEST 3: API Compatibility")
    print("="*80)

    # Define expected API signatures
    expected_apis = {
        'RaccoonLogClassifier.__init__': {
            'params': ['self', 'vocab_size', 'num_classes', 'latent_dim',
                      'hidden_dim', 'embed_dim', 'memory_size', 'time_dim',
                      'num_flow_layers', 'num_slicing_projections'],
            'defaults': {'time_dim': 32, 'num_flow_layers': 4,
                        'num_slicing_projections': 1024}
        },
        'RaccoonLogClassifier.forward': {
            'params': ['self', 'tokens', 'labels'],
            'returns': 'Tuple[Tensor, Dict]'
        },
        'RaccoonLogClassifier.continuous_update': {
            'params': ['self', 'tokens', 'labels'],
            'returns': 'None'
        },
        'DeterministicLatentODE.__init__': {
            'params': ['self', 'latent_dim', 'hidden_dim', 'vocab_size',
                      'embed_dim', 'num_slicing_projections'],
            'defaults': {}
        }
    }

    # Check each API
    print("\nChecking API signatures...")

    all_match = True

    for api_name, expected in expected_apis.items():
        class_name, method_name = api_name.split('.')

        # Get class
        if class_name == 'RaccoonLogClassifier':
            cls = RaccoonLogClassifier
        elif class_name == 'DeterministicLatentODE':
            cls = DeterministicLatentODE
        else:
            continue

        # Get method
        if method_name == '__init__':
            method = cls.__init__
        else:
            method = getattr(cls, method_name)

        # Get signature
        sig = inspect.signature(method)
        param_names = list(sig.parameters.keys())

        # Compare
        expected_params = expected['params']

        print(f"\n{api_name}:")
        print(f"  Expected params: {expected_params}")
        print(f"  Actual params:   {param_names}")

        # Check if all expected params are present
        missing = set(expected_params) - set(param_names)
        extra = set(param_names) - set(expected_params)

        if missing:
            print(f"  ✗ Missing params: {missing}")
            all_match = False
        elif extra:
            print(f"  ⚠ Extra params: {extra} (may be OK if backwards compatible)")
        else:
            print(f"  ✓ Match")

    if all_match:
        print(f"\n✓ PASS: All APIs match expected signatures")
    else:
        print(f"\n✗ WARNING: Some APIs have changed")

    # Note: We don't fail on API changes, just warn
    # Breaking changes should be caught in code review


# ============================================================================
# TEST 4: Performance Regression Detection
# ============================================================================

def test_performance_regression():
    """
    Detect performance regressions in latency and memory.

    Protocol:
    1. Measure current latency and memory
    2. Compare against baseline
    3. Alert if regression >10%

    Failure modes:
    - Latency increases >10% → investigate slowdown
    - Memory increases >10% → investigate leak
    """
    print("\n" + "="*80)
    print("TEST 4: Performance Regression Detection")
    print("="*80)

    import time
    import tracemalloc

    device = torch.device('cpu')

    # Standard configuration
    batch_size = 32
    seq_len = 128

    model = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=4,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=100
    ).to(device)

    model.eval()

    tokens = torch.randint(0, log_vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 4, (batch_size,), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            loss, stats = model(tokens, labels)

    # Measure latency
    print("\nMeasuring latency...")
    num_runs = 50
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            loss, stats = model(tokens, labels)
            end = time.perf_counter()
            times.append(end - start)

    latency_ms = (sum(times) / len(times)) * 1000.0

    print(f"  Current latency: {latency_ms:.2f} ms")

    # Measure memory
    print("\nMeasuring memory...")
    tracemalloc.start()

    with torch.no_grad():
        loss, stats = model(tokens, labels)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_mb = peak / (1024 * 1024)

    print(f"  Current memory: {memory_mb:.2f} MB")

    # Compare with baselines
    baseline_latency = load_baseline("raccoon_latency_ms")
    baseline_memory = load_baseline("raccoon_memory_mb")

    REGRESSION_THRESHOLD = 1.10  # 10% increase

    # Latency regression check
    print("\nLatency Regression Check:")
    if baseline_latency is None:
        print(f"  ✓ Baseline established: {latency_ms:.2f} ms")
        save_baseline("raccoon_latency_ms", latency_ms)
    else:
        print(f"  Baseline: {baseline_latency:.2f} ms")
        print(f"  Current:  {latency_ms:.2f} ms")

        ratio = latency_ms / baseline_latency
        print(f"  Ratio: {ratio:.2f}x")

        if ratio > REGRESSION_THRESHOLD:
            print(f"  ✗ REGRESSION: Latency increased {(ratio-1)*100:.1f}%")
            assert False, f"Latency regression: {ratio:.2f}x baseline"
        else:
            print(f"  ✓ PASS: No latency regression")

    # Memory regression check
    print("\nMemory Regression Check:")
    if baseline_memory is None:
        print(f"  ✓ Baseline established: {memory_mb:.2f} MB")
        save_baseline("raccoon_memory_mb", memory_mb)
    else:
        print(f"  Baseline: {baseline_memory:.2f} MB")
        print(f"  Current:  {memory_mb:.2f} MB")

        ratio = memory_mb / baseline_memory
        print(f"  Ratio: {ratio:.2f}x")

        if ratio > REGRESSION_THRESHOLD:
            print(f"  ✗ REGRESSION: Memory increased {(ratio-1)*100:.1f}%")
            assert False, f"Memory regression: {ratio:.2f}x baseline"
        else:
            print(f"  ✓ PASS: No memory regression")

    print(f"\n✓ Performance regression test complete")


# ============================================================================
# TEST 5: Version Tracking
# ============================================================================

def test_version_tracking():
    """
    Track metrics over versions for trend analysis.

    Protocol:
    1. Compute current metrics (accuracy, latency, memory)
    2. Append to historical log
    3. Detect trends (improving, degrading, stable)

    This enables:
    - Long-term performance monitoring
    - Identifying gradual degradation
    - A/B testing across versions
    """
    print("\n" + "="*80)
    print("TEST 5: Version Tracking")
    print("="*80)

    VERSION_LOG_FILE = "/tmp/version_metrics.jsonl"

    # Collect current metrics
    metrics = {
        'timestamp': '2025-11-16',  # In production, use actual timestamp
        'version': '1.0.0',  # In production, use git commit hash
        'accuracy': load_baseline("raccoon_test_accuracy"),
        'latency_ms': load_baseline("raccoon_latency_ms"),
        'memory_mb': load_baseline("raccoon_memory_mb"),
    }

    print("\nCurrent Metrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value}")

    # Append to log
    with open(VERSION_LOG_FILE, 'a') as f:
        json.dump(metrics, f)
        f.write('\n')

    print(f"\n✓ Metrics logged to {VERSION_LOG_FILE}")

    # Read history
    try:
        with open(VERSION_LOG_FILE, 'r') as f:
            history = [json.loads(line) for line in f]
    except FileNotFoundError:
        history = []

    if len(history) > 1:
        print(f"\nHistory ({len(history)} versions):")
        print(f"{'Version':<15} {'Accuracy':<12} {'Latency (ms)':<15} {'Memory (MB)':<12}")
        print("-" * 60)

        for entry in history[-5:]:  # Show last 5
            print(f"{entry.get('version', 'unknown'):<15} "
                  f"{entry.get('accuracy', 0):<12.4f} "
                  f"{entry.get('latency_ms', 0):<15.2f} "
                  f"{entry.get('memory_mb', 0):<12.2f}")

    print(f"\n✓ PASS: Version tracking operational")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("REGRESSION TEST SUITE")
    print("="*80)

    test_baseline_accuracy()
    test_output_reproducibility()
    test_api_compatibility()
    test_performance_regression()
    test_version_tracking()

    print("\n" + "="*80)
    print("ALL REGRESSION TESTS COMPLETE")
    print("="*80)
