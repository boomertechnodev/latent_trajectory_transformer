"""
Baseline Performance Benchmark for Latent Trajectory Transformer
================================================================

Measures current performance on CPU before optimizations:
1. Inference latency (time per forward pass)
2. Training speed (steps per second)
3. Memory usage (peak RAM)
4. Sample quality (perplexity, if applicable)

This establishes quantitative baseline for comparison after:
- Flow matching integration
- CPU optimizations (BFloat16, Flash Attention, INT8)
- Application development
"""

import time
import torch
import numpy as np
from latent_drift_trajectory import (
    LogDataset, RaccoonLogClassifier, train_raccoon_classifier,
    log_vocab_size, NUM_LOG_CLASSES, decode_log, LOG_CATEGORIES
)
from torch.utils.data import DataLoader
import tracemalloc


def measure_inference_latency(model, test_tokens, n_runs=100):
    """
    Measure latency for a single forward pass.

    Returns:
        dict: {mean_ms, p50_ms, p95_ms, p99_ms}
    """
    model.eval()
    latencies = []

    for _ in range(n_runs):
        tokens = test_tokens[np.random.randint(len(test_tokens))]
        tokens = tokens.unsqueeze(0)  # Add batch dimension

        start = time.perf_counter()
        with torch.no_grad():
            mean, logvar = model.encode(tokens)
            z = model.sample_latent(mean, logvar)
            logits = model.classifier(z)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'throughput_per_sec': 1000 / np.mean(latencies)
    }


def measure_training_speed(model, train_loader, device, n_steps=50):
    """
    Measure training speed (steps per second).

    Returns:
        float: steps per second
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    start = time.perf_counter()

    dataiter = iter(train_loader)
    for step in range(n_steps):
        try:
            tokens, labels = next(dataiter)
        except StopIteration:
            dataiter = iter(train_loader)
            tokens, labels = next(dataiter)

        tokens = tokens.to(device)
        labels = labels.to(device)

        # Forward pass (returns loss, stats tuple)
        total_loss, stats = model(tokens, labels)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    end = time.perf_counter()
    elapsed = end - start

    return n_steps / elapsed


def measure_memory_usage(model, test_tokens, device):
    """
    Measure peak memory usage during inference.

    Returns:
        dict: {peak_mb, current_mb}
    """
    tracemalloc.start()

    model.eval()
    with torch.no_grad():
        for tokens in test_tokens[:10]:  # Run 10 samples
            tokens = tokens.unsqueeze(0).to(device)
            mean, logvar = model.encode(tokens)
            z = model.sample_latent(mean, logvar)
            logits = model.classifier(z)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'peak_mb': peak / (1024 * 1024),
        'current_mb': current / (1024 * 1024)
    }


def profile_components(model, test_tokens):
    """
    Profile individual components to identify bottlenecks.

    Returns:
        dict: {component_name: time_ms}
    """
    model.eval()
    tokens = test_tokens[0].unsqueeze(0)

    timings = {}

    # Encoder
    start = time.perf_counter()
    with torch.no_grad():
        mean, logvar = model.encode(tokens)
    timings['encoder_ms'] = (time.perf_counter() - start) * 1000

    # Sampling
    start = time.perf_counter()
    with torch.no_grad():
        z = model.sample_latent(mean, logvar)
    timings['sampling_ms'] = (time.perf_counter() - start) * 1000

    # SDE trajectory (if enabled)
    from latent_drift_trajectory import solve_sde
    start = time.perf_counter()
    with torch.no_grad():
        t_span = torch.linspace(0.0, 0.1, 3, device=tokens.device)
        z_traj = solve_sde(model.dynamics, z, t_span)
    timings['sde_solve_ms'] = (time.perf_counter() - start) * 1000

    # Normalizing flow
    start = time.perf_counter()
    with torch.no_grad():
        z_final = z_traj[:, -1, :]
        t_flow = torch.tensor([0.5], device=tokens.device)
        z_flowed, _ = model.flow(z_final, t_flow)
    timings['flow_ms'] = (time.perf_counter() - start) * 1000

    # Classifier
    start = time.perf_counter()
    with torch.no_grad():
        logits = model.classifier(z_flowed)
    timings['classifier_ms'] = (time.perf_counter() - start) * 1000

    # Total
    timings['total_ms'] = sum(timings.values())

    return timings


def main():
    print("=" * 80)
    print("BASELINE PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()

    # Force CPU (matching latent_drift_trajectory.py line 1670)
    device = torch.device('cpu')
    print(f"Device: {device}")
    print()

    # Create small datasets for benchmarking
    print("📦 Loading datasets...")
    train_ds = LogDataset(n_samples=500, seq_len=50)  # Smaller for faster benchmarking
    test_ds = LogDataset(n_samples=100, seq_len=50, drift_point=None)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Collect test tokens for inference benchmarking
    test_tokens = [test_ds[i][0] for i in range(len(test_ds))]
    test_tokens_tensor = torch.stack(test_tokens)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print()

    # Create model
    print("🦝 Initializing Raccoon model...")
    model = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=NUM_LOG_CLASSES,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=2000,
        adaptation_rate=1e-4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print()

    # Benchmark 1: Inference Latency
    print("⏱️  BENCHMARK 1: Inference Latency")
    print("-" * 80)
    latency_stats = measure_inference_latency(model, test_tokens_tensor, n_runs=100)
    print(f"  Mean latency:    {latency_stats['mean_ms']:.2f} ms")
    print(f"  P50 latency:     {latency_stats['p50_ms']:.2f} ms")
    print(f"  P95 latency:     {latency_stats['p95_ms']:.2f} ms")
    print(f"  P99 latency:     {latency_stats['p99_ms']:.2f} ms")
    print(f"  Throughput:      {latency_stats['throughput_per_sec']:.1f} samples/sec")
    print()

    target_latency = 100  # ms
    if latency_stats['mean_ms'] > target_latency:
        speedup_needed = latency_stats['mean_ms'] / target_latency
        print(f"  ⚠️  Need {speedup_needed:.1f}x speedup to reach <{target_latency}ms target")
    else:
        print(f"  ✅ Already below {target_latency}ms target!")
    print()

    # Benchmark 2: Training Speed
    print("🏃 BENCHMARK 2: Training Speed")
    print("-" * 80)
    steps_per_sec = measure_training_speed(model, train_loader, device, n_steps=50)
    print(f"  Training speed:  {steps_per_sec:.2f} steps/sec")
    print(f"  Time per step:   {1000/steps_per_sec:.2f} ms")
    print()

    # Benchmark 3: Memory Usage
    print("💾 BENCHMARK 3: Memory Usage")
    print("-" * 80)
    memory_stats = measure_memory_usage(model, test_tokens_tensor, device)
    print(f"  Peak memory:     {memory_stats['peak_mb']:.1f} MB")
    print(f"  Current memory:  {memory_stats['current_mb']:.1f} MB")
    print()

    target_memory = 2048  # MB (2GB)
    if memory_stats['peak_mb'] > target_memory:
        reduction_needed = memory_stats['peak_mb'] / target_memory
        print(f"  ⚠️  Need {reduction_needed:.1f}x memory reduction to reach <{target_memory}MB target")
    else:
        print(f"  ✅ Already below {target_memory}MB target!")
    print()

    # Benchmark 4: Component Profiling
    print("🔍 BENCHMARK 4: Component Profiling")
    print("-" * 80)
    component_times = profile_components(model, test_tokens_tensor)

    # Sort by time (descending)
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)

    for component, time_ms in sorted_components:
        if component != 'total_ms':
            percentage = (time_ms / component_times['total_ms']) * 100
            print(f"  {component:20s} {time_ms:8.3f} ms ({percentage:5.1f}%)")

    print(f"  {'─' * 20} {'─' * 8}    {'─' * 6}")
    print(f"  {'TOTAL':20s} {component_times['total_ms']:8.3f} ms (100.0%)")
    print()

    # Identify bottleneck
    bottleneck = sorted_components[0][0]
    bottleneck_pct = (sorted_components[0][1] / component_times['total_ms']) * 100
    print(f"  🎯 Bottleneck: {bottleneck} ({bottleneck_pct:.1f}% of total time)")
    print()

    # Summary
    print("=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)
    print(f"  Inference latency:  {latency_stats['mean_ms']:.2f} ms (target: <100ms)")
    print(f"  Training speed:     {steps_per_sec:.2f} steps/sec")
    print(f"  Peak memory:        {memory_stats['peak_mb']:.1f} MB (target: <2048MB)")
    print(f"  Model parameters:   {param_count:,}")
    print(f"  Bottleneck:         {bottleneck}")
    print()

    # Optimization recommendations
    print("🚀 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 80)
    print("  1. Flow Matching:         Replace ODE with rectified flows (2-3x faster)")
    print("  2. BFloat16:              Mixed precision training (1.8x faster, 2x memory)")
    print("  3. Flash Attention:       Memory-efficient attention (1.5x faster)")
    print("  4. Gradient Checkpointing: Reduce memory (2x), enables larger batches")
    print("  5. INT8 Quantization:     Inference only (2-4x faster, 4x smaller)")
    print()
    print("  Combined expected speedup: ~10x (1000ms → <100ms)")
    print("=" * 80)

    return {
        'latency_stats': latency_stats,
        'training_speed': steps_per_sec,
        'memory_stats': memory_stats,
        'component_times': component_times,
        'param_count': param_count
    }


if __name__ == '__main__':
    results = main()
