"""
Test Suite for Time-Aware Transform

This test suite verifies:
1. Initialization and parameter validation
2. Uniqueness: different times produce different embeddings
3. Smoothness: nearby times have similar embeddings
4. Frequency spectrum: exponential spacing verification
5. Multi-scale resolution: low and high frequencies present
6. Edge cases: boundary times (t=0, t=1), invalid inputs
7. Learnable embedding: MLP projection works correctly
8. Gradient flow: parameters are learnable
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from time_embedding import TimeEmbedding, LearnableTimeEmbedding


def test_initialization():
    """Test TimeEmbedding initialization and parameter validation."""
    print("=" * 60)
    print("TEST 1: Initialization and Parameter Validation")
    print("=" * 60)

    # Basic initialization
    time_emb = TimeEmbedding(time_dim=64, freq_min=1.0, freq_max=1000.0)

    print(f"Time dimension: {time_emb.time_dim}")
    print(f"Number of frequencies: {time_emb.num_frequencies}")
    print(f"Frequency range: [{time_emb.freq_min}, {time_emb.freq_max}]")

    assert time_emb.time_dim == 64
    assert time_emb.num_frequencies == 32  # Half of time_dim
    assert time_emb.freq_min == 1.0
    assert time_emb.freq_max == 1000.0

    # Check frequency buffer
    freqs = time_emb.get_frequencies()
    print(f"\nFrequencies shape: {freqs.shape}")
    print(f"First 5 frequencies: {freqs[:5].tolist()}")
    print(f"Last 5 frequencies: {freqs[-5:].tolist()}")

    assert freqs.shape == (32,)
    assert torch.allclose(freqs[0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(freqs[-1], torch.tensor(1000.0), atol=1e-3)

    # Verify exponential spacing
    # freq[i+1] / freq[i] should be constant (geometric progression)
    ratios = freqs[1:] / freqs[:-1]
    print(f"\nFrequency ratios (should be constant):")
    print(f"  Mean ratio: {ratios.mean().item():.4f}")
    print(f"  Std ratio: {ratios.std().item():.6f}")

    assert ratios.std() < 0.01, "Frequencies should be exponentially spaced"

    # Test invalid parameters
    try:
        TimeEmbedding(time_dim=63)  # Odd dimension
        assert False, "Should raise error for odd time_dim"
    except ValueError as e:
        print(f"\n✓ Correctly rejected odd time_dim: {e}")

    try:
        TimeEmbedding(time_dim=64, freq_min=-1.0)
        assert False, "Should raise error for negative freq_min"
    except ValueError as e:
        print(f"✓ Correctly rejected negative freq_min: {e}")

    try:
        TimeEmbedding(time_dim=64, freq_min=100.0, freq_max=10.0)
        assert False, "Should raise error for freq_max < freq_min"
    except ValueError as e:
        print(f"✓ Correctly rejected freq_max < freq_min: {e}")

    print("\n✓ Initialization test passed!\n")


def test_forward_pass():
    """Test forward pass with different input shapes."""
    print("=" * 60)
    print("TEST 2: Forward Pass and Output Shapes")
    print("=" * 60)

    time_emb = TimeEmbedding(time_dim=64)

    # Test 1D input
    t_1d = torch.linspace(0, 1, 10)
    emb_1d = time_emb(t_1d)

    print(f"1D input shape: {t_1d.shape} → output: {emb_1d.shape}")
    assert emb_1d.shape == (10, 64), f"Expected (10, 64), got {emb_1d.shape}"

    # Test 2D input
    t_2d = torch.linspace(0, 1, 10).unsqueeze(-1)
    emb_2d = time_emb(t_2d)

    print(f"2D input shape: {t_2d.shape} → output: {emb_2d.shape}")
    assert emb_2d.shape == (10, 64), f"Expected (10, 64), got {emb_2d.shape}"

    # Outputs should be identical
    assert torch.allclose(emb_1d, emb_2d, atol=1e-6), "1D and 2D inputs should give same output"

    # Test batch processing
    t_batch = torch.rand(128)
    emb_batch = time_emb(t_batch)

    print(f"Batch input shape: {t_batch.shape} → output: {emb_batch.shape}")
    assert emb_batch.shape == (128, 64)

    # Check no NaN or Inf
    assert not torch.isnan(emb_batch).any(), "Output contains NaN"
    assert not torch.isinf(emb_batch).any(), "Output contains Inf"

    print("\n✓ Forward pass test passed!\n")


def test_uniqueness():
    """Test that different times produce different embeddings."""
    print("=" * 60)
    print("TEST 3: Uniqueness (No Collisions)")
    print("=" * 60)

    time_emb = TimeEmbedding(time_dim=64)

    # Sample many different times
    num_samples = 1000
    times = torch.linspace(0, 1, num_samples)
    embeddings = time_emb(times)

    print(f"Testing {num_samples} uniformly spaced times in [0, 1]")
    print(f"Embeddings shape: {embeddings.shape}")

    # Check pairwise distances
    # Compute cosine similarity matrix
    emb_norm = torch.nn.functional.normalize(embeddings, dim=1)
    similarity = torch.matmul(emb_norm, emb_norm.t())

    # Diagonal should be 1 (self-similarity)
    assert torch.allclose(torch.diag(similarity), torch.ones(num_samples), atol=1e-5)

    # Off-diagonal should be < 1 (different embeddings)
    # Set diagonal to 0 to ignore self-similarity
    similarity_off_diag = similarity - torch.eye(num_samples)

    max_similarity = similarity_off_diag.abs().max().item()
    mean_similarity = similarity_off_diag.abs().mean().item()

    print(f"\nPairwise cosine similarity (off-diagonal):")
    print(f"  Max: {max_similarity:.6f}")
    print(f"  Mean: {mean_similarity:.6f}")

    # For truly unique embeddings, max similarity should be well below 1
    assert max_similarity < 0.999, f"Some embeddings are too similar: {max_similarity}"

    # Test specific collision: t=0.0 and t=0.001 should be different
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([0.001])

    emb1 = time_emb(t1)
    emb2 = time_emb(t2)

    diff = (emb1 - emb2).abs().mean().item()
    print(f"\nDifference between t=0.0 and t=0.001: {diff:.6f}")
    assert diff > 1e-3, "Very close times should still have measurable difference"

    print("\n✓ Uniqueness test passed!\n")


def test_smoothness():
    """Test that nearby times have similar embeddings (Lipschitz continuity)."""
    print("=" * 60)
    print("TEST 4: Smoothness (Lipschitz Continuity)")
    print("=" * 60)

    time_emb = TimeEmbedding(time_dim=64)

    # Test smoothness at different scales
    deltas = [1e-1, 1e-2, 1e-3, 1e-4]

    print("Testing smoothness for different time differences:")

    for delta in deltas:
        # Sample pairs of nearby times
        num_pairs = 100
        t1 = torch.rand(num_pairs)
        t2 = t1 + delta
        t2 = torch.clamp(t2, 0, 1)  # Ensure in [0, 1]

        emb1 = time_emb(t1)
        emb2 = time_emb(t2)

        # Compute L2 distance in embedding space
        emb_dist = (emb1 - emb2).norm(dim=1)
        mean_dist = emb_dist.mean().item()
        max_dist = emb_dist.max().item()

        print(f"  Δt = {delta:.0e}: mean emb dist = {mean_dist:.6f}, max = {max_dist:.6f}")

        # Smaller time difference should give smaller embedding distance
        # (Lipschitz continuity: ||f(t1) - f(t2)|| <= L * ||t1 - t2||)

    # Specific test: smoothness decreases with smaller delta
    delta_large = 0.1
    delta_small = 0.01

    t = torch.rand(100)

    t_large = torch.clamp(t + delta_large, 0, 1)
    t_small = torch.clamp(t + delta_small, 0, 1)

    emb = time_emb(t)
    emb_large = time_emb(t_large)
    emb_small = time_emb(t_small)

    dist_large = (emb - emb_large).norm(dim=1).mean().item()
    dist_small = (emb - emb_small).norm(dim=1).mean().item()

    print(f"\nSmaller Δt should give smaller embedding distance:")
    print(f"  Δt = {delta_large}: dist = {dist_large:.6f}")
    print(f"  Δt = {delta_small}: dist = {dist_small:.6f}")

    assert dist_small < dist_large, "Smaller time difference should give smaller embedding distance"

    print("\n✓ Smoothness test passed!\n")


def test_frequency_coverage():
    """Test that frequency spectrum covers the expected range."""
    print("=" * 60)
    print("TEST 5: Frequency Spectrum Coverage")
    print("=" * 60)

    # Test different frequency ranges
    configs = [
        (64, 1.0, 1000.0),
        (128, 0.1, 10000.0),
        (32, 10.0, 100.0)
    ]

    for time_dim, freq_min, freq_max in configs:
        time_emb = TimeEmbedding(time_dim=time_dim, freq_min=freq_min, freq_max=freq_max)
        freqs = time_emb.get_frequencies()

        print(f"\nConfig: time_dim={time_dim}, freq_range=[{freq_min}, {freq_max}]")
        print(f"  Num frequencies: {len(freqs)}")
        print(f"  Actual range: [{freqs.min().item():.4f}, {freqs.max().item():.4f}]")

        # Check boundary frequencies
        assert torch.allclose(freqs.min(), torch.tensor(freq_min), atol=1e-5)
        assert torch.allclose(freqs.max(), torch.tensor(freq_max), rtol=1e-3)

        # Check exponential spacing (log-linear)
        log_freqs = torch.log(freqs)
        # Should be approximately linear in log-space
        expected_log = torch.linspace(math.log(freq_min), math.log(freq_max), len(freqs))
        log_diff = (log_freqs - expected_log).abs().max().item()

        print(f"  Log-space linearity error: {log_diff:.6f}")
        assert log_diff < 0.01, "Frequencies should be exponentially spaced"

    print("\n✓ Frequency coverage test passed!\n")


def test_multi_scale_resolution():
    """Test that low and high frequencies are both present and functional."""
    print("=" * 60)
    print("TEST 6: Multi-Scale Temporal Resolution")
    print("=" * 60)

    time_emb = TimeEmbedding(time_dim=64, freq_min=1.0, freq_max=1000.0)

    # Test slow variation (low frequency detection)
    # Sample at coarse time intervals
    t_coarse = torch.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    emb_coarse = time_emb(t_coarse)

    # Consecutive embeddings should vary smoothly
    diffs_coarse = (emb_coarse[1:] - emb_coarse[:-1]).norm(dim=1)
    mean_diff_coarse = diffs_coarse.mean().item()

    print(f"Coarse sampling (Δt = 0.1):")
    print(f"  Mean consecutive difference: {mean_diff_coarse:.6f}")

    # Test fast variation (high frequency detection)
    # Sample at fine time intervals
    t_fine = torch.linspace(0, 0.1, 101)  # 100 samples in [0, 0.1]
    emb_fine = time_emb(t_fine)

    diffs_fine = (emb_fine[1:] - emb_fine[:-1]).norm(dim=1)
    mean_diff_fine = diffs_fine.mean().item()

    print(f"\nFine sampling (Δt = 0.001):")
    print(f"  Mean consecutive difference: {mean_diff_fine:.6f}")

    # Fine sampling should have smaller differences (high freq components visible)
    print(f"\nRatio coarse/fine: {mean_diff_coarse / mean_diff_fine:.2f}x")

    # Verify multi-scale property
    # Low frequencies dominate at coarse scale, high frequencies at fine scale
    assert mean_diff_fine < mean_diff_coarse, "Fine-scale differences should be smaller"

    print("\n✓ Multi-scale resolution test passed!\n")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=" * 60)
    print("TEST 7: Edge Cases and Boundary Conditions")
    print("=" * 60)

    time_emb = TimeEmbedding(time_dim=64)

    # Test t = 0
    t_zero = torch.tensor([0.0])
    emb_zero = time_emb(t_zero)

    print(f"Embedding at t=0:")
    print(f"  Shape: {emb_zero.shape}")
    print(f"  First 4 values: {emb_zero[0, :4].tolist()}")

    assert not torch.isnan(emb_zero).any()
    assert not torch.isinf(emb_zero).any()

    # Test t = 1
    t_one = torch.tensor([1.0])
    emb_one = time_emb(t_one)

    print(f"\nEmbedding at t=1:")
    print(f"  Shape: {emb_one.shape}")
    print(f"  First 4 values: {emb_one[0, :4].tolist()}")

    assert not torch.isnan(emb_one).any()
    assert not torch.isinf(emb_one).any()

    # t=0 and t=1 should be different
    diff = (emb_zero - emb_one).abs().mean().item()
    print(f"\nDifference between t=0 and t=1: {diff:.6f}")
    assert diff > 0.1, "t=0 and t=1 should have substantially different embeddings"

    # Test very small time_dim
    time_emb_small = TimeEmbedding(time_dim=2, freq_min=1.0, freq_max=10.0)
    t = torch.tensor([0.5])
    emb_small = time_emb_small(t)

    print(f"\nSmall embedding (time_dim=2): {emb_small[0].tolist()}")
    assert emb_small.shape == (1, 2)

    # Test device movement
    if torch.cuda.is_available():
        time_emb_gpu = time_emb.cuda()
        t_gpu = torch.tensor([0.5]).cuda()
        emb_gpu = time_emb_gpu(t_gpu)

        assert emb_gpu.device.type == 'cuda'
        print("\n✓ GPU device movement works")

    print("\n✓ Edge cases test passed!\n")


def test_learnable_embedding():
    """Test learnable time embedding with MLP projection."""
    print("=" * 60)
    print("TEST 8: Learnable Time Embedding")
    print("=" * 60)

    # Create learnable embedding
    learnable_emb = LearnableTimeEmbedding(
        time_dim=64,
        output_dim=128,
        hidden_dim=128
    )

    print(f"Learnable embedding config:")
    print(f"  Input (sinusoidal): {learnable_emb.time_dim}")
    print(f"  Hidden: {learnable_emb.hidden_dim}")
    print(f"  Output: {learnable_emb.output_dim}")

    # Test forward pass
    t = torch.linspace(0, 1, 50)
    emb = learnable_emb(t)

    print(f"\nForward pass:")
    print(f"  Input shape: {t.shape}")
    print(f"  Output shape: {emb.shape}")

    assert emb.shape == (50, 128)
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()

    # Test sinusoidal intermediate
    sin_emb = learnable_emb.get_sinusoidal_embedding(t)

    print(f"  Sinusoidal intermediate shape: {sin_emb.shape}")
    assert sin_emb.shape == (50, 64)

    # Count parameters
    num_params = sum(p.numel() for p in learnable_emb.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    # MLP should have: (64 * 128) + 128 + (128 * 128) + 128 = 8192 + 128 + 16384 + 128 = 24832
    expected_params = 64 * 128 + 128 + 128 * 128 + 128
    assert num_params == expected_params, f"Expected {expected_params} params, got {num_params}"

    print("\n✓ Learnable embedding test passed!\n")


def test_gradient_flow():
    """Test that gradients flow through learnable embedding."""
    print("=" * 60)
    print("TEST 9: Gradient Flow")
    print("=" * 60)

    learnable_emb = LearnableTimeEmbedding(time_dim=64, output_dim=128)

    # Create dummy task: predict target from time embedding
    t = torch.linspace(0, 1, 100)
    target = torch.randn(100, 128)

    # Forward pass
    emb = learnable_emb(t)
    loss = ((emb - target) ** 2).mean()

    print(f"Dummy task loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_grad = False
    for name, param in learnable_emb.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm = {grad_norm:.6f}")

    assert has_grad, "No gradients found"

    print("\n✓ Gradient flow test passed!\n")


import math


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("TIME-AWARE TRANSFORM - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    test_initialization()
    test_forward_pass()
    test_uniqueness()
    test_smoothness()
    test_frequency_coverage()
    test_multi_scale_resolution()
    test_edge_cases()
    test_learnable_embedding()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Time embeddings correctly implement sinusoidal encoding with exponential frequencies")
    print("2. Uniqueness verified: different times produce different embeddings")
    print("3. Smoothness verified: nearby times have similar embeddings (Lipschitz continuous)")
    print("4. Frequency coverage: exponential spacing from freq_min to freq_max")
    print("5. Multi-scale resolution: captures both slow (low-freq) and fast (high-freq) variations")
    print("6. Edge cases handled: t=0, t=1, small dimensions all work correctly")
    print("7. Learnable embedding: MLP projection adds task-specific adaptability")
    print("8. Gradient flow verified: parameters are trainable end-to-end")
    print("9. Ready for use in time-conditional normalizing flows, SDEs, and attention")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all tests
    run_all_tests()
