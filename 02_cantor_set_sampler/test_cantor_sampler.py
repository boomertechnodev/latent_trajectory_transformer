"""
Test Suite for Cantor Set Sampler

This test suite verifies:
1. Self-similarity property across scales
2. Correct sample counts (2^k at scale k)
3. Coverage measurement
4. Fractal dimension calculation
5. Batched operation
6. Edge cases and validation
"""

import torch
import numpy as np
from cantor_sampler import CantorSetSampler


def test_initialization():
    """Test sampler initializes correctly with different parameters."""
    print("=" * 60)
    print("TEST 1: Initialization")
    print("=" * 60)

    # Default initialization
    sampler = CantorSetSampler()
    print(f"Default - Scales: {sampler.num_scales}, Max distance: {sampler.max_distance}")
    print(f"Max samples: {2**sampler.num_scales}")
    print(f"Fractal dimension: {sampler.get_fractal_dimension():.6f}")

    assert sampler.num_scales == 5, "Default num_scales should be 5"
    assert sampler.max_distance == 0.5, "Default max_distance should be 0.5"

    # Expected fractal dimension
    expected_dim = np.log(2) / np.log(3)
    assert abs(sampler.get_fractal_dimension() - expected_dim) < 1e-10

    # Custom parameters
    sampler_custom = CantorSetSampler(num_scales=3, max_distance=0.3)
    print(f"\nCustom - Scales: {sampler_custom.num_scales}, Max distance: {sampler_custom.max_distance}")
    assert sampler_custom.num_scales == 3
    assert sampler_custom.max_distance == 0.3

    # Test invalid parameters
    try:
        CantorSetSampler(num_scales=0)
        assert False, "Should raise error for num_scales=0"
    except ValueError as e:
        print(f"\n✓ Correctly rejected num_scales=0: {e}")

    try:
        CantorSetSampler(max_distance=1.5)
        assert False, "Should raise error for max_distance > 1"
    except ValueError as e:
        print(f"✓ Correctly rejected max_distance=1.5: {e}")

    print("✓ Initialization test passed!\n")


def test_sample_counts():
    """Test that each scale generates exactly 2^k samples."""
    print("=" * 60)
    print("TEST 2: Sample Counts at Each Scale")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=5)
    center_idx = 512
    seq_len = 1024

    print(f"Center index: {center_idx}, Sequence length: {seq_len}\n")

    for scale in range(sampler.num_scales + 1):
        samples = sampler.forward(center_idx, seq_len, scale=scale)
        expected_count = 2 ** scale
        actual_count = len(samples)

        print(f"Scale {scale}: {actual_count} samples (expected {expected_count})")

        assert actual_count == expected_count, \
            f"Scale {scale} should have {expected_count} samples, got {actual_count}"

        # Check samples are within bounds
        assert samples.min() >= 0, f"Samples below 0 at scale {scale}"
        assert samples.max() < seq_len, f"Samples >= seq_len at scale {scale}"

    print("✓ Sample count test passed!\n")


def test_self_similarity():
    """
    Test the critical self-similarity property of Cantor sets.

    Each scale should contain all positions from the previous scale,
    plus additional positions. This is the defining property of fractals.
    """
    print("=" * 60)
    print("TEST 3: Self-Similarity Property (CRITICAL)")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=6)

    for scale in range(1, sampler.num_scales + 1):
        is_similar, message = sampler.verify_self_similarity(scale)

        print(f"Scale {scale}: {message}")
        assert is_similar, f"Self-similarity violated at scale {scale}: {message}"

    # Visual demonstration: show how scale 2 contains scale 1
    print("\nVisual demonstration:")
    center = 500
    seq_len = 1000

    scale1_samples = sampler.get_scale_samples(center, seq_len, scale=1)
    scale2_samples = sampler.get_scale_samples(center, seq_len, scale=2)

    print(f"Scale 1 samples: {scale1_samples.tolist()}")
    print(f"Scale 2 samples: {scale2_samples.tolist()}")

    # Check scale 1 samples are in scale 2
    for s1 in scale1_samples:
        assert s1 in scale2_samples, f"Scale 1 sample {s1} not in scale 2"

    print("\n✓ Self-similarity test passed!\n")


def test_coverage():
    """Test coverage measurement across different scales."""
    print("=" * 60)
    print("TEST 4: Sequence Coverage")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=5)
    center_idx = 512
    seq_len = 1024

    print(f"Sequence length: {seq_len}\n")

    for scale in range(sampler.num_scales + 1):
        coverage = sampler.compute_coverage(center_idx, seq_len, scale=scale)
        num_samples = 2 ** scale
        theoretical_coverage = num_samples / seq_len

        print(f"Scale {scale}: {coverage:.4f} coverage "
              f"({num_samples} samples / {seq_len} positions)")

        # Coverage should be approximately num_samples / seq_len
        # Allow some deviation due to clamping at boundaries
        assert 0 <= coverage <= 1, f"Coverage out of bounds: {coverage}"
        assert coverage <= theoretical_coverage * 1.1, \
            f"Coverage too high: {coverage} > {theoretical_coverage * 1.1}"

    print("✓ Coverage test passed!\n")


def test_batched_operation():
    """Test that sampler works with batched inputs."""
    print("=" * 60)
    print("TEST 5: Batched Operation")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=4)
    seq_len = 1024

    # Test scalar input
    scalar_idx = 100
    scalar_samples = sampler(scalar_idx, seq_len, scale=3)
    print(f"Scalar input shape: {scalar_samples.shape}")
    assert scalar_samples.dim() == 1, "Scalar input should give 1D output"
    assert len(scalar_samples) == 8, "Scale 3 should give 8 samples"

    # Test batched input
    batch_indices = torch.tensor([100, 200, 300, 400])
    batch_samples = sampler(batch_indices, seq_len, scale=3)
    print(f"Batch input shape: {batch_samples.shape}")
    assert batch_samples.shape == (4, 8), "Should be (batch_size, num_samples)"

    # Verify each batch element
    for i, center in enumerate(batch_indices):
        expected = sampler(center.item(), seq_len, scale=3)
        actual = batch_samples[i]
        assert torch.allclose(expected, actual), \
            f"Batch element {i} doesn't match scalar computation"

    print("✓ Batched operation test passed!\n")


def test_all_scales_retrieval():
    """Test retrieving all scales at once."""
    print("=" * 60)
    print("TEST 6: All Scales Retrieval")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=4)
    center_idx = 512
    seq_len = 1024

    all_scales = sampler.get_all_scales(center_idx, seq_len)

    print(f"Number of scales: {len(all_scales)}")
    assert len(all_scales) == sampler.num_scales + 1, \
        "Should return num_scales + 1 (including scale 0)"

    for scale, samples in enumerate(all_scales):
        expected_count = 2 ** scale
        actual_count = len(samples)
        print(f"Scale {scale}: {actual_count} samples")
        assert actual_count == expected_count

    print("✓ All scales retrieval test passed!\n")


def test_boundary_cases():
    """Test edge cases like center at boundaries."""
    print("=" * 60)
    print("TEST 7: Boundary Cases")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=4)
    seq_len = 1024

    # Test center at start
    samples_start = sampler(0, seq_len, scale=3)
    print(f"Center at 0: samples range [{samples_start.min()}, {samples_start.max()}]")
    assert samples_start.min() >= 0
    assert samples_start.max() < seq_len

    # Test center at end
    samples_end = sampler(seq_len - 1, seq_len, scale=3)
    print(f"Center at {seq_len-1}: samples range [{samples_end.min()}, {samples_end.max()}]")
    assert samples_end.min() >= 0
    assert samples_end.max() < seq_len

    # Test center in middle
    samples_mid = sampler(seq_len // 2, seq_len, scale=3)
    print(f"Center at {seq_len//2}: samples range [{samples_mid.min()}, {samples_mid.max()}]")
    assert samples_mid.min() >= 0
    assert samples_mid.max() < seq_len

    # All samples should be valid indices
    for samples in [samples_start, samples_end, samples_mid]:
        assert all(0 <= s < seq_len for s in samples), "Invalid sample indices"

    print("✓ Boundary cases test passed!\n")


def test_fractal_dimension():
    """Test that fractal dimension matches theoretical value."""
    print("=" * 60)
    print("TEST 8: Fractal Dimension")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=5)

    theoretical_dim = np.log(2) / np.log(3)
    computed_dim = sampler.get_fractal_dimension()

    print(f"Theoretical fractal dimension: {theoretical_dim:.10f}")
    print(f"Computed fractal dimension: {computed_dim:.10f}")
    print(f"Difference: {abs(theoretical_dim - computed_dim):.2e}")

    assert abs(computed_dim - theoretical_dim) < 1e-10, \
        "Fractal dimension doesn't match theoretical value"

    # Verify it's between 0 and 1 (characteristic of fractal curves)
    assert 0 < computed_dim < 1, \
        f"Fractal dimension should be in (0,1), got {computed_dim}"

    print(f"\n✓ Fractal dimension = {computed_dim:.6f} ≈ 0.631")
    print("✓ Fractal dimension test passed!\n")


def visualize_cantor_pattern():
    """
    Visualize the Cantor set sampling pattern for different scales.

    Creates ASCII visualization showing the hierarchical structure.
    """
    print("=" * 60)
    print("VISUALIZATION: Cantor Set Hierarchical Pattern")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=4, max_distance=0.5)
    center = 50
    seq_len = 100

    for scale in range(sampler.num_scales + 1):
        samples = sampler.get_scale_samples(center, seq_len, scale=scale)

        # Create visualization string
        vis = [' '] * seq_len
        for s in samples:
            vis[s.item()] = '█'
        vis[center] = '●'  # Mark center

        # Print scale info
        coverage = sampler.compute_coverage(center, seq_len, scale=scale)
        print(f"\nScale {scale} - {2**scale} samples ({coverage:.1%} coverage):")

        # Print visualization in chunks
        chunk_size = 50
        for i in range(0, seq_len, chunk_size):
            chunk = ''.join(vis[i:i+chunk_size])
            print(f"  {i:3d}-{min(i+chunk_size-1, seq_len-1):3d}: {chunk}")

    print("\nLegend: ● = center, █ = sampled position")
    print("=" * 60 + "\n")


def test_hierarchical_structure():
    """
    Test that sampling pattern captures hierarchical structure.

    Verify that samples are distributed at multiple distance scales.
    """
    print("=" * 60)
    print("TEST 9: Hierarchical Structure")
    print("=" * 60)

    sampler = CantorSetSampler(num_scales=5)
    center_idx = 512
    seq_len = 1024

    samples = sampler(center_idx, seq_len, scale=5)

    # Compute distances from center
    distances = torch.abs(samples - center_idx).float()

    # Sort distances to see the hierarchical pattern
    sorted_distances = torch.sort(distances)[0]

    print(f"Sample distances from center (sorted):")
    print(f"  First 10: {sorted_distances[:10].tolist()}")
    print(f"  Last 10: {sorted_distances[-10:].tolist()}")

    # Check we have samples at multiple distance scales
    min_dist = distances.min().item()
    max_dist = distances.max().item()
    mean_dist = distances.mean().item()

    print(f"\nDistance statistics:")
    print(f"  Min: {min_dist:.1f}")
    print(f"  Mean: {mean_dist:.1f}")
    print(f"  Max: {max_dist:.1f}")
    print(f"  Range: {max_dist - min_dist:.1f}")

    # Should have good spread (hierarchical)
    # Note: At boundaries, clamping may reduce the ratio
    assert max_dist > min_dist * 2, \
        "Distances should span multiple scales"

    # Group into distance bins to verify multi-scale coverage
    bins = torch.tensor([0.0, 200.0, 300.0, 400.0, 600.0])
    hist = torch.histogram(distances, bins=bins)[0]

    print(f"\nDistance distribution:")
    for i, count in enumerate(hist):
        print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}: {count.item()} samples")

    # Should have samples in multiple bins (multi-scale)
    non_empty_bins = (hist > 0).sum().item()
    assert non_empty_bins >= 3, \
        f"Should cover multiple distance scales, only {non_empty_bins} bins"

    print("✓ Hierarchical structure test passed!\n")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("CANTOR SET SAMPLER - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    test_initialization()
    test_sample_counts()
    test_self_similarity()
    test_coverage()
    test_batched_operation()
    test_all_scales_retrieval()
    test_boundary_cases()
    test_fractal_dimension()
    test_hierarchical_structure()
    visualize_cantor_pattern()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Cantor set correctly generates 2^k samples at scale k")
    print("2. Perfect self-similarity verified across all scales")
    print("3. Fractal dimension = log(2)/log(3) ≈ 0.631 (theoretical)")
    print("4. Hierarchical multi-scale sampling confirmed")
    print("5. Ready for use in fractal attention for long-range dependencies")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all tests
    run_all_tests()
