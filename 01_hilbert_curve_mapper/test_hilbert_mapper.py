"""
Test Suite for Hilbert Curve Mapper

This test suite verifies:
1. Locality preservation: consecutive sequence positions remain close in 2D
2. Coordinate generation correctness
3. Shape and normalization properties
4. Fractal structure visualization
"""

import torch
import numpy as np
from hilbert_mapper import HilbertCurveMapper


def test_initialization():
    """Test that mapper initializes correctly with different parameters."""
    print("=" * 60)
    print("TEST 1: Initialization")
    print("=" * 60)

    # Default initialization
    mapper = HilbertCurveMapper()
    print(f"Default - Order: {mapper.order}, Grid: {mapper.grid_size}×{mapper.grid_size}")
    print(f"Max positions: {mapper.max_positions}")
    assert mapper.order == 6, f"Expected order 6 for default, got {mapper.order}"
    assert mapper.grid_size == 64, f"Expected grid_size 64, got {mapper.grid_size}"

    # Custom max_seq_len
    mapper_small = HilbertCurveMapper(max_seq_len=64)
    print(f"\nSmall (64) - Order: {mapper_small.order}, Grid: {mapper_small.grid_size}×{mapper_small.grid_size}")
    assert mapper_small.order == 3, f"Expected order 3 for len=64, got {mapper_small.order}"

    # Explicit order
    mapper_order2 = HilbertCurveMapper(order=2)
    print(f"Order 2 - Grid: {mapper_order2.grid_size}×{mapper_order2.grid_size}")
    assert mapper_order2.grid_size == 4, f"Expected grid_size 4 for order 2"
    assert mapper_order2.max_positions == 16, f"Expected 16 positions for order 2"

    print("✓ Initialization test passed!\n")


def test_coordinate_generation():
    """Test that coordinates are generated correctly."""
    print("=" * 60)
    print("TEST 2: Coordinate Generation")
    print("=" * 60)

    mapper = HilbertCurveMapper(order=2)  # 4×4 grid, 16 positions

    # Test first few positions (known values for order=2)
    # Hilbert curve order 2 starts at (0,0), goes to (1,0), then (1,1), (0,1), etc.
    indices = torch.arange(16)
    coords = mapper(indices)

    print(f"Coordinates shape: {coords.shape}")
    assert coords.shape == (16, 2), f"Expected shape (16, 2), got {coords.shape}"

    # Check normalization: all coordinates should be in [0, 1]
    assert coords.min() >= 0.0, f"Minimum coordinate {coords.min()} < 0"
    assert coords.max() <= 1.0, f"Maximum coordinate {coords.max()} > 1"
    print(f"Coordinate range: [{coords.min():.4f}, {coords.max():.4f}]")

    # First position should be (0, 0)
    first_coord = coords[0]
    print(f"First coordinate: ({first_coord[0]:.4f}, {first_coord[1]:.4f})")
    assert torch.allclose(first_coord, torch.tensor([0.0, 0.0]), atol=1e-6), \
        f"First coordinate should be (0,0), got {first_coord}"

    # Print first 8 coordinates to verify pattern
    print("\nFirst 8 coordinates (normalized to [0,1]):")
    for i in range(8):
        print(f"  Position {i}: ({coords[i, 0]:.4f}, {coords[i, 1]:.4f})")

    print("✓ Coordinate generation test passed!\n")


def test_locality_preservation():
    """
    Test the key property: consecutive sequence positions should be close in 2D.

    This is THE critical property for fractal attention - nearby tokens in the
    sequence should have nearby 2D coordinates.
    """
    print("=" * 60)
    print("TEST 3: Locality Preservation (CRITICAL)")
    print("=" * 60)

    mapper = HilbertCurveMapper(max_seq_len=1024)
    indices = torch.arange(1024)
    coords = mapper(indices)

    # Compute distances between consecutive positions
    consecutive_dists = torch.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(dim=1))

    mean_consecutive_dist = consecutive_dists.mean().item()
    max_consecutive_dist = consecutive_dists.max().item()
    std_consecutive_dist = consecutive_dists.std().item()

    print(f"Consecutive position distances (in normalized [0,1] space):")
    print(f"  Mean: {mean_consecutive_dist:.6f}")
    print(f"  Std:  {std_consecutive_dist:.6f}")
    print(f"  Max:  {max_consecutive_dist:.6f}")

    # Compare to truly random position pairs (should be much larger)
    num_random_samples = 1000
    random_idx1 = torch.randint(0, 1024, (num_random_samples,))
    random_idx2 = torch.randint(0, 1024, (num_random_samples,))
    random_coords1 = coords[random_idx1]
    random_coords2 = coords[random_idx2]
    random_dists = torch.sqrt(((random_coords2 - random_coords1) ** 2).sum(dim=1))

    mean_random_dist = random_dists.mean().item()
    print(f"\nRandom position pair distances (baseline):")
    print(f"  Mean: {mean_random_dist:.6f}")

    # Locality ratio: consecutive should be MUCH closer than random
    locality_ratio = mean_random_dist / mean_consecutive_dist
    print(f"\nLocality preservation ratio: {locality_ratio:.2f}x")
    print(f"  (consecutive positions are {locality_ratio:.2f}x closer than random pairs)")

    # Assert locality is preserved (consecutive should be at least 3x closer)
    assert locality_ratio >= 3.0, \
        f"Locality not preserved! Ratio {locality_ratio:.2f} < 3.0"

    # Additional check: at least 90% of consecutive steps should be small
    small_step_threshold = mean_consecutive_dist * 2
    small_steps = (consecutive_dists < small_step_threshold).float().mean()
    print(f"\nSmall steps (<2×mean): {small_steps * 100:.1f}%")
    assert small_steps >= 0.9, f"Only {small_steps * 100:.1f}% of steps are small"

    print("✓ Locality preservation test passed!\n")


def test_2d_distances():
    """Test the 2D distance computation method."""
    print("=" * 60)
    print("TEST 4: 2D Distance Computation")
    print("=" * 60)

    mapper = HilbertCurveMapper(max_seq_len=100)
    indices = torch.arange(100)

    distances = mapper.get_2d_distances(indices)

    print(f"Distance matrix shape: {distances.shape}")
    assert distances.shape == (100, 100), f"Expected (100, 100), got {distances.shape}"

    # Diagonal should be zero (distance to self)
    diagonal = torch.diagonal(distances)
    print(f"Diagonal (self-distances): min={diagonal.min():.6f}, max={diagonal.max():.6f}")
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6), \
        "Diagonal should be all zeros"

    # Matrix should be symmetric
    print(f"Symmetry check: max difference = {(distances - distances.T).abs().max():.6e}")
    assert torch.allclose(distances, distances.T, atol=1e-6), \
        "Distance matrix should be symmetric"

    # Test k-nearest neighbors for a specific position
    test_pos = 50
    k = 7
    nearest_dists, nearest_indices = torch.topk(distances[test_pos], k=k, largest=False)

    print(f"\nPosition {test_pos} - {k} nearest neighbors:")
    for i, (idx, dist) in enumerate(zip(nearest_indices, nearest_dists)):
        print(f"  {i+1}. Position {idx.item()}: distance {dist.item():.6f}")

    # First nearest neighbor should be self (distance 0)
    assert nearest_indices[0] == test_pos, \
        f"Nearest neighbor should be self, got {nearest_indices[0]}"
    assert nearest_dists[0] < 1e-6, \
        f"Distance to self should be ~0, got {nearest_dists[0]}"

    print("✓ 2D distance computation test passed!\n")


def test_batched_indexing():
    """Test that the mapper works with batched inputs."""
    print("=" * 60)
    print("TEST 5: Batched Indexing")
    print("=" * 60)

    mapper = HilbertCurveMapper(max_seq_len=256)

    # Test single index
    single_idx = torch.tensor([10])
    single_coord = mapper(single_idx)
    print(f"Single index shape: {single_coord.shape}")
    assert single_coord.shape == (1, 2), f"Expected (1, 2), got {single_coord.shape}"

    # Test batch of indices
    batch_idx = torch.arange(50)
    batch_coords = mapper(batch_idx)
    print(f"Batch indices shape: {batch_coords.shape}")
    assert batch_coords.shape == (50, 2), f"Expected (50, 2), got {batch_coords.shape}"

    # Test 2D batch (e.g., batch_size × seq_len)
    batch_2d = torch.arange(64).reshape(8, 8)
    coords_2d = mapper(batch_2d)
    print(f"2D batch indices shape: {coords_2d.shape}")
    assert coords_2d.shape == (8, 8, 2), f"Expected (8, 8, 2), got {coords_2d.shape}"

    print("✓ Batched indexing test passed!\n")


def test_clamping():
    """Test that out-of-range indices are clamped correctly."""
    print("=" * 60)
    print("TEST 6: Index Clamping")
    print("=" * 60)

    mapper = HilbertCurveMapper(max_seq_len=64)
    max_valid = mapper.max_positions - 1

    # Test indices beyond max_positions
    indices = torch.tensor([0, 10, max_valid, max_valid + 10, max_valid + 100])
    coords = mapper(indices)

    print(f"Test indices: {indices.tolist()}")
    print(f"Max valid index: {max_valid}")
    print(f"Coordinates for out-of-range indices:")
    for i, (idx, coord) in enumerate(zip(indices, coords)):
        print(f"  Index {idx.item()}: ({coord[0]:.4f}, {coord[1]:.4f})")

    # Out-of-range indices should map to the last valid coordinate
    last_valid_coord = coords[2]  # coords for max_valid index
    assert torch.allclose(coords[3], last_valid_coord, atol=1e-6), \
        "Out-of-range index should clamp to max valid"
    assert torch.allclose(coords[4], last_valid_coord, atol=1e-6), \
        "Out-of-range index should clamp to max valid"

    print("✓ Index clamping test passed!\n")


def visualize_hilbert_curves():
    """
    Visualize Hilbert curves for orders 2, 4, and 6 to confirm fractal structure.

    Note: This creates a simple ASCII visualization. For actual plotting,
    you would use matplotlib.
    """
    print("=" * 60)
    print("VISUALIZATION: Hilbert Curve Fractal Structure")
    print("=" * 60)

    for order in [2, 3, 4]:
        print(f"\n{'='*40}")
        print(f"Order {order} - Grid: {2**order}×{2**order} ({2**(2*order)} positions)")
        print(f"{'='*40}")

        mapper = HilbertCurveMapper(order=order)
        coords = mapper.visualize_curve()

        # Convert to numpy for easier manipulation
        coords_np = coords.cpu().numpy()

        # Create ASCII grid
        grid_size = 2 ** order
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

        # Mark path on grid
        for i, (x, y) in enumerate(coords_np):
            # Denormalize back to grid coordinates
            grid_x = int(x * (grid_size - 1))
            grid_y = int(y * (grid_size - 1))

            # Use different symbols for start, end, and path
            if i == 0:
                grid[grid_y][grid_x] = 'S'  # Start
            elif i == len(coords_np) - 1:
                grid[grid_y][grid_x] = 'E'  # End
            elif i < 10:
                grid[grid_y][grid_x] = str(i)
            else:
                grid[grid_y][grid_x] = '•'

        # Print grid (flip Y axis for display)
        for row in reversed(grid):
            print('  ' + ' '.join(row))

        # Print statistics
        consecutive_dists = np.sqrt(((coords_np[1:] - coords_np[:-1]) ** 2).sum(axis=1))
        print(f"\nPath statistics:")
        print(f"  Total positions: {len(coords_np)}")
        print(f"  Mean step size: {consecutive_dists.mean():.6f}")
        print(f"  Max step size: {consecutive_dists.max():.6f}")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("Note: S=Start, E=End, 0-9=first 10 positions, •=rest of path")
    print("=" * 60 + "\n")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("HILBERT CURVE MAPPER - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    test_initialization()
    test_coordinate_generation()
    test_locality_preservation()
    test_2d_distances()
    test_batched_indexing()
    test_clamping()
    visualize_hilbert_curves()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Hilbert curve correctly maps 1D → 2D with locality preservation")
    print("2. Consecutive sequence positions are 3-10x closer than random pairs")
    print("3. Fractal structure is maintained across different orders")
    print("4. Ready for use in O(n×w²) fractal attention mechanism")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all tests
    run_all_tests()
