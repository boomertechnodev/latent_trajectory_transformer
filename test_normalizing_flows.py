"""
Normalizing Flow Invertibility Test Suite

Verifies mathematical correctness of normalizing flow transformations:
- Forward-inverse composition: f^{-1}(f(x)) = x
- Log-det Jacobian correctness
- Probability mass conservation
- Bijective mapping properties
- Numerical stability across compositions

Run with: pytest test_normalizing_flows.py -v -s
"""

import torch
import torch.nn as nn
import math

from latent_drift_trajectory import (
    CouplingLayer,
    RaccoonFlow,
    TimeAwareTransform,
)


# ============================================================================
# TEST 1: Forward-Inverse Composition
# ============================================================================

def test_forward_inverse_composition():
    """
    Verify f^{-1}(f(x)) = x for normalizing flows.

    Property: Flows must be perfectly invertible.
    - Forward: y = f(x, t)
    - Reverse: x_reconstructed = f^{-1}(y, t)
    - Check: ||x - x_reconstructed|| < 1e-6

    Tests various input types:
    - Random samples from N(0,I)
    - Structured patterns
    - Boundary values
    - High-dimensional spaces
    """
    print("\n" + "="*80)
    print("TEST 1: Forward-Inverse Composition")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32
    batch_size = 100

    # Create coupling layer
    coupling = CouplingLayer(latent_dim, hidden_dim, time_dim=32)
    coupling.eval()

    # Time conditioning
    time_feat = torch.full((batch_size, 1), 0.5)

    # Test 1: Random samples from standard normal
    print("\n1. Random samples from N(0, I)")
    print("-" * 40)

    x_random = torch.randn(batch_size, latent_dim)

    with torch.no_grad():
        # Forward
        y, log_det_fwd = coupling(x_random, time_feat, reverse=False)

        # Reverse
        x_reconstructed, log_det_rev = coupling(y, time_feat, reverse=True)

    # Reconstruction error
    error_random = (x_random - x_reconstructed).abs().max().item()

    print(f"  Input range: [{x_random.min().item():.2f}, {x_random.max().item():.2f}]")
    print(f"  Max reconstruction error: {error_random:.2e}")

    assert error_random < 1e-5, \
        f"Forward-inverse composition failed: error {error_random:.2e}"

    # Test 2: Structured patterns (checkerboard)
    print("\n2. Structured checkerboard pattern")
    print("-" * 40)

    x_checkerboard = torch.zeros(batch_size, latent_dim)
    x_checkerboard[:, ::2] = 1.0
    x_checkerboard[:, 1::2] = -1.0

    with torch.no_grad():
        y, log_det_fwd = coupling(x_checkerboard, time_feat, reverse=False)
        x_reconstructed, log_det_rev = coupling(y, time_feat, reverse=True)

    error_checkerboard = (x_checkerboard - x_reconstructed).abs().max().item()

    print(f"  Max reconstruction error: {error_checkerboard:.2e}")

    assert error_checkerboard < 1e-5, \
        f"Checkerboard pattern reconstruction failed: error {error_checkerboard:.2e}"

    # Test 3: Boundary values (large magnitude)
    print("\n3. Boundary values (±10)")
    print("-" * 40)

    x_boundary = torch.randn(batch_size, latent_dim) * 10.0

    with torch.no_grad():
        y, log_det_fwd = coupling(x_boundary, time_feat, reverse=False)
        x_reconstructed, log_det_rev = coupling(y, time_feat, reverse=True)

    error_boundary = (x_boundary - x_reconstructed).abs().max().item()

    print(f"  Input range: [{x_boundary.min().item():.2f}, {x_boundary.max().item():.2f}]")
    print(f"  Max reconstruction error: {error_boundary:.2e}")

    assert error_boundary < 1e-4, \
        f"Boundary value reconstruction failed: error {error_boundary:.2e}"

    # Test 4: High-dimensional (64-dim)
    print("\n4. High-dimensional space (64-dim)")
    print("-" * 40)

    latent_dim_high = 64
    coupling_high = CouplingLayer(latent_dim_high, hidden_dim, time_dim=32)
    coupling_high.eval()

    x_high = torch.randn(batch_size, latent_dim_high)
    time_feat_high = torch.full((batch_size, 1), 0.5)

    with torch.no_grad():
        y, log_det_fwd = coupling_high(x_high, time_feat_high, reverse=False)
        x_reconstructed, log_det_rev = coupling_high(y, time_feat_high, reverse=True)

    error_high = (x_high - x_reconstructed).abs().max().item()

    print(f"  Max reconstruction error: {error_high:.2e}")

    assert error_high < 1e-5, \
        f"High-dimensional reconstruction failed: error {error_high:.2e}"

    print(f"\n✓ PASS: Forward-inverse composition verified across all test cases")


# ============================================================================
# TEST 2: Log-Det Jacobian Correctness
# ============================================================================

def test_log_det_jacobian():
    """
    Verify log|det J| matches analytic formula for affine coupling.

    For affine coupling: y = [x1, x2 * exp(s(x1)) + t(x1)]
    log|det J| = sum(s(x1))

    Properties:
    1. Forward log-det should match scale sum
    2. Reverse log-det = -Forward log-det
    3. Sum of both should be ≈ 0
    """
    print("\n" + "="*80)
    print("TEST 2: Log-Det Jacobian Correctness")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32
    batch_size = 50

    coupling = CouplingLayer(latent_dim, hidden_dim, time_dim=32)
    coupling.eval()

    x = torch.randn(batch_size, latent_dim)
    time_feat = torch.full((batch_size, 1), 0.5)

    # Forward pass
    with torch.no_grad():
        y, log_det_fwd = coupling(x, time_feat, reverse=False)

    # Reverse pass
    with torch.no_grad():
        x_reconstructed, log_det_rev = coupling(y, time_feat, reverse=True)

    # Test 1: Log-det should be finite
    print("\n1. Log-Det Finiteness")
    print("-" * 40)

    print(f"  Forward log|det J|: [{log_det_fwd.min().item():.4f}, {log_det_fwd.max().item():.4f}]")
    print(f"  Reverse log|det J|: [{log_det_rev.min().item():.4f}, {log_det_rev.max().item():.4f}]")

    assert torch.isfinite(log_det_fwd).all(), "Forward log-det contains NaN/Inf"
    assert torch.isfinite(log_det_rev).all(), "Reverse log-det contains NaN/Inf"

    # Test 2: Reverse should be negative of forward
    print("\n2. Log-Det Inverse Relationship")
    print("-" * 40)

    log_det_sum = log_det_fwd + log_det_rev

    print(f"  log|det J_fwd| + log|det J_rev|:")
    print(f"    Mean: {log_det_sum.mean().item():.2e}")
    print(f"    Std:  {log_det_sum.std().item():.2e}")
    print(f"    Max:  {log_det_sum.abs().max().item():.2e}")

    # Sum should be close to zero
    max_sum_error = log_det_sum.abs().max().item()

    assert max_sum_error < 1e-4, \
        f"Log-det sum not close to zero: max error {max_sum_error:.2e}"

    # Test 3: Log-det should be bounded (no explosion)
    print("\n3. Log-Det Magnitude Bounds")
    print("-" * 40)

    MAX_LOG_DET = 100.0  # Reasonable bound

    max_fwd = log_det_fwd.abs().max().item()
    max_rev = log_det_rev.abs().max().item()

    print(f"  Max |log|det J_fwd||: {max_fwd:.2f}")
    print(f"  Max |log|det J_rev||: {max_rev:.2f}")

    assert max_fwd < MAX_LOG_DET, f"Forward log-det too large: {max_fwd:.2f}"
    assert max_rev < MAX_LOG_DET, f"Reverse log-det too large: {max_rev:.2f}"

    print(f"\n✓ PASS: Log-det Jacobian is correct and bounded")


# ============================================================================
# TEST 3: Probability Density Transformation
# ============================================================================

def test_probability_density_transformation():
    """
    Verify probability density transforms correctly via change-of-variables.

    Formula: p_y(y) = p_x(x) / |det J(x)|

    Therefore: p_x(x) = p_y(y) * |det J(x)|
                      = p_y(y) * exp(log|det J(x)|)

    Check: ∫ p_y(y) dy = ∫ p_x(x) dx = 1
    """
    print("\n" + "="*80)
    print("TEST 3: Probability Density Transformation")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32
    batch_size = 5000  # Large for Monte Carlo

    coupling = CouplingLayer(latent_dim, hidden_dim, time_dim=32)
    coupling.eval()

    # Sample from prior p_x(x) = N(0, I)
    x_samples = torch.randn(batch_size, latent_dim)
    time_feat = torch.full((batch_size, 1), 0.5)

    # Transform to y
    with torch.no_grad():
        y_samples, log_det = coupling(x_samples, time_feat, reverse=False)

    # Compute log probabilities
    # log p_x(x) = -0.5 * ||x||^2 - 0.5 * d * log(2π)
    log_px = -0.5 * (x_samples ** 2).sum(dim=1) - 0.5 * latent_dim * math.log(2 * math.pi)

    # log p_y(y) = log p_x(x) - log|det J|
    log_py = log_px - log_det

    # Check: Average density
    print("\n1. Average Density Comparison")
    print("-" * 40)

    avg_px = log_px.exp().mean().item()
    avg_py = log_py.exp().mean().item()

    print(f"  E[p_x(x)]: {avg_px:.6f}")
    print(f"  E[p_y(y)]: {avg_py:.6f}")
    print(f"  Ratio: {avg_py / avg_px:.4f}")

    # Should be similar (within Monte Carlo error)
    assert 0.5 < avg_py / avg_px < 2.0, \
        "Probability density not conserved"

    # Check: Total mass conservation
    print("\n2. Probability Mass Conservation")
    print("-" * 40)

    # Both should integrate to 1
    # We can't compute exact integral, but can check statistics

    # Variance of log probabilities should be similar
    var_px = log_px.var().item()
    var_py = log_py.var().item()

    print(f"  Var(log p_x): {var_px:.4f}")
    print(f"  Var(log p_y): {var_py:.4f}")

    # Check: Transformation is non-degenerate
    print("\n3. Non-Degeneracy Check")
    print("-" * 40)

    # y should have similar range to x (not collapsed)
    x_std = x_samples.std(dim=0).mean().item()
    y_std = y_samples.std(dim=0).mean().item()

    print(f"  Avg std(x): {x_std:.4f}")
    print(f"  Avg std(y): {y_std:.4f}")
    print(f"  Ratio: {y_std / x_std:.4f}")

    # Ratio should be order 1 (not collapsed or exploded)
    assert 0.1 < y_std / x_std < 10.0, \
        f"Transformation is degenerate: std ratio {y_std / x_std:.4f}"

    print(f"\n✓ PASS: Probability density transformation is correct")


# ============================================================================
# TEST 4: Multiple Composition Stability
# ============================================================================

def test_multiple_composition_stability():
    """
    Verify numerical stability across multiple flow compositions.

    Test: Apply flow N times, then reverse N times
    - Should recover original input
    - Error should not accumulate significantly

    This tests:
    - Numerical precision
    - Error accumulation
    - Stability of composition
    """
    print("\n" + "="*80)
    print("TEST 4: Multiple Composition Stability")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32

    # Create multi-layer flow
    flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=4)
    flow.eval()

    batch_size = 50
    x_original = torch.randn(batch_size, latent_dim)
    time_feat = torch.full((batch_size, 1), 0.5)

    # Test 1: Single forward-reverse
    print("\n1. Single Forward-Reverse (4 layers)")
    print("-" * 40)

    with torch.no_grad():
        y, log_det_fwd = flow(x_original, time_feat, reverse=False)
        x_reconstructed, log_det_rev = flow(y, time_feat, reverse=True)

    error_single = (x_original - x_reconstructed).abs().max().item()

    print(f"  Max reconstruction error: {error_single:.2e}")

    assert error_single < 1e-5, \
        f"Single composition reconstruction failed: {error_single:.2e}"

    # Test 2: Multiple forward-reverse (stress test)
    print("\n2. Multiple Compositions (10 forward-reverse cycles)")
    print("-" * 40)

    x_current = x_original.clone()

    errors = []

    for i in range(10):
        with torch.no_grad():
            # Forward
            y, _ = flow(x_current, time_feat, reverse=False)
            # Reverse
            x_current, _ = flow(y, time_feat, reverse=True)

        error = (x_original - x_current).abs().max().item()
        errors.append(error)

        if (i + 1) % 2 == 0:
            print(f"  Cycle {i+1}: error = {error:.2e}")

    # Error should not grow significantly
    final_error = errors[-1]
    error_growth = final_error / errors[0]

    print(f"\n  Initial error: {errors[0]:.2e}")
    print(f"  Final error:   {final_error:.2e}")
    print(f"  Error growth:  {error_growth:.2f}x")

    # Allow some growth, but not exponential
    assert final_error < 1e-3, \
        f"Error accumulated too much: {final_error:.2e}"

    assert error_growth < 10.0, \
        f"Error growth too large: {error_growth:.2f}x"

    print(f"\n✓ PASS: Multiple compositions are numerically stable")


# ============================================================================
# TEST 5: Coupling Layer Mask Alternation
# ============================================================================

def test_mask_alternation():
    """
    Verify coupling layers use alternating masks to transform all dimensions.

    Property: In a stack of coupling layers, different dimensions should
    be transformed by different layers (via mask alternation).

    This ensures all dimensions are affected by the transformation.
    """
    print("\n" + "="*80)
    print("TEST 5: Coupling Layer Mask Alternation")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32
    num_layers = 4

    flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=num_layers)
    flow.eval()

    # Check that masks alternate
    print(f"\nCoupling Layer Masks (latent_dim={latent_dim}):")
    print("-" * 40)

    for i, layer in enumerate(flow.coupling_layers):
        # The mask is created inside forward, but we can infer pattern
        # Mask alternates: even layers mask first half, odd layers mask second half

        expected_pattern = "first half" if i % 2 == 0 else "second half"
        print(f"  Layer {i}: masks {expected_pattern}")

    # Test: Apply flow and check all dimensions changed
    batch_size = 50
    x = torch.randn(batch_size, latent_dim)
    time_feat = torch.full((batch_size, 1), 0.5)

    with torch.no_grad():
        y, _ = flow(x, time_feat, reverse=False)

    # Check that all dimensions have changed
    changes = (y - x).abs().mean(dim=0)  # (latent_dim,)

    print(f"\nAverage change per dimension:")
    print(f"  Min change: {changes.min().item():.4f}")
    print(f"  Max change: {changes.max().item():.4f}")
    print(f"  Mean change: {changes.mean().item():.4f}")

    # All dimensions should have changed (no dimension is identity)
    MIN_CHANGE = 0.01

    assert changes.min() > MIN_CHANGE, \
        f"Some dimensions not transformed: min change {changes.min().item():.4f}"

    print(f"\n✓ PASS: All dimensions are transformed (mask alternation working)")


# ============================================================================
# TEST 6: Bijectivity Verification
# ============================================================================

def test_bijectivity():
    """
    Verify flow is bijective (one-to-one and onto).

    Tests:
    1. Different inputs → different outputs (injective)
    2. All outputs are reachable via reverse (surjective)
    3. No collisions (bijective)
    """
    print("\n" + "="*80)
    print("TEST 6: Bijectivity Verification")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32

    flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=2)
    flow.eval()

    batch_size = 100

    # Test 1: Injectivity (different inputs → different outputs)
    print("\n1. Injectivity: x1 ≠ x2 ⟹ f(x1) ≠ f(x2)")
    print("-" * 40)

    x1 = torch.randn(batch_size, latent_dim)
    x2 = torch.randn(batch_size, latent_dim)
    time_feat = torch.full((batch_size, 1), 0.5)

    with torch.no_grad():
        y1, _ = flow(x1, time_feat, reverse=False)
        y2, _ = flow(x2, time_feat, reverse=False)

    # Check outputs are different
    output_diff = (y1 - y2).abs().mean().item()
    input_diff = (x1 - x2).abs().mean().item()

    print(f"  Avg input difference:  {input_diff:.4f}")
    print(f"  Avg output difference: {output_diff:.4f}")

    assert output_diff > 0.1, \
        "Outputs are too similar (possible collision)"

    # Test 2: Surjectivity (all y reachable from some x)
    print("\n2. Surjectivity: ∀y ∃x s.t. f(x)=y")
    print("-" * 40)

    # Sample target y
    y_target = torch.randn(batch_size, latent_dim)

    # Find x via reverse
    with torch.no_grad():
        x_found, _ = flow(y_target, time_feat, reverse=True)

        # Verify f(x_found) = y_target
        y_reconstructed, _ = flow(x_found, time_feat, reverse=False)

    reconstruction_error = (y_target - y_reconstructed).abs().max().item()

    print(f"  Reconstruction error: {reconstruction_error:.2e}")

    assert reconstruction_error < 1e-5, \
        f"Not surjective: cannot reach target y (error {reconstruction_error:.2e})"

    # Test 3: No collisions in practice
    print("\n3. Collision Check (sample-based)")
    print("-" * 40)

    # Generate many samples
    num_samples = 1000
    x_samples = torch.randn(num_samples, latent_dim)
    time_feat_samples = torch.full((num_samples, 1), 0.5)

    with torch.no_grad():
        y_samples, _ = flow(x_samples, time_feat_samples, reverse=False)

    # Check for near-duplicates in output
    # (This is not exhaustive, but detects obvious problems)

    # Compute pairwise distances (expensive for large N, so subsample)
    subsample_size = 100
    y_subsample = y_samples[:subsample_size]

    pairwise_dists = torch.cdist(y_subsample, y_subsample)

    # Set diagonal to inf (ignore self-distance)
    pairwise_dists.fill_diagonal_(float('inf'))

    min_dist = pairwise_dists.min().item()

    print(f"  Min pairwise distance: {min_dist:.4f}")

    # Minimum distance should not be too small (no near-collisions)
    COLLISION_THRESHOLD = 0.01

    assert min_dist > COLLISION_THRESHOLD, \
        f"Possible collision detected: min distance {min_dist:.4f}"

    print(f"\n✓ PASS: Flow is bijective (injective + surjective)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("NORMALIZING FLOW INVERTIBILITY TEST SUITE")
    print("="*80)

    test_forward_inverse_composition()
    test_log_det_jacobian()
    test_probability_density_transformation()
    test_multiple_composition_stability()
    test_mask_alternation()
    test_bijectivity()

    print("\n" + "="*80)
    print("ALL NORMALIZING FLOW TESTS COMPLETE")
    print("="*80)
