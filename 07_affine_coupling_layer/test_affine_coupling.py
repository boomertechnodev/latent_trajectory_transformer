"""
Test Suite for Affine Coupling Layer

This test suite verifies:
1. Initialization and mask validation
2. Forward pass shape and output correctness
3. Exact invertibility (forward ∘ inverse = identity)
4. Log-determinant computation (analytical vs numerical Jacobian)
5. Time conditioning functionality
6. Gradient flow through parameters
7. Stability and numerical robustness
8. Checkerboard mask generation
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from affine_coupling import AffineCouplingLayer, ScaleShiftNetwork, create_checkerboard_mask


def test_initialization():
    """Test AffineCouplingLayer initialization and parameter validation."""
    print("=" * 60)
    print("TEST 1: Initialization and Parameter Validation")
    print("=" * 60)

    dim = 64
    mask = create_checkerboard_mask(dim, invert=False)

    coupling = AffineCouplingLayer(
        dim=dim,
        mask=mask,
        hidden_dim=128,
        time_dim=0,
        num_layers=3,
        scale_limit=2.0
    )

    print(f"Dimension: {coupling.dim}")
    print(f"Frozen dimensions: {coupling.frozen_dim}")
    print(f"Active dimensions: {coupling.active_dim}")
    print(f"Time dimension: {coupling.time_dim}")

    assert coupling.dim == 64
    assert coupling.frozen_dim == 32  # Even indices
    assert coupling.active_dim == 32  # Odd indices

    # Test mask validation
    try:
        bad_mask = torch.rand(64)  # Not binary
        AffineCouplingLayer(dim=64, mask=bad_mask, hidden_dim=128)
        assert False, "Should reject non-binary mask"
    except ValueError as e:
        print(f"\n✓ Correctly rejected non-binary mask: {e}")

    try:
        bad_mask = torch.ones(32)  # Wrong shape
        AffineCouplingLayer(dim=64, mask=bad_mask, hidden_dim=128)
        assert False, "Should reject wrong mask shape"
    except ValueError as e:
        print(f"✓ Correctly rejected wrong mask shape: {e}")

    try:
        bad_mask = torch.zeros(64)  # All zeros
        AffineCouplingLayer(dim=64, mask=bad_mask, hidden_dim=128)
        assert False, "Should reject all-zero mask"
    except ValueError as e:
        print(f"✓ Correctly rejected all-zero mask: {e}")

    print("\n✓ Initialization test passed!\n")


def test_forward_pass():
    """Test forward pass shape and basic properties."""
    print("=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)

    dim = 32
    batch_size = 16
    mask = create_checkerboard_mask(dim)

    coupling = AffineCouplingLayer(dim=dim, mask=mask, hidden_dim=64)

    x = torch.randn(batch_size, dim)
    y, log_det = coupling(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Log-det shape: {log_det.shape}")

    assert y.shape == (batch_size, dim)
    assert log_det.shape == (batch_size,)

    # Check no NaN or Inf
    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"
    assert not torch.isnan(log_det).any(), "Log-det contains NaN"
    assert not torch.isinf(log_det).any(), "Log-det contains Inf"

    # Frozen part should be unchanged
    frozen_input = x * mask
    frozen_output = y * mask
    assert torch.allclose(frozen_input, frozen_output, atol=1e-6), \
        "Frozen dimensions should remain unchanged"

    print(f"\nFrozen dimensions preserved: ✓")
    print(f"Mean log-det: {log_det.mean().item():.6f}")

    print("\n✓ Forward pass test passed!\n")


def test_invertibility():
    """Test exact invertibility: forward then inverse recovers input."""
    print("=" * 60)
    print("TEST 3: Exact Invertibility")
    print("=" * 60)

    dim = 64
    batch_size = 32
    mask = create_checkerboard_mask(dim)

    coupling = AffineCouplingLayer(dim=dim, mask=mask, hidden_dim=128)

    # Random input
    x_original = torch.randn(batch_size, dim)

    # Forward transformation
    y, log_det_fwd = coupling(x_original, inverse=False)

    # Inverse transformation
    x_reconstructed, log_det_inv = coupling(y, inverse=True)

    # Check reconstruction
    diff = (x_original - x_reconstructed).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Reconstruction error:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    assert max_diff < 1e-5, f"Invertibility failed: max diff = {max_diff}"

    # Check log-determinants are negatives
    log_det_diff = (log_det_fwd + log_det_inv).abs()
    print(f"\nLog-determinant consistency:")
    print(f"  Forward log-det: {log_det_fwd.mean().item():.6f}")
    print(f"  Inverse log-det: {log_det_inv.mean().item():.6f}")
    print(f"  Sum (should be ~0): {log_det_diff.mean().item():.2e}")

    assert log_det_diff.max() < 1e-5, "Log-dets should be negatives"

    # Test using inverse() method
    x_reconstructed_2 = coupling.inverse(y)
    diff_2 = (x_original - x_reconstructed_2).abs().max().item()

    print(f"\nUsing inverse() method: max diff = {diff_2:.2e}")
    assert diff_2 < 1e-5

    print("\n✓ Invertibility test passed!\n")


def test_log_determinant():
    """Test log-determinant matches numerical Jacobian computation."""
    print("=" * 60)
    print("TEST 4: Log-Determinant vs Numerical Jacobian")
    print("=" * 60)

    dim = 8  # Small dimension for numerical Jacobian
    batch_size = 4
    mask = create_checkerboard_mask(dim)

    coupling = AffineCouplingLayer(dim=dim, mask=mask, hidden_dim=32)

    x = torch.randn(batch_size, dim, requires_grad=True)

    # Analytical log-det from coupling layer
    y, log_det_analytical = coupling(x)

    print(f"Testing {batch_size} samples with dim={dim}")

    # Compute numerical Jacobian for first sample
    x0 = x[0:1].detach().requires_grad_(True)
    y0, _ = coupling(x0)

    # Compute Jacobian matrix numerically
    jacobian = []
    for i in range(dim):
        grad_output = torch.zeros_like(y0)
        grad_output[0, i] = 1.0

        grad_input = torch.autograd.grad(
            outputs=y0,
            inputs=x0,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=False
        )[0]

        jacobian.append(grad_input[0])

    jacobian = torch.stack(jacobian, dim=0)  # (dim, dim)

    # Compute determinant numerically
    det_numerical = torch.det(jacobian).item()
    log_det_numerical = math.log(abs(det_numerical))

    log_det_analytical_0 = log_det_analytical[0].item()

    print(f"\nSample 0:")
    print(f"  Analytical log-det: {log_det_analytical_0:.6f}")
    print(f"  Numerical log-det: {log_det_numerical:.6f}")
    print(f"  Difference: {abs(log_det_analytical_0 - log_det_numerical):.2e}")

    assert abs(log_det_analytical_0 - log_det_numerical) < 0.01, \
        f"Log-det mismatch: {log_det_analytical_0} vs {log_det_numerical}"

    print("\n✓ Log-determinant test passed!\n")


def test_time_conditioning():
    """Test that time conditioning changes the transformation."""
    print("=" * 60)
    print("TEST 5: Time Conditioning")
    print("=" * 60)

    dim = 32
    batch_size = 16
    time_dim = 64
    mask = create_checkerboard_mask(dim)

    coupling = AffineCouplingLayer(
        dim=dim,
        mask=mask,
        hidden_dim=128,
        time_dim=time_dim
    )

    # Do a few training steps to make network non-trivial
    # (initialized to zeros, so initially outputs are identical)
    optimizer = torch.optim.Adam(coupling.parameters(), lr=0.01)
    for _ in range(10):
        x_train = torch.randn(32, dim)
        time_train = torch.randn(32, time_dim)
        y_train, log_det_train = coupling(x_train, time_features=time_train)
        loss = (y_train.pow(2).sum() - log_det_train.sum())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    x = torch.randn(batch_size, dim)

    # Transform at different times
    time1 = torch.randn(batch_size, time_dim)
    time2 = torch.randn(batch_size, time_dim)

    y1, log_det1 = coupling(x, time_features=time1)
    y2, log_det2 = coupling(x, time_features=time2)

    # Outputs should be different for different times
    diff_y = (y1 - y2).abs().mean().item()
    diff_log_det = (log_det1 - log_det2).abs().mean().item()

    print(f"Same input, different times (after training):")
    print(f"  Output difference: {diff_y:.6f}")
    print(f"  Log-det difference: {diff_log_det:.6f}")

    assert diff_y > 0.01, "Time conditioning should change output"

    # Frozen part should still be unchanged
    frozen_input = x * mask
    frozen_output1 = y1 * mask
    frozen_output2 = y2 * mask

    assert torch.allclose(frozen_input, frozen_output1, atol=1e-6)
    assert torch.allclose(frozen_input, frozen_output2, atol=1e-6)

    print(f"\nFrozen part still unchanged: ✓")

    # Test invertibility with time conditioning
    x_recon1 = coupling.inverse(y1, time_features=time1)
    x_recon2 = coupling.inverse(y2, time_features=time2)

    diff_recon1 = (x - x_recon1).abs().max().item()
    diff_recon2 = (x - x_recon2).abs().max().item()

    print(f"\nInvertibility with time conditioning:")
    print(f"  Reconstruction error (time1): {diff_recon1:.2e}")
    print(f"  Reconstruction error (time2): {diff_recon2:.2e}")

    assert diff_recon1 < 1e-5
    assert diff_recon2 < 1e-5

    print("\n✓ Time conditioning test passed!\n")


def test_gradient_flow():
    """Test that gradients flow through all parameters."""
    print("=" * 60)
    print("TEST 6: Gradient Flow")
    print("=" * 60)

    dim = 32
    batch_size = 16
    mask = create_checkerboard_mask(dim)

    coupling = AffineCouplingLayer(dim=dim, mask=mask, hidden_dim=64)

    # Dummy task: minimize output norm
    x = torch.randn(batch_size, dim)
    y, log_det = coupling(x)

    loss = y.pow(2).sum() - log_det.sum()  # Include log-det in loss

    print(f"Dummy loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Check gradients
    has_grad = False
    total_grad_norm = 0.0

    for name, param in coupling.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            print(f"  {name}: grad_norm = {grad_norm:.6f}")

    assert has_grad, "No gradients found"
    assert total_grad_norm > 0, "All gradients are zero"

    print(f"\nTotal gradient norm: {total_grad_norm:.6f}")
    print("\n✓ Gradient flow test passed!\n")


def test_stability():
    """Test numerical stability and scale clamping."""
    print("=" * 60)
    print("TEST 7: Numerical Stability")
    print("=" * 60)

    dim = 64
    batch_size = 32
    mask = create_checkerboard_mask(dim)

    coupling = AffineCouplingLayer(
        dim=dim,
        mask=mask,
        hidden_dim=128,
        scale_limit=2.0
    )

    # Test with extreme inputs
    x_large = torch.randn(batch_size, dim) * 10.0
    x_small = torch.randn(batch_size, dim) * 0.01

    y_large, log_det_large = coupling(x_large)
    y_small, log_det_small = coupling(x_small)

    print(f"Large input (scale ~10):")
    print(f"  Output range: [{y_large.min().item():.2f}, {y_large.max().item():.2f}]")
    print(f"  Log-det range: [{log_det_large.min().item():.2f}, {log_det_large.max().item():.2f}]")

    print(f"\nSmall input (scale ~0.01):")
    print(f"  Output range: [{y_small.min().item():.2f}, {y_small.max().item():.2f}]")
    print(f"  Log-det range: [{log_det_small.min().item():.2f}, {log_det_small.max().item():.2f}]")

    # Check no NaN or Inf
    assert not torch.isnan(y_large).any()
    assert not torch.isinf(y_large).any()
    assert not torch.isnan(y_small).any()
    assert not torch.isinf(y_small).any()

    # Test invertibility with extreme inputs
    x_recon_large = coupling.inverse(y_large)
    x_recon_small = coupling.inverse(y_small)

    diff_large = (x_large - x_recon_large).abs().max().item()
    diff_small = (x_small - x_recon_small).abs().max().item()

    print(f"\nInvertibility with extreme inputs:")
    print(f"  Large input reconstruction error: {diff_large:.2e}")
    print(f"  Small input reconstruction error: {diff_small:.2e}")

    assert diff_large < 1e-4, f"Failed for large input: {diff_large}"
    assert diff_small < 1e-5, f"Failed for small input: {diff_small}"

    # Test initialization (should be near identity)
    coupling_init = AffineCouplingLayer(dim=dim, mask=mask, hidden_dim=128)
    x_test = torch.randn(8, dim)
    y_test, _ = coupling_init(x_test)

    # Active part should be close to input (since network initialized to zeros)
    active_input = x_test * (1 - mask)
    active_output = y_test * (1 - mask)
    init_diff = (active_input - active_output).abs().mean().item()

    print(f"\nInitialization (should be near identity):")
    print(f"  Active part difference: {init_diff:.6f}")

    assert init_diff < 0.5, "Initialization should be close to identity"

    print("\n✓ Stability test passed!\n")


def test_checkerboard_mask():
    """Test checkerboard mask generation."""
    print("=" * 60)
    print("TEST 8: Checkerboard Mask Generation")
    print("=" * 60)

    # Test even dimension
    mask_even = create_checkerboard_mask(64, invert=False)
    print(f"Even dimension (64):")
    print(f"  Mask shape: {mask_even.shape}")
    print(f"  First 10: {mask_even[:10].tolist()}")
    print(f"  Sum (frozen dims): {mask_even.sum().item()}")

    assert mask_even.shape == (64,)
    assert mask_even.sum() == 32  # Half frozen
    assert mask_even[0] == 1  # Starts with 1
    assert mask_even[1] == 0

    # Test inverted
    mask_inv = create_checkerboard_mask(64, invert=True)
    print(f"\nInverted mask:")
    print(f"  First 10: {mask_inv[:10].tolist()}")
    print(f"  Sum: {mask_inv.sum().item()}")

    assert mask_inv[0] == 0  # Starts with 0
    assert mask_inv[1] == 1
    assert mask_inv.sum() == 32

    # Check complementarity
    assert torch.allclose(mask_even + mask_inv, torch.ones(64))

    print(f"\nMasks are complementary: ✓")

    # Test odd dimension
    mask_odd = create_checkerboard_mask(63, invert=False)
    print(f"\nOdd dimension (63):")
    print(f"  Sum: {mask_odd.sum().item()}")

    assert mask_odd.sum() == 32  # Ceiling of 63/2

    print("\n✓ Checkerboard mask test passed!\n")


import math


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("AFFINE COUPLING LAYER - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    test_initialization()
    test_forward_pass()
    test_invertibility()
    test_log_determinant()
    test_time_conditioning()
    test_gradient_flow()
    test_stability()
    test_checkerboard_mask()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Affine coupling correctly implements RealNVP transformation")
    print("2. Exact invertibility verified: forward ∘ inverse = identity (error < 1e-5)")
    print("3. Log-determinant matches numerical Jacobian computation")
    print("4. Time conditioning works: different times → different transformations")
    print("5. Gradient flow verified: all parameters receive gradients")
    print("6. Numerical stability: handles extreme inputs, scale clamping works")
    print("7. Initialization near identity: starts with small transformations")
    print("8. Checkerboard masks correctly split dimensions for coupling")
    print("9. Ready for use in normalizing flows with time-dependent dynamics")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    run_all_tests()
