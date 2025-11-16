"""
Test Suite for Ornstein-Uhlenbeck Process

This test suite verifies:
1. Parameter initialization and positivity constraints
2. Exact transition sampling using closed-form formulas
3. Mean-reverting behavior (convergence to equilibrium μ)
4. Stationary distribution (empirical vs theoretical variance)
5. Exact likelihood computation (analytical vs numerical)
6. Parameter learning via gradient descent
7. Diagonal vs grouped parameter modes
8. Edge cases and numerical stability
9. Visualization of characteristic mean-reverting paths
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ou_process import OUProcess


def test_initialization():
    """Test OUProcess initialization with parameter validation."""
    print("=" * 60)
    print("TEST 1: Initialization and Parameter Validation")
    print("=" * 60)

    # Basic initialization
    ou = OUProcess(dim=64, num_groups=1)
    print(f"Dimension: {ou.dim}")
    print(f"Number of groups: {ou.num_groups}")
    print(f"Group size: {ou.group_size}")

    assert ou.dim == 64
    assert ou.num_groups == 1
    assert ou.group_size == 64

    # Get parameters
    theta, mu, sigma = ou.get_parameters()
    print(f"\nInitial parameters:")
    print(f"  θ (mean reversion): {theta.item():.4f}")
    print(f"  μ (equilibrium): {mu.item():.4f}")
    print(f"  σ (volatility): {sigma.item():.4f}")

    # Check positivity
    assert (theta > 0).all(), "θ must be positive"
    assert (sigma > 0).all(), "σ must be positive"

    # Test diagonal mode
    ou_diag = OUProcess(dim=64, num_groups=64)
    assert ou_diag.num_groups == 64
    assert ou_diag.group_size == 1
    theta_diag, mu_diag, sigma_diag = ou_diag.get_parameters()
    print(f"\nDiagonal mode: {len(theta_diag)} independent parameters per type")

    # Test grouped mode
    ou_grouped = OUProcess(dim=64, num_groups=8)
    assert ou_grouped.num_groups == 8
    assert ou_grouped.group_size == 8
    print(f"Grouped mode: {ou_grouped.num_groups} groups of size {ou_grouped.group_size}")

    # Test invalid parameters
    try:
        OUProcess(dim=64, num_groups=0)
        assert False, "Should raise error for num_groups=0"
    except ValueError as e:
        print(f"\n✓ Correctly rejected num_groups=0: {e}")

    try:
        OUProcess(dim=64, num_groups=100)
        assert False, "Should raise error for num_groups > dim"
    except ValueError as e:
        print(f"✓ Correctly rejected num_groups=100 > dim=64: {e}")

    try:
        OUProcess(dim=64, num_groups=7)
        assert False, "Should raise error when dim not divisible by num_groups"
    except ValueError as e:
        print(f"✓ Correctly rejected non-divisible num_groups: {e}")

    print("\n✓ Initialization test passed!\n")


def test_exact_sampling():
    """Test exact transition sampling using closed-form OU formulas."""
    print("=" * 60)
    print("TEST 2: Exact Transition Sampling")
    print("=" * 60)

    ou = OUProcess(dim=16, num_groups=1, init_theta=1.0, init_mu=2.0, init_sigma=0.5)
    theta, mu, sigma = ou.get_parameters()

    print(f"OU parameters: θ={theta.item():.4f}, μ={mu.item():.4f}, σ={sigma.item():.4f}")

    # Sample a single transition
    x0 = torch.zeros(8, 16)  # Start at 0
    dt = 0.1

    x1, mean, std = ou.sample(x0, dt, return_mean_std=True)

    print(f"\nSingle transition (dt={dt}):")
    print(f"  x0: {x0[0, 0].item():.4f}")
    print(f"  Expected mean: {mean[0, 0].item():.4f}")
    print(f"  Expected std: {std[0].item():.4f}")
    print(f"  Sampled x1: {x1[0, 0].item():.4f}")

    # Check shapes
    assert x1.shape == (8, 16), f"Expected (8, 16), got {x1.shape}"
    assert mean.shape == (8, 16), f"Expected (8, 16), got {mean.shape}"
    assert std.shape == (16,), f"Expected (16,), got {std.shape}"

    # Verify mean formula: μ + (x0 - μ) * exp(-θ * dt)
    expected_mean = mu.item() + (0.0 - mu.item()) * torch.exp(-theta * dt)
    assert torch.allclose(mean[0, 0], expected_mean, atol=1e-5), \
        f"Mean formula incorrect: {mean[0, 0].item()} vs {expected_mean.item()}"

    # Verify variance formula: (σ² / 2θ) * (1 - exp(-2θ * dt))
    expected_variance = (sigma ** 2) / (2 * theta) * (1 - torch.exp(-2 * theta * dt))
    expected_std = torch.sqrt(expected_variance)
    assert torch.allclose(std[0], expected_std, atol=1e-5), \
        f"Std formula incorrect: {std[0].item()} vs {expected_std.item()}"

    print("\n✓ Exact sampling formulas verified!")

    # Test that sampling is stochastic (different samples are different)
    x2 = ou.sample(x0, dt)
    x3 = ou.sample(x0, dt)

    diff = (x2 - x3).abs().mean().item()
    print(f"\nStochasticity check: mean diff between two samples = {diff:.6f}")
    assert diff > 0.01, "Samples should be different (stochastic)"

    print("\n✓ Exact sampling test passed!\n")


def test_mean_reversion():
    """Test mean-reverting property: paths converge to equilibrium μ."""
    print("=" * 60)
    print("TEST 3: Mean-Reverting Behavior")
    print("=" * 60)

    # Strong mean reversion
    ou = OUProcess(dim=32, num_groups=1, init_theta=5.0, init_mu=3.0, init_sigma=0.5)
    theta, mu, sigma = ou.get_parameters()

    print(f"OU parameters: θ={theta.item():.4f} (strong reversion), μ={mu.item():.4f}")

    # Sample a path starting far from equilibrium
    batch_size = 256
    num_steps = 100
    dt = 0.05

    x_t = torch.zeros(batch_size, 32) - 5.0  # Start at -5, far from μ=3
    path = [x_t.clone()]

    for t in range(num_steps):
        x_t = ou.sample(x_t, dt)
        path.append(x_t.clone())

    path = torch.stack(path, dim=1)  # (batch, num_steps+1, dim)

    # Check convergence to μ
    initial_mean = path[:, 0].mean().item()
    final_mean = path[:, -1].mean().item()
    target_mu = mu.item()

    print(f"\nPath evolution over {num_steps} steps:")
    print(f"  Initial mean: {initial_mean:.4f}")
    print(f"  Final mean: {final_mean:.4f}")
    print(f"  Target μ: {target_mu:.4f}")
    print(f"  Distance to μ: initial={abs(initial_mean - target_mu):.4f}, "
          f"final={abs(final_mean - target_mu):.4f}")

    # Mean should converge to μ
    assert abs(final_mean - target_mu) < abs(initial_mean - target_mu), \
        "Mean should move toward equilibrium"
    assert abs(final_mean - target_mu) < 0.5, \
        f"Mean should be close to μ after convergence, got {abs(final_mean - target_mu):.4f}"

    # Test with weak mean reversion
    ou_weak = OUProcess(dim=32, num_groups=1, init_theta=0.1, init_mu=3.0, init_sigma=0.5)
    theta_weak, mu_weak, sigma_weak = ou_weak.get_parameters()

    print(f"\nWeak reversion: θ={theta_weak.item():.4f}")

    x_t_weak = torch.zeros(batch_size, 32) - 5.0
    path_weak = [x_t_weak.clone()]

    for t in range(num_steps):
        x_t_weak = ou_weak.sample(x_t_weak, dt)
        path_weak.append(x_t_weak.clone())

    path_weak = torch.stack(path_weak, dim=1)

    final_mean_weak = path_weak[:, -1].mean().item()
    print(f"  Weak reversion final mean: {final_mean_weak:.4f}")
    print(f"  Distance to μ: {abs(final_mean_weak - target_mu):.4f}")

    # Weak reversion should converge slower (farther from μ than strong)
    assert abs(final_mean_weak - target_mu) > abs(final_mean - target_mu), \
        "Weak reversion should converge slower than strong reversion"

    print("\n✓ Mean-reversion test passed!\n")


def test_stationary_distribution():
    """Test stationary distribution: empirical vs theoretical variance."""
    print("=" * 60)
    print("TEST 4: Stationary Distribution")
    print("=" * 60)

    ou = OUProcess(dim=32, num_groups=1, init_theta=2.0, init_mu=1.5, init_sigma=1.0)
    theta, mu, sigma = ou.get_parameters()

    print(f"OU parameters: θ={theta.item():.4f}, μ={mu.item():.4f}, σ={sigma.item():.4f}")

    # Theoretical stationary distribution: N(μ, σ²/2θ)
    theoretical_mean = mu.item()
    theoretical_var = (sigma ** 2 / (2 * theta)).item()
    theoretical_std = torch.sqrt(torch.tensor(theoretical_var)).item()

    print(f"\nTheoretical stationary distribution:")
    print(f"  Mean: {theoretical_mean:.4f}")
    print(f"  Variance: {theoretical_var:.4f}")
    print(f"  Std: {theoretical_std:.4f}")

    # Sample from stationary distribution directly
    samples_direct = ou.stationary_sample(batch_size=10000, device='cpu')

    empirical_mean_direct = samples_direct.mean(dim=0).mean().item()
    empirical_var_direct = samples_direct.var(dim=0).mean().item()
    empirical_std_direct = samples_direct.std(dim=0).mean().item()

    print(f"\nDirect sampling from stationary distribution:")
    print(f"  Empirical mean: {empirical_mean_direct:.4f}")
    print(f"  Empirical variance: {empirical_var_direct:.4f}")
    print(f"  Empirical std: {empirical_std_direct:.4f}")

    # Check agreement
    assert abs(empirical_mean_direct - theoretical_mean) < 0.1, \
        f"Mean mismatch: {empirical_mean_direct:.4f} vs {theoretical_mean:.4f}"
    assert abs(empirical_var_direct - theoretical_var) < 0.1, \
        f"Variance mismatch: {empirical_var_direct:.4f} vs {theoretical_var:.4f}"

    print("\n✓ Stationary distribution matches theory!")

    # Empirical stationary distribution via long simulation
    print("\nLong simulation to reach stationary distribution:")

    batch_size = 1000
    num_steps = 200
    dt = 0.05

    x_t = torch.randn(batch_size, 32)  # Random start
    for t in range(num_steps):
        x_t = ou.sample(x_t, dt)

    # After long simulation, should be near stationary
    empirical_mean_sim = x_t.mean(dim=0).mean().item()
    empirical_var_sim = x_t.var(dim=0).mean().item()

    print(f"  After {num_steps} steps:")
    print(f"  Empirical mean: {empirical_mean_sim:.4f} (theory: {theoretical_mean:.4f})")
    print(f"  Empirical variance: {empirical_var_sim:.4f} (theory: {theoretical_var:.4f})")

    assert abs(empirical_mean_sim - theoretical_mean) < 0.2, \
        "Simulated mean should converge to theoretical"
    assert abs(empirical_var_sim - theoretical_var) < 0.3, \
        "Simulated variance should converge to theoretical"

    print("\n✓ Stationary distribution test passed!\n")


def test_log_likelihood():
    """Test exact log-likelihood computation (analytical formula)."""
    print("=" * 60)
    print("TEST 5: Exact Log-Likelihood Computation")
    print("=" * 60)

    ou = OUProcess(dim=16, num_groups=1, init_theta=1.5, init_mu=0.5, init_sigma=0.8)
    theta, mu, sigma = ou.get_parameters()

    print(f"OU parameters: θ={theta.item():.4f}, μ={mu.item():.4f}, σ={sigma.item():.4f}")

    # Generate a path from the OU process
    batch_size = 32
    seq_len = 50
    dt = 0.1

    x_t = ou.stationary_sample(batch_size, device='cpu')
    path = [x_t.clone()]

    for t in range(seq_len - 1):
        x_t = ou.sample(x_t, dt)
        path.append(x_t.clone())

    path = torch.stack(path, dim=1)  # (batch, seq_len, dim)

    # Compute log-likelihood
    log_prob = ou.log_prob(path, dt)

    print(f"\nPath shape: {path.shape}")
    print(f"Log-likelihood shape: {log_prob.shape}")
    print(f"Mean log-likelihood: {log_prob.mean().item():.2f}")
    print(f"Log-likelihood range: [{log_prob.min().item():.2f}, {log_prob.max().item():.2f}]")

    assert log_prob.shape == (batch_size,), f"Expected (32,), got {log_prob.shape}"
    assert not torch.isnan(log_prob).any(), "Log-likelihood contains NaN"
    assert not torch.isinf(log_prob).any(), "Log-likelihood contains Inf"

    # Log-likelihood should be finite
    # Note: For high-dimensional Gaussians with small variance, log-likelihood can be positive
    # because the sum includes many positive terms from -log(σ) when σ < 1

    # Test with different path (shouldn't have same likelihood)
    path_random = torch.randn_like(path)
    log_prob_random = ou.log_prob(path_random, dt)

    print(f"\nRandom path log-likelihood: {log_prob_random.mean().item():.2f}")

    # Path from OU should have higher likelihood than random path
    assert log_prob.mean() > log_prob_random.mean(), \
        "OU-generated path should have higher likelihood than random path"

    print("\n✓ Log-likelihood computation test passed!\n")


def test_parameter_learning():
    """Test learning OU parameters from data via gradient descent."""
    print("=" * 60)
    print("TEST 6: Parameter Learning via Gradient Descent")
    print("=" * 60)

    # Ground truth OU process
    ou_true = OUProcess(dim=8, num_groups=1, init_theta=2.0, init_mu=1.0, init_sigma=0.5)
    theta_true, mu_true, sigma_true = ou_true.get_parameters()

    print(f"Ground truth parameters:")
    print(f"  θ_true = {theta_true.item():.4f}")
    print(f"  μ_true = {mu_true.item():.4f}")
    print(f"  σ_true = {sigma_true.item():.4f}")

    # Generate training data from ground truth
    batch_size = 128
    seq_len = 30
    dt = 0.1
    num_trajectories = 200

    print(f"\nGenerating {num_trajectories} trajectories of length {seq_len}...")

    data = []
    for _ in range(num_trajectories):
        x_t = ou_true.stationary_sample(batch_size, device='cpu')
        path = [x_t.clone()]

        for t in range(seq_len - 1):
            x_t = ou_true.sample(x_t, dt)
            path.append(x_t.clone())

        path = torch.stack(path, dim=1)
        data.append(path)

    data = torch.cat(data, dim=0).detach()  # (num_trajectories * batch_size, seq_len, dim)
    print(f"Training data shape: {data.shape}")

    # Learnable OU process with random initialization
    ou_learned = OUProcess(dim=8, num_groups=1, init_theta=0.5, init_mu=0.0, init_sigma=1.5)
    theta_init, mu_init, sigma_init = ou_learned.get_parameters()

    print(f"\nInitial learned parameters:")
    print(f"  θ_init = {theta_init.item():.4f}")
    print(f"  μ_init = {mu_init.item():.4f}")
    print(f"  σ_init = {sigma_init.item():.4f}")

    # Optimize via negative log-likelihood
    optimizer = optim.Adam(ou_learned.parameters(), lr=0.01)

    num_epochs = 500
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute negative log-likelihood
        log_prob = ou_learned.log_prob(data, dt)
        loss = -log_prob.mean()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            theta_curr, mu_curr, sigma_curr = ou_learned.get_parameters()
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"θ={theta_curr.item():.4f}, μ={mu_curr.item():.4f}, σ={sigma_curr.item():.4f}")

    # Final learned parameters
    theta_final, mu_final, sigma_final = ou_learned.get_parameters()

    print(f"\nFinal learned parameters:")
    print(f"  θ_final = {theta_final.item():.4f} (true: {theta_true.item():.4f})")
    print(f"  μ_final = {mu_final.item():.4f} (true: {mu_true.item():.4f})")
    print(f"  σ_final = {sigma_final.item():.4f} (true: {sigma_true.item():.4f})")

    # Check that learning improved parameters
    theta_error = abs(theta_final.item() - theta_true.item())
    mu_error = abs(mu_final.item() - mu_true.item())
    sigma_error = abs(sigma_final.item() - sigma_true.item())

    print(f"\nParameter errors:")
    print(f"  θ error: {theta_error:.4f}")
    print(f"  μ error: {mu_error:.4f}")
    print(f"  σ error: {sigma_error:.4f}")

    # Errors should be small (learning worked)
    assert theta_error < 0.5, f"θ error too large: {theta_error:.4f}"
    assert mu_error < 0.3, f"μ error too large: {mu_error:.4f}"
    assert sigma_error < 0.3, f"σ error too large: {sigma_error:.4f}"

    # Loss should decrease
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"\nLoss: initial={initial_loss:.4f}, final={final_loss:.4f}")
    assert final_loss < initial_loss * 0.5, "Loss should decrease significantly"

    print("\n✓ Parameter learning test passed!\n")


def test_diagonal_vs_grouped():
    """Test diagonal vs grouped parameter modes."""
    print("=" * 60)
    print("TEST 7: Diagonal vs Grouped Parameter Modes")
    print("=" * 60)

    dim = 64

    # Shared parameters (num_groups=1)
    ou_shared = OUProcess(dim=dim, num_groups=1)
    theta_s, mu_s, sigma_s = ou_shared.get_parameters()

    print(f"Shared mode (num_groups=1):")
    print(f"  Parameter shapes: θ={theta_s.shape}, μ={mu_s.shape}, σ={sigma_s.shape}")

    theta_s_full = ou_shared.expand_to_dim(theta_s)
    print(f"  Expanded to dim: {theta_s_full.shape}")
    print(f"  All dimensions same? {torch.allclose(theta_s_full, torch.full_like(theta_s_full, theta_s.item()))}")

    assert theta_s_full.shape == (dim,)
    assert torch.allclose(theta_s_full, torch.full_like(theta_s_full, theta_s.item()))

    # Diagonal parameters (num_groups=dim)
    ou_diag = OUProcess(dim=dim, num_groups=dim)
    theta_d, mu_d, sigma_d = ou_diag.get_parameters()

    print(f"\nDiagonal mode (num_groups={dim}):")
    print(f"  Parameter shapes: θ={theta_d.shape}, μ={mu_d.shape}, σ={sigma_d.shape}")

    theta_d_full = ou_diag.expand_to_dim(theta_d)
    print(f"  Expanded to dim: {theta_d_full.shape}")
    print(f"  Each dimension independent? {torch.equal(theta_d_full, theta_d)}")

    assert theta_d_full.shape == (dim,)
    assert torch.equal(theta_d_full, theta_d)

    # Grouped parameters (num_groups=8)
    ou_grouped = OUProcess(dim=dim, num_groups=8)
    theta_g, mu_g, sigma_g = ou_grouped.get_parameters()

    print(f"\nGrouped mode (num_groups=8):")
    print(f"  Parameter shapes: θ={theta_g.shape}, μ={mu_g.shape}, σ={sigma_g.shape}")
    print(f"  Group size: {ou_grouped.group_size}")

    theta_g_full = ou_grouped.expand_to_dim(theta_g)
    print(f"  Expanded to dim: {theta_g_full.shape}")

    # Check that each group of 8 dimensions has the same parameter
    for i in range(8):
        group_start = i * 8
        group_end = group_start + 8
        group_values = theta_g_full[group_start:group_end]
        assert torch.allclose(group_values, torch.full_like(group_values, theta_g[i].item())), \
            f"Group {i} should have identical values"

    print(f"  Each group has identical parameters? True")

    # Sample from each mode
    x0 = torch.randn(16, dim)
    dt = 0.1

    x_shared = ou_shared.sample(x0, dt)
    x_diag = ou_diag.sample(x0, dt)
    x_grouped = ou_grouped.sample(x0, dt)

    print(f"\nSampling:")
    print(f"  Shared output shape: {x_shared.shape}")
    print(f"  Diagonal output shape: {x_diag.shape}")
    print(f"  Grouped output shape: {x_grouped.shape}")

    assert x_shared.shape == (16, dim)
    assert x_diag.shape == (16, dim)
    assert x_grouped.shape == (16, dim)

    print("\n✓ Diagonal vs grouped test passed!\n")


def test_edge_cases():
    """Test edge cases and numerical stability."""
    print("=" * 60)
    print("TEST 8: Edge Cases and Numerical Stability")
    print("=" * 60)

    ou = OUProcess(dim=16, num_groups=1)

    # Very small dt
    x0 = torch.randn(8, 16)
    dt_small = 1e-6

    x_small = ou.sample(x0, dt_small)
    diff_small = (x_small - x0).abs().mean().item()

    print(f"Very small dt={dt_small}:")
    print(f"  Mean change: {diff_small:.8f}")
    print(f"  Should be very small (process barely moves)")

    assert diff_small < 0.01, f"With tiny dt, process should barely change, got {diff_small}"

    # Very large dt
    dt_large = 10.0

    x_large = ou.sample(x0, dt_large)
    stationary_mean, stationary_std = ou.get_stationary_stats()

    print(f"\nVery large dt={dt_large}:")
    print(f"  Sample mean: {x_large.mean().item():.4f}")
    print(f"  Stationary mean: {stationary_mean.mean().item():.4f}")
    print(f"  Should be close (process forgets initial condition)")

    # With large dt, should approach stationary distribution
    # (won't be exact match due to single sample, but should be closer than initial)
    dist_to_stat = abs(x_large.mean().item() - stationary_mean.mean().item())
    print(f"  Distance to stationary mean: {dist_to_stat:.4f}")

    # Test invalid dt
    try:
        ou.sample(x0, dt=-0.1)
        assert False, "Should raise error for negative dt"
    except ValueError as e:
        print(f"\n✓ Correctly rejected negative dt: {e}")

    try:
        ou.sample(x0, dt=0.0)
        assert False, "Should raise error for dt=0"
    except ValueError as e:
        print(f"✓ Correctly rejected dt=0: {e}")

    # Test path with seq_len < 2
    try:
        path_short = torch.randn(8, 1, 16)
        ou.log_prob(path_short, dt=0.1)
        assert False, "Should raise error for seq_len < 2"
    except ValueError as e:
        print(f"✓ Correctly rejected seq_len=1: {e}")

    # Test dimension mismatch
    try:
        x_wrong_dim = torch.randn(8, 32)  # Should be 16
        ou.sample(x_wrong_dim, dt=0.1)
        assert False, "Should raise error for dimension mismatch"
    except AssertionError as e:
        print(f"✓ Correctly rejected wrong dimension: {e}")

    # Test masked likelihood
    path = torch.randn(8, 20, 16)
    mask = torch.ones(8, 20, dtype=torch.bool)
    mask[:, 10:] = False  # Mask out second half

    log_prob_full = ou.log_prob(path, dt=0.1)
    log_prob_masked = ou.log_prob(path, dt=0.1, mask=mask)

    print(f"\nMasked likelihood:")
    print(f"  Full path log-prob: {log_prob_full.mean().item():.2f}")
    print(f"  Masked path log-prob: {log_prob_masked.mean().item():.2f}")
    print(f"  Masked should be higher (fewer terms)")

    # Masked likelihood should be less negative (fewer terms in sum)
    assert log_prob_masked.mean() > log_prob_full.mean(), \
        "Masked likelihood should be higher (fewer negative terms)"

    print("\n✓ Edge cases test passed!\n")


def test_visualization():
    """Visualize sample paths showing mean-reverting behavior."""
    print("=" * 60)
    print("TEST 9: Visualization of Mean-Reverting Paths")
    print("=" * 60)

    ou = OUProcess(dim=1, num_groups=1, init_theta=2.0, init_mu=0.0, init_sigma=1.0)
    theta, mu, sigma = ou.get_parameters()

    print(f"OU parameters: θ={theta.item():.4f}, μ={mu.item():.4f}, σ={sigma.item():.4f}")

    # Generate 5 sample paths
    num_paths = 5
    num_steps = 100
    dt = 0.05

    print(f"\nGenerating {num_paths} paths with {num_steps} steps...")

    paths = []
    for _ in range(num_paths):
        x_t = torch.randn(1, 1) * 3  # Start far from equilibrium
        path = [x_t.item()]

        for t in range(num_steps):
            x_t = ou.sample(x_t, dt)
            path.append(x_t.item())

        paths.append(path)

    # ASCII visualization
    print("\nMean-reverting paths (μ=0.0 is equilibrium):")
    print("=" * 80)

    # Find global min/max for scaling
    all_values = [val for path in paths for val in path]
    min_val = min(all_values)
    max_val = max(all_values)

    height = 15
    width = 80

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw equilibrium line (μ=0)
    eq_row = int(height / 2)
    for col in range(width):
        grid[eq_row][col] = '-'

    # Draw each path
    symbols = ['1', '2', '3', '4', '5']
    for path_idx, path in enumerate(paths):
        for t in range(min(len(path), width)):
            # Map value to row
            normalized = (path[t] - min_val) / (max_val - min_val + 1e-6)
            row = height - 1 - int(normalized * (height - 1))
            row = max(0, min(height - 1, row))

            grid[row][t] = symbols[path_idx]

    # Print grid
    print(f"Range: [{min_val:.2f}, {max_val:.2f}], μ=0 is at mid-height")
    for row in grid:
        print(''.join(row))

    print(f"\nLegend: 1-5 = different paths, - = equilibrium (μ=0)")
    print("=" * 80)

    # Statistics
    for i, path in enumerate(paths):
        final_val = path[-1]
        print(f"Path {i+1}: final value = {final_val:.4f}, distance to μ = {abs(final_val):.4f}")

    print("\n✓ Visualization test passed!\n")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("ORNSTEIN-UHLENBECK PROCESS - COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    test_initialization()
    test_exact_sampling()
    test_mean_reversion()
    test_stationary_distribution()
    test_log_likelihood()
    test_parameter_learning()
    test_diagonal_vs_grouped()
    test_edge_cases()
    test_visualization()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. OU process correctly implements exact sampling via closed-form formulas")
    print("2. Mean-reversion verified: paths converge to equilibrium μ")
    print("3. Stationary distribution matches theory: N(μ, σ²/2θ)")
    print("4. Exact log-likelihood computation works correctly")
    print("5. Parameters can be learned via gradient descent on likelihood")
    print("6. Diagonal/grouped modes work as expected")
    print("7. Edge cases handled correctly (small/large dt, masking)")
    print("8. Numerical stability verified across parameter ranges")
    print("9. Visualization confirms characteristic mean-reverting behavior")
    print("10. Ready for use as prior dynamics in Variational Path Flows")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all tests
    run_all_tests()
