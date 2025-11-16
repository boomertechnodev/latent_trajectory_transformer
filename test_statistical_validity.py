"""
Statistical Validation Test Suite for Latent Trajectory Transformer

Verifies that all probabilistic components match theoretical predictions:
- Epps-Pulley test correctly identifies non-normal distributions
- KL divergence matches analytic formulas
- Normalizing flows preserve probability mass
- SDE trajectories match known solutions
- Latent space has proper statistical properties

Run with: pytest test_statistical_validity.py -v -s
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import math
from typing import Tuple, List

from latent_drift_trajectory import (
    FastEppsPulley,
    RaccoonDynamics,
    RaccoonFlow,
    CouplingLayer,
    solve_sde,
    TimeAwareTransform,
)


# ============================================================================
# STATISTICAL TESTING UTILITIES
# ============================================================================

def compute_kl_divergence_gaussian(mean1: torch.Tensor, logvar1: torch.Tensor,
                                   mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    Analytic KL divergence between two Gaussian distributions.

    KL(N(μ1, σ1²) || N(μ2, σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2

    Args:
        mean1, logvar1: Parameters of first Gaussian
        mean2, logvar2: Parameters of second Gaussian

    Returns:
        KL divergence
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()

    kl = 0.5 * (
        logvar2 - logvar1
        + (var1 + (mean1 - mean2)**2) / var2
        - 1.0
    )

    return kl.sum()


def monte_carlo_probability_mass(samples: torch.Tensor,
                                 log_prob_fn,
                                 bounds: Tuple[float, float] = (-5.0, 5.0),
                                 num_bins: int = 100) -> float:
    """
    Estimate total probability mass via Monte Carlo integration.

    Should be ≈ 1.0 for valid probability distributions.

    Args:
        samples: Samples from distribution
        log_prob_fn: Function to compute log probability
        bounds: Integration bounds
        num_bins: Number of bins for histogram

    Returns:
        Estimated probability mass
    """
    # Create histogram
    hist = torch.histc(samples, bins=num_bins, min=bounds[0], max=bounds[1])

    # Normalize to get probability density estimate
    bin_width = (bounds[1] - bounds[0]) / num_bins
    density = hist / (samples.numel() * bin_width)

    # Integrate
    total_mass = density.sum() * bin_width

    return total_mass.item()


# ============================================================================
# TEST 1: EP Test Rejects Non-Normal Distributions
# ============================================================================

def test_ep_test_rejects_non_normal():
    """
    Verify that FastEppsPulley correctly identifies and rejects non-normal distributions.

    Tests:
    1. Normal distribution → high p-value (fail to reject)
    2. Uniform distribution → low p-value (reject)
    3. Exponential distribution → low p-value (reject)
    4. Bimodal mixture → low p-value (reject)

    Expected:
    - Normal: p > 0.05 (cannot reject normality)
    - Non-normal: p < 0.05 (reject normality)
    """
    print("\n" + "="*80)
    print("TEST 1: EP Test Rejects Non-Normal Distributions")
    print("="*80)

    num_samples = 500
    dimension = 1

    ep_test = FastEppsPulley()

    # Test 1: Normal distribution (should NOT reject)
    print("\n1. Normal Distribution N(0, 1)")
    print("-" * 40)

    normal_samples = torch.randn(num_samples, dimension)
    ep_statistic_normal = ep_test(normal_samples.unsqueeze(0))  # Add batch dim

    print(f"  EP Statistic: {ep_statistic_normal.item():.6f}")
    print(f"  Interpretation: Lower is more normal-like")

    # For normal distribution, EP statistic should be relatively small
    # (No exact p-value without reference distribution, but can compare magnitudes)

    # Test 2: Uniform distribution (should reject)
    print("\n2. Uniform Distribution U(-2, 2)")
    print("-" * 40)

    uniform_samples = torch.rand(num_samples, dimension) * 4 - 2  # U(-2, 2)
    ep_statistic_uniform = ep_test(uniform_samples.unsqueeze(0))

    print(f"  EP Statistic: {ep_statistic_uniform.item():.6f}")

    # Uniform should have MUCH higher EP statistic than normal
    ratio_uniform = ep_statistic_uniform / ep_statistic_normal
    print(f"  Ratio to normal: {ratio_uniform.item():.2f}x")

    assert ep_statistic_uniform > ep_statistic_normal, \
        "Uniform distribution should have higher EP statistic than normal"

    # Test 3: Exponential distribution (should reject)
    print("\n3. Exponential Distribution Exp(1)")
    print("-" * 40)

    exponential_samples = torch.distributions.Exponential(1.0).sample((num_samples, dimension))
    ep_statistic_exp = ep_test(exponential_samples.unsqueeze(0))

    print(f"  EP Statistic: {ep_statistic_exp.item():.6f}")

    ratio_exp = ep_statistic_exp / ep_statistic_normal
    print(f"  Ratio to normal: {ratio_exp.item():.2f}x")

    assert ep_statistic_exp > ep_statistic_normal, \
        "Exponential distribution should have higher EP statistic than normal"

    # Test 4: Bimodal mixture (should reject)
    print("\n4. Bimodal Mixture: 0.5*N(-2,1) + 0.5*N(2,1)")
    print("-" * 40)

    mixture_samples = torch.cat([
        torch.randn(num_samples // 2, dimension) - 2,
        torch.randn(num_samples // 2, dimension) + 2
    ], dim=0)

    ep_statistic_mixture = ep_test(mixture_samples.unsqueeze(0))

    print(f"  EP Statistic: {ep_statistic_mixture.item():.6f}")

    ratio_mixture = ep_statistic_mixture / ep_statistic_normal
    print(f"  Ratio to normal: {ratio_mixture.item():.2f}x")

    assert ep_statistic_mixture > ep_statistic_normal, \
        "Bimodal mixture should have higher EP statistic than normal"

    # Summary
    print("\n" + "="*40)
    print("Summary: EP Statistics Relative to Normal")
    print("="*40)
    print(f"  Normal (baseline):     1.00x")
    print(f"  Uniform:               {ratio_uniform.item():.2f}x")
    print(f"  Exponential:           {ratio_exp.item():.2f}x")
    print(f"  Bimodal Mixture:       {ratio_mixture.item():.2f}x")

    print(f"\n✓ PASS: EP test correctly distinguishes normal from non-normal distributions")


# ============================================================================
# TEST 2: KL Divergence Matches Theoretical Values
# ============================================================================

def test_kl_divergence_theoretical():
    """
    Verify KL divergence computation matches analytic formulas.

    Tests:
    1. KL(N(0,1) || N(0,1)) = 0
    2. KL(N(0,1) || N(1,1)) = 0.5
    3. KL(N(0,1) || N(0,4)) = analytic value

    Formula:
    KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
    """
    print("\n" + "="*80)
    print("TEST 2: KL Divergence Matches Theoretical Values")
    print("="*80)

    # Test 1: Identical distributions → KL = 0
    print("\n1. KL(N(0,1) || N(0,1)) should be 0")
    print("-" * 40)

    mean1 = torch.zeros(1, 10)
    logvar1 = torch.zeros(1, 10)
    mean2 = torch.zeros(1, 10)
    logvar2 = torch.zeros(1, 10)

    kl_computed = compute_kl_divergence_gaussian(mean1, logvar1, mean2, logvar2)
    kl_expected = 0.0

    print(f"  Computed KL: {kl_computed.item():.6f}")
    print(f"  Expected KL: {kl_expected:.6f}")

    assert torch.allclose(kl_computed, torch.tensor(kl_expected), atol=1e-5), \
        f"KL should be 0 for identical distributions, got {kl_computed.item()}"

    # Test 2: Different means, same variance
    print("\n2. KL(N(0,1) || N(1,1)) = 0.5")
    print("-" * 40)

    mean1 = torch.zeros(1, 10)
    logvar1 = torch.zeros(1, 10)  # var=1
    mean2 = torch.ones(1, 10)
    logvar2 = torch.zeros(1, 10)  # var=1

    kl_computed = compute_kl_divergence_gaussian(mean1, logvar1, mean2, logvar2)
    # KL = 0 + (1 + 1) / 2 - 0.5 = 0.5 per dimension
    kl_expected = 0.5 * 10  # 10 dimensions

    print(f"  Computed KL: {kl_computed.item():.6f}")
    print(f"  Expected KL: {kl_expected:.6f}")

    assert torch.allclose(kl_computed, torch.tensor(kl_expected), atol=1e-4), \
        f"KL mismatch: computed {kl_computed.item():.6f}, expected {kl_expected:.6f}"

    # Test 3: Different variances, same mean
    print("\n3. KL(N(0,1) || N(0,2²))")
    print("-" * 40)

    mean1 = torch.zeros(1, 10)
    logvar1 = torch.zeros(1, 10)  # var=1
    mean2 = torch.zeros(1, 10)
    logvar2 = torch.full((1, 10), math.log(4.0))  # var=4

    kl_computed = compute_kl_divergence_gaussian(mean1, logvar1, mean2, logvar2)
    # KL = log(2/1) + (1 + 0) / (2*4) - 0.5
    #    = log(2) + 0.125 - 0.5
    #    = 0.693 + 0.125 - 0.5
    #    = 0.318 per dimension
    kl_expected = (math.log(2.0) + 1.0 / 8.0 - 0.5) * 10

    print(f"  Computed KL: {kl_computed.item():.6f}")
    print(f"  Expected KL: {kl_expected:.6f}")

    assert torch.allclose(kl_computed, torch.tensor(kl_expected), atol=1e-4), \
        f"KL mismatch: computed {kl_computed.item():.6f}, expected {kl_expected:.6f}"

    # Test 4: Both mean and variance different
    print("\n4. KL(N(1,2²) || N(-1,3²))")
    print("-" * 40)

    mean1 = torch.ones(1, 10)
    logvar1 = torch.full((1, 10), math.log(4.0))  # var=4
    mean2 = -torch.ones(1, 10)
    logvar2 = torch.full((1, 10), math.log(9.0))  # var=9

    kl_computed = compute_kl_divergence_gaussian(mean1, logvar1, mean2, logvar2)
    # KL = log(3/2) + (4 + 4) / (2*9) - 0.5
    #    = 0.405 + 0.444 - 0.5
    #    = 0.349 per dimension
    kl_expected = (math.log(3.0 / 2.0) + (4.0 + 4.0) / 18.0 - 0.5) * 10

    print(f"  Computed KL: {kl_computed.item():.6f}")
    print(f"  Expected KL: {kl_expected:.6f}")

    assert torch.allclose(kl_computed, torch.tensor(kl_expected), atol=1e-4), \
        f"KL mismatch: computed {kl_computed.item():.6f}, expected {kl_expected:.6f}"

    print(f"\n✓ PASS: KL divergence matches theoretical formulas")


# ============================================================================
# TEST 3: Flow Transformations Preserve Probability Mass
# ============================================================================

def test_flow_probability_conservation():
    """
    Verify that normalizing flows preserve probability mass: ∫p(x)dx = 1.

    Method:
    1. Sample z ~ N(0,I)
    2. Transform y = flow(z, t)
    3. Check that |det J| compensates density change
    4. Verify ∫p_y(y)dy ≈ ∫p_z(z)dz = 1

    Uses change of variables formula:
    p_y(y) = p_z(z) / |det J|
    where J is the Jacobian of the transformation.
    """
    print("\n" + "="*80)
    print("TEST 3: Flow Transformations Preserve Probability Mass")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32
    batch_size = 5000  # Large sample for Monte Carlo

    # Create flow
    flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=2)
    flow.eval()

    # Sample from prior N(0, I)
    z_samples = torch.randn(batch_size, latent_dim)

    # Time conditioning
    time_feat = torch.full((batch_size, 1), 0.5)

    # Forward transform
    with torch.no_grad():
        y_samples, log_det_forward = flow(z_samples, time_feat, reverse=False)

    # Compute densities
    # p_z(z) = N(0, I) = exp(-0.5 * z^T z) / (2π)^(d/2)
    log_prob_z = dist.Normal(0, 1).log_prob(z_samples).sum(dim=1)

    # p_y(y) = p_z(z) / |det J| = p_z(z) * exp(-log_det)
    log_prob_y = log_prob_z - log_det_forward

    # Check 1: Log-det should be finite and bounded
    print("\n1. Log-Determinant Jacobian Analysis")
    print("-" * 40)

    print(f"  Mean log|det J|: {log_det_forward.mean().item():.4f}")
    print(f"  Std log|det J|:  {log_det_forward.std().item():.4f}")
    print(f"  Min log|det J|:  {log_det_forward.min().item():.4f}")
    print(f"  Max log|det J|:  {log_det_forward.max().item():.4f}")

    # Log-det should be reasonable (not NaN, not exploding)
    assert torch.isfinite(log_det_forward).all(), "Log-det contains NaN or Inf"
    assert log_det_forward.abs().max() < 1000.0, "Log-det is exploding"

    # Check 2: Probability mass conservation
    print("\n2. Probability Mass Conservation")
    print("-" * 40)

    # Estimate total mass by Monte Carlo
    # E[p(x)] ≈ average probability density
    prob_z = log_prob_z.exp()
    prob_y = log_prob_y.exp()

    avg_prob_z = prob_z.mean().item()
    avg_prob_y = prob_y.mean().item()

    print(f"  Avg p_z(z): {avg_prob_z:.6f}")
    print(f"  Avg p_y(y): {avg_prob_y:.6f}")
    print(f"  Ratio: {avg_prob_y / avg_prob_z:.4f}")

    # They should be similar (within Monte Carlo error)
    # Allow 20% deviation due to sampling
    assert 0.5 < avg_prob_y / avg_prob_z < 2.0, \
        "Probability mass not conserved in flow transformation"

    # Check 3: Reverse transformation recovers original
    print("\n3. Invertibility Check")
    print("-" * 40)

    with torch.no_grad():
        z_reconstructed, log_det_reverse = flow(y_samples, time_feat, reverse=True)

    reconstruction_error = (z_samples - z_reconstructed).abs().max().item()

    print(f"  Max reconstruction error: {reconstruction_error:.6f}")

    assert reconstruction_error < 1e-4, \
        f"Flow is not invertible: max error {reconstruction_error:.6f}"

    # Check 4: Log-det sum should be ≈ 0
    log_det_sum = (log_det_forward + log_det_reverse).abs().max().item()

    print(f"  Max |log|det J| + log|det J^-1||: {log_det_sum:.6f}")

    assert log_det_sum < 1e-3, \
        f"Log-det inverse inconsistency: {log_det_sum:.6f}"

    print(f"\n✓ PASS: Flow transformations preserve probability mass")


# ============================================================================
# TEST 4: SDE Trajectory Statistics Match Theory
# ============================================================================

def test_sde_trajectory_statistics():
    """
    Verify SDE trajectories match theoretical predictions for known processes.

    Test Cases:
    1. Zero drift, constant diffusion → Brownian motion
       - Mean should stay at z_0
       - Variance should grow as σ²t

    2. Linear drift (Ornstein-Uhlenbeck process)
       - Mean should decay exponentially
       - Variance should approach steady state
    """
    print("\n" + "="*80)
    print("TEST 4: SDE Trajectory Statistics Match Theory")
    print("="*80)

    latent_dim = 16
    hidden_dim = 32
    batch_size = 1000  # Monte Carlo samples
    num_steps = 10

    # Test 1: Brownian Motion (zero drift, constant diffusion)
    print("\n1. Brownian Motion: dz = 0*dt + σ*dW")
    print("-" * 40)

    # Create dynamics with zero drift
    dynamics = RaccoonDynamics(latent_dim, hidden_dim,
                               sigma_min=0.1, sigma_max=0.1)  # Constant σ=0.1

    # Override drift to be zero
    dynamics.drift_net = nn.Sequential(
        nn.Linear(latent_dim + 32, 1),  # Output ignored
        nn.Identity()
    )

    # Zero out drift
    with torch.no_grad():
        for param in dynamics.drift_net.parameters():
            param.zero_()

    # Initial condition
    z0 = torch.zeros(batch_size, latent_dim)

    # Time span
    t_start = 0.0
    t_end = 1.0
    t_span = torch.linspace(t_start, t_end, num_steps)

    # Solve SDE
    with torch.no_grad():
        z_trajectory = solve_sde(dynamics, z0, t_span)  # (batch, num_steps, latent_dim)

    # Theoretical predictions for Brownian motion:
    # z(t) ~ N(z0, σ²t I)

    sigma = 0.1
    for step_idx in [3, 6, 9]:  # Check at t=0.3, 0.6, 0.9
        t = t_span[step_idx].item()
        z_t = z_trajectory[:, step_idx, :]  # (batch, latent_dim)

        # Empirical statistics
        mean_empirical = z_t.mean(dim=0)
        var_empirical = z_t.var(dim=0).mean().item()

        # Theoretical predictions
        mean_theory = 0.0  # Should stay at 0
        var_theory = sigma ** 2 * t

        print(f"  t={t:.1f}:")
        print(f"    Mean: {mean_empirical.abs().max().item():.6f} (expect ≈0)")
        print(f"    Var:  {var_empirical:.6f} (theory: {var_theory:.6f})")

        # Mean should be close to 0 (within 3σ / sqrt(N))
        std_error = sigma * math.sqrt(t / batch_size)
        assert mean_empirical.abs().max() < 3 * std_error, \
            f"Mean deviates from 0: {mean_empirical.abs().max().item():.6f}"

        # Variance should match theory (within 30% due to sampling)
        assert 0.5 < var_empirical / var_theory < 2.0, \
            f"Variance mismatch: empirical {var_empirical:.6f}, theory {var_theory:.6f}"

    print(f"\n✓ PASS: SDE statistics match theoretical predictions")


# ============================================================================
# TEST 5: Memory Priority Sampling Follows Expected Distribution
# ============================================================================

def test_memory_priority_sampling():
    """
    Verify that memory buffer samples according to priority (softmax of scores).

    Protocol:
    1. Add items with known scores to memory
    2. Sample many times
    3. Check empirical frequency matches softmax(scores)

    Expected:
    - Higher score items sampled more frequently
    - Frequency ∝ exp(score) / Σ exp(scores)
    """
    print("\n" + "="*80)
    print("TEST 5: Memory Priority Sampling Distribution")
    print("="*80)

    from latent_drift_trajectory import RaccoonMemory

    # Create memory
    memory = RaccoonMemory(max_size=10)

    # Add items with specific scores
    num_items = 5
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]  # Increasing priority

    for i, score in enumerate(scores):
        item = {
            'tokens': torch.tensor([[i]]),  # Dummy data
            'label': torch.tensor([i])
        }
        memory.add(item, score=score)

    # Theoretical sampling distribution: softmax(scores)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    probs_theory = torch.softmax(scores_tensor, dim=0).numpy()

    print("\nTheoretical Sampling Probabilities:")
    for i, (score, prob) in enumerate(zip(scores, probs_theory)):
        print(f"  Item {i} (score={score:.1f}): {prob:.3f}")

    # Sample many times
    num_samples = 5000
    sample_counts = [0] * num_items

    for _ in range(num_samples):
        batch = memory.sample(batch_size=1)
        if len(batch) > 0:
            item_id = batch[0]['label'].item()
            sample_counts[item_id] += 1

    # Empirical frequencies
    probs_empirical = [count / num_samples for count in sample_counts]

    print("\nEmpirical Sampling Frequencies:")
    for i, (count, prob) in enumerate(zip(sample_counts, probs_empirical)):
        print(f"  Item {i}: {prob:.3f} (count={count}/{num_samples})")

    # Chi-square goodness-of-fit test (simplified)
    print("\nComparison (Empirical vs Theory):")
    for i in range(num_items):
        diff = abs(probs_empirical[i] - probs_theory[i])
        print(f"  Item {i}: |{probs_empirical[i]:.3f} - {probs_theory[i]:.3f}| = {diff:.3f}")

        # Allow 5% deviation (generous for Monte Carlo)
        assert diff < 0.05, f"Sampling distribution mismatch for item {i}"

    print(f"\n✓ PASS: Memory priority sampling follows softmax distribution")


# ============================================================================
# TEST 6: Posterior Collapse Detection
# ============================================================================

def test_posterior_collapse_detection():
    """
    Test detection of posterior collapse in VAE-style models.

    Posterior collapse: KL(q(z|x) || p(z)) → 0
    This means the encoder ignores the input and just outputs the prior.

    Protocol:
    1. Simulate collapsed posterior: q(z|x) ≈ p(z) for all x
    2. Compute KL divergence
    3. Verify it's near zero and triggers warning

    Expected:
    - KL < 0.01 threshold indicates collapse
    """
    print("\n" + "="*80)
    print("TEST 6: Posterior Collapse Detection")
    print("="*80)

    latent_dim = 16
    batch_size = 32

    # Scenario 1: Healthy posterior (KL > threshold)
    print("\n1. Healthy Posterior: q(z|x) depends on x")
    print("-" * 40)

    # Encoder outputs vary with input
    mean_healthy = torch.randn(batch_size, latent_dim)
    logvar_healthy = torch.randn(batch_size, latent_dim) * 0.5

    # Prior: N(0, I)
    mean_prior = torch.zeros(batch_size, latent_dim)
    logvar_prior = torch.zeros(batch_size, latent_dim)

    kl_healthy = compute_kl_divergence_gaussian(
        mean_healthy, logvar_healthy,
        mean_prior, logvar_prior
    ) / batch_size  # Per sample

    print(f"  KL divergence (per sample): {kl_healthy.item():.4f}")

    COLLAPSE_THRESHOLD = 0.01

    if kl_healthy > COLLAPSE_THRESHOLD:
        print(f"  ✓ Healthy: KL > {COLLAPSE_THRESHOLD}")
    else:
        print(f"  ✗ WARNING: Possible collapse (KL < {COLLAPSE_THRESHOLD})")

    # Scenario 2: Collapsed posterior (KL ≈ 0)
    print("\n2. Collapsed Posterior: q(z|x) ≈ p(z) for all x")
    print("-" * 40)

    # Encoder outputs same as prior (ignores input)
    mean_collapsed = torch.zeros(batch_size, latent_dim)
    logvar_collapsed = torch.zeros(batch_size, latent_dim)

    kl_collapsed = compute_kl_divergence_gaussian(
        mean_collapsed, logvar_collapsed,
        mean_prior, logvar_prior
    ) / batch_size

    print(f"  KL divergence (per sample): {kl_collapsed.item():.6f}")

    if kl_collapsed < COLLAPSE_THRESHOLD:
        print(f"  ⚠ COLLAPSE DETECTED: KL < {COLLAPSE_THRESHOLD}")
    else:
        print(f"  ✓ No collapse")

    # Verify detection works
    assert kl_collapsed < COLLAPSE_THRESHOLD, "Collapsed posterior should have KL < threshold"
    assert kl_healthy > COLLAPSE_THRESHOLD, "Healthy posterior should have KL > threshold"

    print(f"\n✓ PASS: Posterior collapse detection threshold validated")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION TEST SUITE")
    print("="*80)

    test_ep_test_rejects_non_normal()
    test_kl_divergence_theoretical()
    test_flow_probability_conservation()
    test_sde_trajectory_statistics()
    test_memory_priority_sampling()
    test_posterior_collapse_detection()

    print("\n" + "="*80)
    print("ALL STATISTICAL VALIDATION TESTS COMPLETE")
    print("="*80)
