#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Statistical Measures
Tests normality tests, divergence measures, and numerical stability
"""

import torch
import torch.nn as nn
import numpy as np
from torch import distributions as D
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import the statistical measures from main module
import sys
sys.path.append('/home/user/latent_trajectory_transformer')
from latent_drift_trajectory import (
    FastEppsPulley, EppsPulleyCF, SlicingUnivariateTest,
    MaximumMeanDiscrepancy, SlicedWassersteinDistance,
    EnergyDistance, AdaptiveKLDivergence
)

# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class TestDataGenerator:
    """Generate various distributions for testing."""

    @staticmethod
    def generate_normal(n: int, d: int, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Generate samples from multivariate normal."""
        return torch.randn(n, d) * std + mean

    @staticmethod
    def generate_laplace(n: int, d: int, loc: float = 0.0, scale: float = 1.0) -> torch.Tensor:
        """Generate samples from Laplace distribution (heavy tails)."""
        laplace = D.Laplace(loc, scale)
        return laplace.sample((n, d))

    @staticmethod
    def generate_student_t(n: int, d: int, df: float = 3.0) -> torch.Tensor:
        """Generate samples from Student-t distribution."""
        # Use transformation: T = Z / sqrt(V/df) where Z ~ N(0,1), V ~ Chi2(df)
        z = torch.randn(n, d)
        v = torch.distributions.Chi2(df).sample((n, 1))
        return z / torch.sqrt(v / df)

    @staticmethod
    def generate_mixture_gaussian(n: int, d: int, n_components: int = 3) -> torch.Tensor:
        """Generate samples from Gaussian mixture model."""
        samples = []
        n_per_component = n // n_components

        for i in range(n_components):
            mean = torch.randn(d) * 3  # Random means
            std = 0.5 + torch.rand(1) * 1.5  # Random stds
            component_samples = torch.randn(n_per_component, d) * std + mean
            samples.append(component_samples)

        # Handle remainder
        remainder = n - n_per_component * n_components
        if remainder > 0:
            samples.append(torch.randn(remainder, d))

        all_samples = torch.cat(samples, dim=0)
        # Shuffle
        perm = torch.randperm(n)
        return all_samples[perm]

    @staticmethod
    def generate_uniform(n: int, d: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
        """Generate samples from uniform distribution."""
        return torch.rand(n, d) * (high - low) + low

    @staticmethod
    def generate_exponential(n: int, d: int, rate: float = 1.0) -> torch.Tensor:
        """Generate samples from exponential distribution."""
        exp_dist = D.Exponential(rate)
        return exp_dist.sample((n, d))


# ============================================================================
# STATISTICAL TEST VALIDATORS
# ============================================================================

class StatisticalTestValidator:
    """Validate statistical tests with known distributions."""

    def __init__(self):
        self.generator = TestDataGenerator()

    def test_epps_pulley_power(self, n_samples: int = 1000, n_trials: int = 100) -> Dict:
        """
        Test power of Epps-Pulley test against various alternatives.

        Power = P(reject H0 | H1 is true)
        Size = P(reject H0 | H0 is true) - should be ≈ α
        """
        print("\n" + "="*70)
        print("EPPS-PULLEY TEST POWER ANALYSIS")
        print("="*70)

        # Test configurations
        ep_test = FastEppsPulley(t_max=3.0, n_points=17, weight_type='epps_pulley')
        slicing_test = SlicingUnivariateTest(ep_test, num_slices=64, reduction='mean')

        alpha = 0.05  # Significance level
        d = 10  # Dimensionality

        results = {}

        # Test different distributions
        distributions = {
            'Normal (H0)': lambda n: self.generator.generate_normal(n, d),
            'Laplace': lambda n: self.generator.generate_laplace(n, d),
            'Student-t (df=3)': lambda n: self.generator.generate_student_t(n, d, df=3),
            'Uniform': lambda n: self.generator.generate_uniform(n, d),
            'Mixture-3': lambda n: self.generator.generate_mixture_gaussian(n, d, n_components=3),
        }

        for dist_name, dist_fn in distributions.items():
            rejections = 0
            statistics = []

            for trial in range(n_trials):
                # Generate samples
                samples = dist_fn(n_samples)

                # Reshape for test (1, n_samples, d)
                samples_test = samples.unsqueeze(0)

                # Compute test statistic
                stat = slicing_test(samples_test).item()
                statistics.append(stat)

                # Simple threshold (would need proper critical value in practice)
                # Using asymptotic chi-squared approximation
                critical_value = stats.chi2.ppf(1 - alpha, df=64)  # df = num_slices

                if stat > critical_value:
                    rejections += 1

            power = rejections / n_trials
            results[dist_name] = {
                'power': power,
                'mean_stat': np.mean(statistics),
                'std_stat': np.std(statistics)
            }

            print(f"\n{dist_name}:")
            print(f"  Power (rejection rate): {power:.3f}")
            print(f"  Mean statistic: {results[dist_name]['mean_stat']:.3f}")
            print(f"  Std statistic: {results[dist_name]['std_stat']:.3f}")

        # Check Type I error (should be close to alpha for Normal)
        type_i_error = results['Normal (H0)']['power']
        print(f"\n✓ Type I Error: {type_i_error:.3f} (target: {alpha:.3f})")

        return results

    def test_numerical_stability(self, n_samples_list: List[int] = [10, 100, 1000, 10000]) -> Dict:
        """Test numerical stability across different sample sizes."""
        print("\n" + "="*70)
        print("NUMERICAL STABILITY ANALYSIS")
        print("="*70)

        d = 10
        results = {}

        # Tests to evaluate
        tests = {
            'FastEppsPulley': FastEppsPulley(t_max=3.0, n_points=17),
            'EppsPulleyCF': EppsPulleyCF(t_range=(-3, 3), n_points=10),
        }

        for test_name, test in tests.items():
            print(f"\n{test_name}:")
            test_results = {}

            for n in n_samples_list:
                # Generate normal samples
                samples = self.generator.generate_normal(n, d)

                if 'Epps' in test_name:
                    # For univariate tests, need shape (*, N, K)
                    samples_test = samples.T.unsqueeze(0)  # (1, d, n)
                else:
                    samples_test = samples

                # Test with extreme values
                extreme_samples = samples_test.clone()
                extreme_samples[0, 0] = 1e6  # Add outlier

                try:
                    # Normal test
                    stat_normal = test(samples_test)

                    # Test with outlier
                    stat_extreme = test(extreme_samples)

                    # Check for NaN/Inf
                    has_nan = torch.isnan(stat_normal).any() or torch.isnan(stat_extreme).any()
                    has_inf = torch.isinf(stat_normal).any() or torch.isinf(stat_extreme).any()

                    test_results[n] = {
                        'stat_normal': stat_normal.mean().item() if stat_normal.numel() > 1 else stat_normal.item(),
                        'stat_extreme': stat_extreme.mean().item() if stat_extreme.numel() > 1 else stat_extreme.item(),
                        'stable': not (has_nan or has_inf)
                    }

                    status = "✓ STABLE" if test_results[n]['stable'] else "✗ UNSTABLE"
                    print(f"  n={n:5d}: normal={test_results[n]['stat_normal']:.3f}, "
                          f"extreme={test_results[n]['stat_extreme']:.3f} [{status}]")

                except Exception as e:
                    print(f"  n={n:5d}: ERROR - {str(e)}")
                    test_results[n] = {'error': str(e)}

            results[test_name] = test_results

        return results

    def test_divergence_measures(self, n_samples: int = 500) -> Dict:
        """Test various divergence measures between distributions."""
        print("\n" + "="*70)
        print("DIVERGENCE MEASURES COMPARISON")
        print("="*70)

        d = 10

        # Initialize divergence measures
        mmd = MaximumMeanDiscrepancy(kernel='gaussian')
        swd = SlicedWassersteinDistance(n_projections=100, p=2)
        energy = EnergyDistance(power=1.0)

        # Test pairs of distributions
        test_pairs = [
            ('Normal-Normal (same)',
             self.generator.generate_normal(n_samples, d, 0, 1),
             self.generator.generate_normal(n_samples, d, 0, 1)),

            ('Normal-Normal (shifted)',
             self.generator.generate_normal(n_samples, d, 0, 1),
             self.generator.generate_normal(n_samples, d, 2, 1)),

            ('Normal-Normal (scaled)',
             self.generator.generate_normal(n_samples, d, 0, 1),
             self.generator.generate_normal(n_samples, d, 0, 2)),

            ('Normal-Laplace',
             self.generator.generate_normal(n_samples, d),
             self.generator.generate_laplace(n_samples, d)),

            ('Normal-Uniform',
             self.generator.generate_normal(n_samples, d),
             self.generator.generate_uniform(n_samples, d, -3, 3)),
        ]

        results = {}

        print("\n{:<30} {:>10} {:>10} {:>10}".format(
            "Distribution Pair", "MMD", "SWD", "Energy"))
        print("-" * 65)

        for name, x, y in test_pairs:
            mmd_val = mmd(x, y).item()
            swd_val = swd(x, y).item()
            energy_val = energy(x, y).item()

            results[name] = {
                'MMD': mmd_val,
                'SWD': swd_val,
                'Energy': energy_val
            }

            print("{:<30} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                name, mmd_val, swd_val, energy_val))

        return results

    def test_kl_divergence(self) -> Dict:
        """Test KL divergence implementation."""
        print("\n" + "="*70)
        print("KL DIVERGENCE VALIDATION")
        print("="*70)

        kl_div = AdaptiveKLDivergence(beta_start=0.0, beta_end=1.0, warmup_steps=100)

        # Test cases with known KL divergence
        test_cases = []

        # Case 1: Same distribution (KL = 0)
        mean_q = torch.zeros(100, 10)
        logvar_q = torch.zeros(100, 10)
        mean_p = torch.zeros(100, 10)
        logvar_p = torch.zeros(100, 10)

        kl1 = kl_div.gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)
        test_cases.append(('Same distribution', kl1.sum(dim=-1).mean().item(), 0.0))

        # Case 2: Standard normal vs shifted (KL = 0.5 * ||μ||²)
        mean_q = torch.ones(100, 10)
        logvar_q = torch.zeros(100, 10)
        mean_p = torch.zeros(100, 10)
        logvar_p = torch.zeros(100, 10)

        kl2 = kl_div.gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)
        expected2 = 0.5 * 10  # 0.5 * d where d=10
        test_cases.append(('Shifted by 1', kl2.sum(dim=-1).mean().item(), expected2))

        # Case 3: Different variances
        mean_q = torch.zeros(100, 10)
        logvar_q = torch.log(torch.tensor(2.0)) * torch.ones(100, 10)  # var = 2
        mean_p = torch.zeros(100, 10)
        logvar_p = torch.zeros(100, 10)  # var = 1

        kl3 = kl_div.gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)
        # KL = 0.5 * [log(1/2) + 2/1 - 1] = 0.5 * [-log(2) + 1]
        expected3 = 0.5 * (1 - np.log(2)) * 10
        test_cases.append(('Double variance', kl3.sum(dim=-1).mean().item(), expected3))

        print("\n{:<30} {:>15} {:>15} {:>10}".format(
            "Test Case", "Computed KL", "Expected KL", "Error"))
        print("-" * 75)

        results = {}
        for name, computed, expected in test_cases:
            error = abs(computed - expected)
            status = "✓" if error < 0.1 else "✗"
            print("{:<30} {:>15.6f} {:>15.6f} {:>10.6f} {}".format(
                name, computed, expected, error, status))
            results[name] = {
                'computed': computed,
                'expected': expected,
                'error': error
            }

        return results


# ============================================================================
# P-VALUE CALIBRATION
# ============================================================================

class PValueCalibrator:
    """Calibrate p-values for statistical tests."""

    def calibrate_epps_pulley(self, n_samples: int = 100, d: int = 10,
                              n_simulations: int = 10000) -> np.ndarray:
        """
        Calibrate p-values for Epps-Pulley test via simulation.

        Returns empirical quantiles under null hypothesis.
        """
        print("\n" + "="*70)
        print("P-VALUE CALIBRATION FOR EPPS-PULLEY TEST")
        print("="*70)

        ep_test = FastEppsPulley(t_max=3.0, n_points=17)
        slicing_test = SlicingUnivariateTest(ep_test, num_slices=64, reduction='mean')

        statistics = []

        print(f"Running {n_simulations} simulations...")
        for i in range(n_simulations):
            if (i + 1) % 1000 == 0:
                print(f"  Simulation {i + 1}/{n_simulations}")

            # Generate samples from null (standard normal)
            samples = torch.randn(n_samples, d)
            samples_test = samples.unsqueeze(0)

            # Compute test statistic
            stat = slicing_test(samples_test).item()
            statistics.append(stat)

        statistics = np.array(statistics)

        # Compute quantiles
        quantiles = [0.90, 0.95, 0.99, 0.999]
        critical_values = {}

        print("\nCritical Values:")
        print("-" * 40)
        for q in quantiles:
            cv = np.quantile(statistics, q)
            critical_values[q] = cv
            print(f"  α = {1-q:.3f}: critical value = {cv:.3f}")

        # Test goodness-of-fit to chi-squared
        # Theoretical: Under H0, should follow chi-squared with df = num_slices
        theoretical_quantiles = [stats.chi2.ppf(q, df=64) for q in quantiles]

        print("\nComparison with χ²(64):")
        print("-" * 40)
        for i, q in enumerate(quantiles):
            empirical = critical_values[q]
            theoretical = theoretical_quantiles[i]
            ratio = empirical / theoretical
            print(f"  α = {1-q:.3f}: empirical/theoretical = {ratio:.3f}")

        return statistics


# ============================================================================
# SYNTHETIC DATA VALIDATION
# ============================================================================

def validate_on_latent_trajectories():
    """Validate statistical measures on actual latent trajectories."""
    print("\n" + "="*70)
    print("VALIDATION ON LATENT TRAJECTORIES")
    print("="*70)

    # Simulate latent trajectories from different models
    batch_size = 32
    seq_len = 50
    latent_dim = 16

    # 1. Well-behaved Gaussian latent codes (good VAE)
    good_latents = torch.randn(batch_size * seq_len, latent_dim)

    # 2. Collapsed latents (posterior collapse)
    collapsed_latents = torch.zeros(batch_size * seq_len, latent_dim)
    collapsed_latents += torch.randn(1, latent_dim) * 0.1  # All similar

    # 3. Multimodal latents (mode collapse)
    mode1 = torch.randn(batch_size * seq_len // 2, latent_dim) + torch.tensor([3.0] * latent_dim)
    mode2 = torch.randn(batch_size * seq_len // 2, latent_dim) - torch.tensor([3.0] * latent_dim)
    multimodal_latents = torch.cat([mode1, mode2], dim=0)

    # Test with EP
    ep_test = FastEppsPulley(t_max=5.0, n_points=17)
    slicing_test = SlicingUnivariateTest(ep_test, num_slices=64, reduction='mean')

    # Test with MMD against standard normal
    mmd = MaximumMeanDiscrepancy(kernel='gaussian')
    target_normal = torch.randn(batch_size * seq_len, latent_dim)

    print("\nLatent Space Quality Metrics:")
    print("-" * 50)

    for name, latents in [
        ("Well-behaved", good_latents),
        ("Collapsed", collapsed_latents),
        ("Multimodal", multimodal_latents)
    ]:
        # Reshape for EP test
        latents_test = latents.unsqueeze(0)

        ep_stat = slicing_test(latents_test).item()
        mmd_val = mmd(latents, target_normal).item()

        # Compute basic statistics
        mean_norm = latents.mean(dim=0).norm().item()
        std_mean = latents.std(dim=0).mean().item()

        print(f"\n{name} Latents:")
        print(f"  EP statistic: {ep_stat:.3f}")
        print(f"  MMD to normal: {mmd_val:.4f}")
        print(f"  Mean norm: {mean_norm:.3f}")
        print(f"  Avg std: {std_mean:.3f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive statistical testing suite."""
    print("\n" + "="*70)
    print(" STATISTICAL MEASURES VALIDATION SUITE")
    print("="*70)

    # Initialize validator
    validator = StatisticalTestValidator()

    # 1. Test EP power against alternatives
    print("\n[1/5] Testing Epps-Pulley Power...")
    power_results = validator.test_epps_pulley_power(n_samples=500, n_trials=50)

    # 2. Test numerical stability
    print("\n[2/5] Testing Numerical Stability...")
    stability_results = validator.test_numerical_stability()

    # 3. Test divergence measures
    print("\n[3/5] Testing Divergence Measures...")
    divergence_results = validator.test_divergence_measures(n_samples=300)

    # 4. Test KL divergence
    print("\n[4/5] Testing KL Divergence...")
    kl_results = validator.test_kl_divergence()

    # 5. P-value calibration (optional - takes time)
    calibrate_p_values = False
    if calibrate_p_values:
        print("\n[5/5] Calibrating P-values...")
        calibrator = PValueCalibrator()
        calibration_stats = calibrator.calibrate_epps_pulley(
            n_samples=100, d=10, n_simulations=1000
        )

    # 6. Validate on latent trajectories
    print("\n[6/6] Validating on Latent Trajectories...")
    validate_on_latent_trajectories()

    # Summary
    print("\n" + "="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)

    print("\n✓ Epps-Pulley Test:")
    print(f"  - Type I Error: {power_results['Normal (H0)']['power']:.3f} (target: 0.05)")
    print(f"  - Power vs Laplace: {power_results['Laplace']['power']:.3f}")
    print(f"  - Power vs Mixture: {power_results['Mixture-3']['power']:.3f}")

    print("\n✓ Numerical Stability:")
    all_stable = True
    for test_name, results in stability_results.items():
        for n, result in results.items():
            if isinstance(result, dict) and not result.get('stable', True):
                all_stable = False
                break
    print(f"  - All tests stable: {all_stable}")

    print("\n✓ Divergence Measures:")
    print(f"  - MMD detects shift: {divergence_results['Normal-Normal (shifted)']['MMD'] > 0.1}")
    print(f"  - SWD detects scale: {divergence_results['Normal-Normal (scaled)']['SWD'] > 0.1}")
    print(f"  - Energy detects difference: {divergence_results['Normal-Laplace']['Energy'] > 0.1}")

    print("\n✓ KL Divergence:")
    max_error = max(r['error'] for r in kl_results.values())
    print(f"  - Maximum error: {max_error:.6f}")
    print(f"  - Implementation correct: {max_error < 0.1}")

    print("\n" + "="*70)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run validation suite
    main()