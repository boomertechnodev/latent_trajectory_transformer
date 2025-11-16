# Statistical Testing & Regularization - Advanced Skill Module

## Mathematical Foundations

### Characteristic Function Theory

**Definition and Properties:**

The characteristic function (CF) of a random variable X is:
```
φ_X(t) = E[e^(itX)] = ∫ e^(itx) dF_X(x)
```

Key properties:
1. **Uniqueness**: φ_X completely determines the distribution F_X
2. **Inversion Formula**: F_X can be recovered from φ_X
3. **Convolution**: φ_{X+Y}(t) = φ_X(t)·φ_Y(t) for independent X,Y
4. **Moments**: φ^(n)(0) = i^n E[X^n]

**Epps-Pulley Test Foundation:**

The EP test statistic is based on the integrated squared difference:
```
T_n = n ∫ |φ̂_n(t) - φ_0(t)|² w(t) dt
```

Where:
- φ̂_n(t) = empirical characteristic function
- φ_0(t) = hypothesized CF (e.g., standard normal)
- w(t) = weight function (typically exp(-t²/2))

**Asymptotic Distribution:**
Under H₀: T_n → ∑ᵢ λᵢ Z_i² where Z_i ~ N(0,1) and λᵢ are eigenvalues of the covariance operator.

### Reproducing Kernel Hilbert Space (RKHS) Theory

**Maximum Mean Discrepancy:**

MMD is the RKHS norm of the difference in mean embeddings:
```
MMD²[F,P,Q] = ||μ_P - μ_Q||²_H
            = E_xx'[k(x,x')] + E_yy'[k(y,y')] - 2E_xy[k(x,y)]
```

**Key Results:**
1. **Characteristic Kernels**: k is characteristic iff MMD[F,P,Q]=0 ⟺ P=Q
2. **Gaussian Kernel**: k(x,y) = exp(-||x-y||²/2σ²) is characteristic
3. **Test Power**: Achieves minimax optimal rate for smooth alternatives

### Optimal Transport Theory

**Wasserstein Distance:**

The p-Wasserstein distance between distributions P and Q:
```
W_p(P,Q) = inf_{γ∈Γ(P,Q)} (∫∫ ||x-y||^p dγ(x,y))^(1/p)
```

Where Γ(P,Q) is the set of couplings with marginals P and Q.

**Kantorovich-Rubinstein Duality:**
For p=1:
```
W_1(P,Q) = sup_{||f||_L≤1} |E_P[f(X)] - E_Q[f(Y)]|
```

**Sinkhorn Algorithm:**
Entropy-regularized OT:
```
W_ε(P,Q) = min_{γ∈Γ(P,Q)} ⟨C,γ⟩ + ε·H(γ)
```
Solved via alternating projections.

## Advanced Implementation Techniques

### 1. High-Performance Statistical Tests

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import stats, special
from typing import Optional, Tuple, Dict, Union
import warnings

class ComprehensiveNormalityTest:
    """
    Suite of normality tests with power analysis.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.test_results = {}

    def shapiro_wilk(self, x: np.ndarray) -> Dict:
        """
        Shapiro-Wilk test (best for small samples).
        """
        n = len(x)
        if n < 3:
            return {'statistic': np.nan, 'p_value': np.nan, 'reject': False}

        if n > 5000:
            warnings.warn("Shapiro-Wilk may be inaccurate for n>5000")

        statistic, p_value = stats.shapiro(x)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'reject': p_value < self.alpha,
            'power': self._estimate_power('shapiro', n)
        }

    def anderson_darling(self, x: np.ndarray) -> Dict:
        """
        Anderson-Darling test (sensitive to tails).
        """
        result = stats.anderson(x, dist='norm')
        # Find appropriate critical value
        idx = np.searchsorted(result.significance_level, self.alpha * 100)
        if idx >= len(result.critical_values):
            idx = -1

        return {
            'statistic': result.statistic,
            'critical_value': result.critical_values[idx],
            'reject': result.statistic > result.critical_values[idx],
            'significance_levels': result.significance_level
        }

    def jarque_bera(self, x: np.ndarray) -> Dict:
        """
        Jarque-Bera test (based on skewness and kurtosis).
        """
        n = len(x)
        x_centered = x - x.mean()

        # Moments
        s = np.mean(x_centered**3) / (np.mean(x_centered**2)**(3/2))  # Skewness
        k = np.mean(x_centered**4) / (np.mean(x_centered**2)**2)       # Kurtosis

        # JB statistic
        jb = n * (s**2 / 6 + (k - 3)**2 / 24)

        # Chi-squared p-value with 2 df
        p_value = 1 - stats.chi2.cdf(jb, df=2)

        return {
            'statistic': jb,
            'p_value': p_value,
            'reject': p_value < self.alpha,
            'skewness': s,
            'kurtosis': k
        }

    def dagostino_pearson(self, x: np.ndarray) -> Dict:
        """
        D'Agostino-Pearson omnibus test.
        """
        statistic, p_value = stats.normaltest(x)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'reject': p_value < self.alpha
        }

    def epps_pulley_advanced(
        self,
        x: torch.Tensor,
        n_mc: int = 1000
    ) -> Dict:
        """
        Advanced Epps-Pulley test with Monte Carlo critical values.
        """
        n, d = x.shape

        # Standardize
        x_std = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

        # Test statistic
        ep_test = FastEppsPulley(n_points=25)
        statistic = ep_test.test_statistic(x_std).item()

        # Monte Carlo critical value
        mc_statistics = []
        for _ in range(n_mc):
            z = torch.randn_like(x)
            mc_stat = ep_test.test_statistic(z).item()
            mc_statistics.append(mc_stat)

        critical_value = np.percentile(mc_statistics, (1 - self.alpha) * 100)
        p_value = np.mean(np.array(mc_statistics) >= statistic)

        return {
            'statistic': statistic,
            'critical_value': critical_value,
            'p_value': p_value,
            'reject': p_value < self.alpha,
            'mc_quantiles': np.percentile(mc_statistics, [5, 25, 50, 75, 95])
        }

    def run_all_tests(self, x: Union[np.ndarray, torch.Tensor]) -> Dict:
        """
        Run comprehensive test suite.
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            x_torch = x
        else:
            x_np = x
            x_torch = torch.tensor(x, dtype=torch.float32)

        # Flatten for univariate tests
        if x_np.ndim > 1:
            x_flat = x_np.flatten()
        else:
            x_flat = x_np

        results = {
            'n_samples': len(x_flat),
            'shapiro_wilk': self.shapiro_wilk(x_flat),
            'anderson_darling': self.anderson_darling(x_flat),
            'jarque_bera': self.jarque_bera(x_flat),
            'dagostino_pearson': self.dagostino_pearson(x_flat),
        }

        # Multivariate EP test if applicable
        if x_torch.ndim == 2:
            results['epps_pulley'] = self.epps_pulley_advanced(x_torch)

        # Overall decision (majority vote)
        rejects = [
            results[test]['reject']
            for test in results
            if isinstance(results[test], dict) and 'reject' in results[test]
        ]
        results['overall_reject'] = sum(rejects) > len(rejects) / 2

        return results

    def _estimate_power(self, test: str, n: int) -> float:
        """
        Estimate test power based on sample size.
        """
        # Rough power approximations
        if test == 'shapiro':
            if n < 20:
                return 0.3
            elif n < 50:
                return 0.5
            elif n < 100:
                return 0.7
            else:
                return 0.9
        return 0.5  # Default


class FastEppsPulley:
    """
    Optimized Epps-Pulley implementation with GPU support.
    """
    def __init__(
        self,
        n_points: int = 17,
        t_max: float = 1.4,
        slices: int = 1024
    ):
        self.n_points = n_points
        self.t_max = t_max
        self.slices = slices

        # Quadrature setup
        self._setup_quadrature()

    def _setup_quadrature(self):
        """Setup Gauss-Hermite quadrature."""
        # Use Gauss-Hermite for better integration
        points, weights = np.polynomial.hermite.hermgauss(self.n_points)

        # Transform to [0, t_max]
        self.t_points = torch.tensor(
            self.t_max * (points + 3) / 6,  # Map from [-3,3] to [0,t_max]
            dtype=torch.float32
        )
        self.weights = torch.tensor(
            weights * self.t_max / 6,
            dtype=torch.float32
        )

    @torch.jit.script
    def empirical_cf_vectorized(
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        JIT-compiled characteristic function computation.
        """
        # x: (n, d), t: (m,)
        n, d = x.shape
        m = t.shape[0]

        # Efficient batched computation
        x_flat = x.view(n, -1)  # (n, d)
        t_expanded = t.view(-1, 1, 1)  # (m, 1, 1)

        # Random projections for high dimensions
        if d > 10:
            # Use random projections to reduce dimension
            proj = torch.randn(d, 10, device=x.device) / np.sqrt(10)
            x_proj = x_flat @ proj  # (n, 10)
            phase = torch.einsum('ni,mij->mn', x_proj, t_expanded.expand(m, 10, 1))
        else:
            phase = torch.einsum('ni,m->mn', x_flat, t)

        # Stable CF computation
        cf_real = torch.cos(phase).mean(dim=0)
        cf_imag = torch.sin(phase).mean(dim=0)

        return cf_real, cf_imag

    def test_statistic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute test statistic with slicing for high dimensions.
        """
        n, d = x.shape

        if d == 1:
            return self._univariate_statistic(x.squeeze())

        # Slicing for multivariate
        statistics = []
        for _ in range(self.slices):
            # Random projection
            v = torch.randn(d, device=x.device)
            v = v / v.norm()

            # Project data
            x_proj = x @ v  # (n,)

            # Compute statistic
            stat = self._univariate_statistic(x_proj)
            statistics.append(stat)

        return torch.stack(statistics).mean()

    def _univariate_statistic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Univariate EP statistic.
        """
        n = x.shape[0]

        # Standardize
        x_std = (x - x.mean()) / (x.std() + 1e-8)

        # Empirical CF
        ecf_real = torch.cos(x_std.unsqueeze(1) * self.t_points.unsqueeze(0)).mean(0)
        ecf_imag = torch.sin(x_std.unsqueeze(1) * self.t_points.unsqueeze(0)).mean(0)

        # Standard normal CF
        ncf = torch.exp(-0.5 * self.t_points**2)

        # Weighted difference
        diff_sq = (ecf_real - ncf)**2 + ecf_imag**2
        weight = torch.exp(-0.5 * self.t_points**2)

        # Integrate
        statistic = n * torch.sum(diff_sq * weight * self.weights)

        return statistic
```

### 2. Advanced Divergence Measures

```python
class ComprehensiveDivergences(nn.Module):
    """
    Collection of divergence measures with gradient support.
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def kl_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        KL(P||Q) with numerical stability.
        """
        p = p + self.eps
        q = q + self.eps
        kl = p * (p.log() - q.log())

        if reduction == 'mean':
            return kl.mean()
        elif reduction == 'sum':
            return kl.sum()
        elif reduction == 'none':
            return kl
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def js_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        Jensen-Shannon divergence (symmetric).
        """
        m = (p + q) / 2
        return (self.kl_divergence(p, m) + self.kl_divergence(q, m)) / 2

    def renyi_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Rényi divergence of order α.
        """
        if alpha == 1:
            return self.kl_divergence(p, q)

        p = p + self.eps
        q = q + self.eps

        if alpha == 0:
            # D_0(P||Q) = -log P(Q>0)
            return -torch.log((q > 0).float().mean())
        elif alpha == float('inf'):
            # D_∞(P||Q) = log max(p/q)
            return torch.log(torch.max(p / q))
        else:
            # General case
            return torch.log((p**alpha * q**(1-alpha)).sum()) / (alpha - 1)

    def f_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        f: str = 'kl'
    ) -> torch.Tensor:
        """
        General f-divergence: D_f(P||Q) = ∫ q(x)f(p(x)/q(x))dx
        """
        ratio = (p + self.eps) / (q + self.eps)

        if f == 'kl':
            return (p * ratio.log()).mean()
        elif f == 'reverse_kl':
            return (-q * ratio.log()).mean()
        elif f == 'chi2':
            return ((ratio - 1)**2 * q).mean()
        elif f == 'hellinger':
            return ((ratio.sqrt() - 1)**2 * q).mean()
        elif f == 'tv':  # Total variation
            return 0.5 * (ratio - 1).abs().mean()
        else:
            raise ValueError(f"Unknown f-divergence: {f}")

    def mmd_with_gram(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel: str = 'gaussian',
        bandwidth: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MMD with Gram matrix (useful for visualization).
        """
        n, m = x.shape[0], y.shape[0]

        if bandwidth is None:
            # Median heuristic
            combined = torch.cat([x, y], dim=0)
            distances = torch.pdist(combined)
            bandwidth = distances.median() / np.sqrt(np.log(n + m))

        # Compute Gram matrices
        if kernel == 'gaussian':
            k_xx = torch.exp(-torch.cdist(x, x)**2 / (2 * bandwidth**2))
            k_yy = torch.exp(-torch.cdist(y, y)**2 / (2 * bandwidth**2))
            k_xy = torch.exp(-torch.cdist(x, y)**2 / (2 * bandwidth**2))
        elif kernel == 'laplacian':
            k_xx = torch.exp(-torch.cdist(x, x) / bandwidth)
            k_yy = torch.exp(-torch.cdist(y, y) / bandwidth)
            k_xy = torch.exp(-torch.cdist(x, y) / bandwidth)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # MMD estimate
        mmd = (k_xx.sum() - k_xx.trace()) / (n * (n - 1)) + \
              (k_yy.sum() - k_yy.trace()) / (m * (m - 1)) - \
              2 * k_xy.sum() / (n * m)

        # Combined Gram matrix for visualization
        gram = torch.zeros(n + m, n + m)
        gram[:n, :n] = k_xx
        gram[n:, n:] = k_yy
        gram[:n, n:] = k_xy
        gram[n:, :n] = k_xy.t()

        return torch.clamp(mmd, min=0), gram
```

### 3. Latent Space Regularization

```python
class LatentRegularizer(nn.Module):
    """
    Advanced regularization for latent variable models.
    """
    def __init__(
        self,
        reg_type: str = 'vae',
        beta: float = 1.0,
        capacity: Optional[float] = None,
        gamma: float = 1000.0
    ):
        super().__init__()
        self.reg_type = reg_type
        self.beta = beta
        self.capacity = capacity
        self.gamma = gamma
        self.step = 0

    def forward(
        self,
        z: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.
        """
        if training_step is not None:
            self.step = training_step

        losses = {}

        if self.reg_type == 'vae':
            losses['kl'] = self.vae_kl(mu, logvar)
        elif self.reg_type == 'beta_vae':
            losses['kl'] = self.beta * self.vae_kl(mu, logvar)
        elif self.reg_type == 'beta_vae_b':
            losses['kl'] = self.beta_vae_b(mu, logvar)
        elif self.reg_type == 'factor_vae':
            losses['kl'] = self.vae_kl(mu, logvar)
            losses['tc'] = self.total_correlation(z, mu, logvar)
        elif self.reg_type == 'dip_vae':
            losses['kl'] = self.vae_kl(mu, logvar)
            losses['dip'] = self.dip_regularizer(mu)
        elif self.reg_type == 'wae':
            losses['mmd'] = self.wae_mmd(z)
        elif self.reg_type == 'aae':
            losses['adversarial'] = 0  # Requires discriminator
        else:
            raise ValueError(f"Unknown regularizer: {self.reg_type}")

        # Normality enforcement
        if z is not None:
            losses['normality'] = self.normality_loss(z)

        return losses

    def vae_kl(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard VAE KL divergence.
        """
        return -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=-1).mean()

    def beta_vae_b(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        β-VAE with capacity increase.
        """
        kl = self.vae_kl(mu, logvar)

        if self.capacity is not None:
            # Linearly increase capacity
            c = min(self.capacity, self.capacity * self.step / 10000)
            return self.gamma * (kl - c).abs()
        else:
            return self.beta * kl

    def total_correlation(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Total Correlation for FactorVAE.
        """
        # log q(z)
        log_qz = self.log_density_gaussian(z, mu, logvar)

        # log q(z_i) summed over dimensions
        log_qzi = self.log_density_gaussian(
            z.unsqueeze(1),
            mu.unsqueeze(0),
            logvar.unsqueeze(0)
        ).sum(dim=-1)

        # log ∏q(z_i) = Σlog q(z_i)
        log_prod_qzi = log_qzi.logsumexp(dim=0) - np.log(z.shape[0])

        # TC = KL(q(z) || ∏q(z_i))
        return (log_qz - log_prod_qzi).mean()

    def log_density_gaussian(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Log density of Gaussian.
        """
        norm = -0.5 * (np.log(2 * np.pi) + logvar)
        log_density = norm - 0.5 * ((z - mu)**2 / logvar.exp())
        return log_density.sum(dim=-1)

    def dip_regularizer(
        self,
        mu: torch.Tensor,
        lambda_od: float = 10.0,
        lambda_d: float = 100.0
    ) -> torch.Tensor:
        """
        DIP-VAE regularizer for disentanglement.
        """
        # Covariance of mean
        cov_mu = torch.cov(mu.t())

        # Off-diagonal penalty
        cov_od = cov_mu**2
        mask = torch.eye(cov_mu.shape[0], device=mu.device).bool()
        cov_od.masked_fill_(mask, 0)
        loss_od = lambda_od * cov_od.sum() / 2

        # Diagonal penalty
        cov_d = torch.diagonal(cov_mu)
        loss_d = lambda_d * ((cov_d - 1)**2).sum() / 2

        return loss_od + loss_d

    def wae_mmd(
        self,
        z: torch.Tensor,
        n_projections: int = 50
    ) -> torch.Tensor:
        """
        Wasserstein Autoencoder MMD regularizer.
        """
        batch_size = z.shape[0]

        # Sample from prior
        z_prior = torch.randn_like(z)

        # Sliced Wasserstein distance
        theta = torch.randn(z.shape[1], n_projections, device=z.device)
        theta = theta / theta.norm(dim=0, keepdim=True)

        # Project
        z_proj = z @ theta
        z_prior_proj = z_prior @ theta

        # Sort projections
        z_proj_sorted = z_proj.sort(dim=0)[0]
        z_prior_proj_sorted = z_prior_proj.sort(dim=0)[0]

        # Sliced Wasserstein
        sw_dist = (z_proj_sorted - z_prior_proj_sorted).pow(2).mean()

        return sw_dist

    def normality_loss(
        self,
        z: torch.Tensor,
        method: str = 'moment'
    ) -> torch.Tensor:
        """
        Enforce normality through moment matching or tests.
        """
        if method == 'moment':
            # Match first 4 moments
            z_std = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-8)

            # Skewness (should be 0)
            skew = (z_std**3).mean()

            # Kurtosis (should be 3)
            kurt = (z_std**4).mean() - 3

            return skew.abs() + 0.1 * kurt.abs()

        elif method == 'ep_differentiable':
            # Differentiable version of EP test
            t = torch.linspace(0.1, 2.0, 20, device=z.device)
            ecf = torch.cos((z.unsqueeze(-1) * t).sum(dim=1)).mean(dim=0)
            ncf = torch.exp(-0.5 * t**2)
            return ((ecf - ncf)**2).mean()

        else:
            raise ValueError(f"Unknown method: {method}")
```

### 4. Distribution Drift Detection

```python
class DriftDetector:
    """
    Online distribution drift detection.
    """
    def __init__(
        self,
        window_size: int = 100,
        method: str = 'ks',
        threshold: float = 0.05
    ):
        self.window_size = window_size
        self.method = method
        self.threshold = threshold
        self.reference_window = []
        self.test_window = []
        self.drift_points = []

    def update(
        self,
        x: Union[float, np.ndarray],
        timestamp: Optional[int] = None
    ) -> bool:
        """
        Update with new sample and check for drift.
        """
        # Add to test window
        self.test_window.append(x)

        # Initialize reference window
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(x)
            return False

        # Maintain window size
        if len(self.test_window) > self.window_size:
            self.test_window.pop(0)

        # Test for drift
        drift_detected = self.test_drift()

        if drift_detected:
            # Reset reference window
            self.reference_window = self.test_window.copy()
            self.drift_points.append(timestamp or len(self.drift_points))

        return drift_detected

    def test_drift(self) -> bool:
        """
        Test for distribution drift.
        """
        ref_data = np.array(self.reference_window)
        test_data = np.array(self.test_window)

        if self.method == 'ks':
            # Kolmogorov-Smirnov test
            if ref_data.ndim == 1:
                _, p_value = stats.ks_2samp(ref_data, test_data)
            else:
                # Multivariate: test each dimension
                p_values = []
                for d in range(ref_data.shape[1]):
                    _, p = stats.ks_2samp(ref_data[:, d], test_data[:, d])
                    p_values.append(p)
                # Bonferroni correction
                p_value = min(p_values) * len(p_values)

        elif self.method == 'mmd':
            # Maximum Mean Discrepancy
            mmd = self.compute_mmd(ref_data, test_data)
            # Permutation test for p-value
            p_value = self.mmd_permutation_test(ref_data, test_data, mmd)

        elif self.method == 'energy':
            # Energy distance
            energy_dist = self.energy_distance(ref_data, test_data)
            # Bootstrap for p-value
            p_value = self.bootstrap_test(ref_data, test_data, energy_dist)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return p_value < self.threshold

    def compute_mmd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        gamma: float = 1.0
    ) -> float:
        """
        Compute MMD with RBF kernel.
        """
        xx = np.dot(x, x.T)
        yy = np.dot(y, y.T)
        xy = np.dot(x, y.T)

        rx = np.diag(xx)[:, np.newaxis]
        ry = np.diag(yy)[:, np.newaxis]

        k_xx = np.exp(-gamma * (rx + rx.T - 2 * xx))
        k_yy = np.exp(-gamma * (ry + ry.T - 2 * yy))
        k_xy = np.exp(-gamma * (rx + ry.T - 2 * xy))

        n, m = len(x), len(y)
        mmd = (k_xx.sum() - n) / (n * (n - 1)) + \
              (k_yy.sum() - m) / (m * (m - 1)) - \
              2 * k_xy.sum() / (n * m)

        return max(0, mmd)

    def mmd_permutation_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        observed_mmd: float,
        n_permutations: int = 100
    ) -> float:
        """
        Permutation test for MMD.
        """
        combined = np.vstack([x, y])
        n = len(x)

        count = 0
        for _ in range(n_permutations):
            perm = np.random.permutation(len(combined))
            x_perm = combined[perm[:n]]
            y_perm = combined[perm[n:]]
            mmd_perm = self.compute_mmd(x_perm, y_perm)

            if mmd_perm >= observed_mmd:
                count += 1

        return count / n_permutations
```

## Debugging Strategies

### Common Issues and Solutions

1. **NaN in Characteristic Function**
   ```python
   def debug_cf_computation(x):
       # Check for extreme values
       print(f"Data range: [{x.min():.2f}, {x.max():.2f}]")
       print(f"Data std: {x.std():.4f}")

       # Standardize if needed
       if x.std() > 10:
           x = (x - x.mean()) / x.std()

       # Use smaller t values
       t = torch.linspace(0.01, 0.5, 10)

       # Compute CF step by step
       for t_val in t:
           phase = x * t_val
           print(f"t={t_val:.2f}: max_phase={phase.abs().max():.2f}")
   ```

2. **Divergence Explosion in VAE**
   ```python
   def diagnose_vae_collapse(model, dataloader):
       kl_per_dim = []
       for batch in dataloader:
           mu, logvar = model.encode(batch)
           kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1)
           kl_per_dim.append(kl.mean(0).cpu())

       kl_per_dim = torch.stack(kl_per_dim).mean(0)
       print(f"KL per dimension: {kl_per_dim}")
       print(f"Collapsed dims: {(kl_per_dim < 0.01).sum()}")
   ```

3. **MMD Always Zero**
   ```python
   def fix_mmd_bandwidth(x, y):
       # Check if bandwidth is appropriate
       distances = torch.cdist(x, y).flatten()
       print(f"Distance percentiles: {np.percentile(distances.cpu(), [25, 50, 75])}")

       # Try multiple bandwidths
       for bw in [0.1, 1.0, 10.0]:
           mmd = compute_mmd(x, y, bandwidth=bw)
           print(f"Bandwidth {bw}: MMD = {mmd:.4f}")
   ```

## Performance Optimization

### Batch Processing
```python
def batch_statistical_tests(data_loader, test_fn, batch_size=1000):
    """
    Efficiently run tests on large datasets.
    """
    results = []
    batch_data = []

    for x, _ in data_loader:
        batch_data.append(x)

        if len(batch_data) >= batch_size:
            # Process batch
            combined = torch.cat(batch_data)
            result = test_fn(combined)
            results.append(result)
            batch_data = []

    return results
```

### GPU Acceleration
```python
@torch.cuda.amp.autocast()
def fast_mmd_gpu(x, y, num_features=100):
    """
    GPU-accelerated MMD with random features.
    """
    d = x.shape[1]

    # Random Fourier features
    w = torch.randn(d, num_features, device=x.device) / np.sqrt(d)
    b = torch.rand(num_features, device=x.device) * 2 * np.pi

    # Feature maps
    z_x = np.sqrt(2/num_features) * torch.cos(x @ w + b)
    z_y = np.sqrt(2/num_features) * torch.cos(y @ w + b)

    # MMD in feature space
    mmd = z_x.mean(0) - z_y.mean(0)
    return mmd.norm()**2
```

## Literature References

### Classical Statistical Testing
- **Epps & Pulley (1983)**: "A test for normality based on the empirical characteristic function"
- **Shapiro & Wilk (1965)**: "An analysis of variance test for normality"
- **Anderson & Darling (1952)**: "Asymptotic theory of certain goodness of fit criteria"

### Modern Machine Learning Tests
- **Gretton et al. (2012)**: "A Kernel Two-Sample Test" (MMD)
- **Székely & Rizzo (2013)**: "Energy statistics: A class of statistics based on distances"
- **Chwialkowski et al. (2016)**: "A Kernel Test of Goodness of Fit"

### VAE Regularization
- **Kingma & Welling (2014)**: "Auto-Encoding Variational Bayes"
- **Higgins et al. (2017)**: "β-VAE: Learning Basic Visual Concepts"
- **Kim & Mnih (2018)**: "Disentangling by Factorising" (FactorVAE)
- **Kumar et al. (2018)**: "Variational Inference of Disentangled Latent Concepts" (DIP-VAE)

## Quick Reference Tables

### Test Selection Guide

| Scenario | Recommended Test | Reason |
|----------|-----------------|--------|
| Small sample (n<50) | Shapiro-Wilk | Best power for small samples |
| Large sample (n>1000) | Anderson-Darling | Sensitive to tails |
| Multivariate | Epps-Pulley | Works in high dimensions |
| Time series | KPSS/ADF | Handles autocorrelation |
| Distribution drift | MMD | Non-parametric, works for any distribution |

### Divergence Properties

| Divergence | Symmetric | Metric | Differentiable | Computational |
|------------|----------|--------|----------------|---------------|
| KL | No | No | Yes | O(n) |
| JS | Yes | No | Yes | O(n) |
| Wasserstein | Yes | Yes | Yes* | O(n³) exact |
| MMD | Yes | Yes | Yes | O(n²) |
| Energy | Yes | Yes | Yes | O(n²) |

### Regularization Hyperparameters

| Method | Key Parameter | Typical Range | Effect |
|--------|--------------|---------------|--------|
| VAE | β | 1-10 | KL weight |
| β-VAE-B | C | 0-50 | Capacity |
| FactorVAE | γ | 1-100 | TC weight |
| DIP-VAE | λ_od, λ_d | 1-100 | Disentanglement |
| WAE | λ | 1-100 | MMD weight |

Remember: Statistical rigor is the foundation of reliable machine learning. Test assumptions, validate methods, and always consider the null hypothesis.