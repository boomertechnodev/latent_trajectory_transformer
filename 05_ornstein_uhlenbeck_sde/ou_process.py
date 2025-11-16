"""
Ornstein-Uhlenbeck Stochastic Differential Equation Process

This module implements the Ornstein-Uhlenbeck (OU) process, a continuous-time
mean-reverting stochastic process with beautiful analytical properties. The OU
process is the continuous analog of the discrete AR(1) process.

SDE Formulation:
    dX = -θ(X - μ)dt + σdW

Where:
    θ > 0: Mean reversion speed (spring stiffness pulling toward equilibrium)
    μ: Long-term mean (equilibrium point the process reverts to)
    σ > 0: Volatility (noise strength/diffusion coefficient)
    dW: Brownian motion increment

Key Properties:
    1. Mean-reverting: Process is pulled toward μ with strength θ
    2. Stationary distribution: X ~ N(μ, σ²/(2θ)) as t → ∞
    3. Exact transition: P(X(t+Δt) | X(t)) is Gaussian with closed-form parameters
    4. Markov property: Future depends only on present, not past

Exact Transition Distribution:
    Given X(t), the distribution of X(t+Δt) is:

    X(t+Δt) ~ N(mean, variance)

    mean = μ + (X(t) - μ) * exp(-θ * Δt)
    variance = (σ² / 2θ) * (1 - exp(-2θ * Δt))

This allows:
    - Exact sampling without discretization error
    - Exact likelihood computation without numerical integration
    - Efficient gradient-based learning of θ, μ, σ

Use Cases:
    - Prior dynamics in Variational Path Flows
    - Latent trajectory evolution in continuous-time models
    - Baseline SDE for comparison with learned dynamics
    - Financial modeling (interest rates, mean-reverting assets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class OUProcess(nn.Module):
    """
    Learnable Ornstein-Uhlenbeck process with exact sampling and likelihood.

    This module learns the parameters θ (mean reversion), μ (equilibrium), and
    σ (volatility) of the OU SDE. Parameters are stored in log-space to ensure
    positivity, then transformed via softplus during computation.

    Supports two modes:
    - Diagonal: Each dimension has independent parameters (D separate OU processes)
    - Grouped: Dimensions are grouped, each group shares parameters

    Args:
        dim: Dimensionality of the process
        num_groups: Number of parameter groups (1 = shared, D = diagonal)
        init_theta: Initial mean reversion speed (before log transform)
        init_mu: Initial equilibrium mean
        init_sigma: Initial volatility (before log transform)
        min_theta: Minimum θ after softplus (for numerical stability)
        min_sigma: Minimum σ after softplus (for numerical stability)

    Shape:
        Sample: x_t (batch_size, dim) → x_{t+dt} (batch_size, dim)
        Likelihood: x_path (batch_size, seq_len, dim) → log_prob (batch_size,)

    Example:
        >>> ou = OUProcess(dim=64, num_groups=1)  # Shared parameters
        >>> x0 = torch.randn(16, 64)
        >>> x1 = ou.sample(x0, dt=0.1)
        >>> path = torch.randn(16, 100, 64)
        >>> log_prob = ou.log_prob(path, dt=0.01)
    """

    def __init__(
        self,
        dim: int,
        num_groups: int = 1,
        init_theta: float = 1.0,
        init_mu: float = 0.0,
        init_sigma: float = 1.0,
        min_theta: float = 0.01,
        min_sigma: float = 0.01
    ):
        super().__init__()

        if num_groups < 1 or num_groups > dim:
            raise ValueError(f"num_groups must be in [1, {dim}], got {num_groups}")

        if dim % num_groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_groups ({num_groups})")

        self.dim = dim
        self.num_groups = num_groups
        self.group_size = dim // num_groups
        self.min_theta = min_theta
        self.min_sigma = min_sigma

        # Learnable parameters in log-space (for θ and σ) or direct (for μ)
        # Shape: (num_groups,) - one parameter per group

        # θ: mean reversion speed (stored as log(θ - min_theta))
        self.log_theta = nn.Parameter(
            torch.full((num_groups,), math.log(init_theta - min_theta + 1e-6))
        )

        # μ: equilibrium mean (stored directly, can be any real number)
        self.mu = nn.Parameter(torch.full((num_groups,), init_mu))

        # σ: volatility (stored as log(σ - min_sigma))
        self.log_sigma = nn.Parameter(
            torch.full((num_groups,), math.log(init_sigma - min_sigma + 1e-6))
        )

    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actual OU parameters (θ, μ, σ) after transformations.

        Returns:
            theta: Mean reversion speed (num_groups,)
            mu: Equilibrium mean (num_groups,)
            sigma: Volatility (num_groups,)
        """
        # Transform log-space to ensure positivity
        theta = F.softplus(self.log_theta) + self.min_theta
        mu = self.mu
        sigma = F.softplus(self.log_sigma) + self.min_sigma

        return theta, mu, sigma

    def expand_to_dim(self, param: torch.Tensor) -> torch.Tensor:
        """
        Expand group parameter to full dimension.

        Args:
            param: Parameter tensor (num_groups,) or (batch, num_groups)

        Returns:
            Expanded tensor (dim,) or (batch, dim)
        """
        if param.dim() == 1:
            # (num_groups,) → (num_groups, group_size) → (dim,)
            return param.unsqueeze(-1).expand(-1, self.group_size).reshape(-1)
        else:
            # (batch, num_groups) → (batch, num_groups, group_size) → (batch, dim)
            return param.unsqueeze(-1).expand(-1, -1, self.group_size).reshape(param.shape[0], -1)

    def sample(
        self,
        x_t: torch.Tensor,
        dt: float,
        return_mean_std: bool = False
    ) -> torch.Tensor:
        """
        Sample x_{t+dt} given x_t using exact OU transition distribution.

        Uses closed-form Gaussian transition:
            X(t+dt) ~ N(μ + (X(t) - μ)e^{-θΔt}, σ²/(2θ) * (1 - e^{-2θΔt}))

        Args:
            x_t: Current state (batch_size, dim)
            dt: Time step size (must be > 0)
            return_mean_std: If True, also return (mean, std) of transition

        Returns:
            x_{t+dt}: Next state (batch_size, dim)
            If return_mean_std: (x_{t+dt}, mean, std)
        """
        batch_size, dim = x_t.shape
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"

        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Get parameters: (num_groups,)
        theta, mu, sigma = self.get_parameters()

        # Expand to full dimension: (dim,)
        theta_full = self.expand_to_dim(theta)
        mu_full = self.expand_to_dim(mu)
        sigma_full = self.expand_to_dim(sigma)

        # Compute exact transition mean and variance
        # mean = μ + (x_t - μ) * exp(-θ * dt)
        exp_neg_theta_dt = torch.exp(-theta_full * dt)
        mean = mu_full + (x_t - mu_full) * exp_neg_theta_dt

        # variance = (σ² / 2θ) * (1 - exp(-2θ * dt))
        exp_neg_2theta_dt = torch.exp(-2 * theta_full * dt)
        variance = (sigma_full ** 2) / (2 * theta_full) * (1 - exp_neg_2theta_dt)
        std = torch.sqrt(variance)

        # Sample: x_{t+dt} = mean + std * randn()
        noise = torch.randn_like(x_t)
        x_next = mean + std * noise

        if return_mean_std:
            return x_next, mean, std
        else:
            return x_next

    def log_prob(
        self,
        x_path: torch.Tensor,
        dt: float,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute exact log-likelihood of a trajectory under the OU process.

        The path likelihood is the product of transition likelihoods:
            log p(x_{0:T}) = Σ_t log p(x_{t+1} | x_t)

        Each transition is Gaussian with exact mean and variance.

        Args:
            x_path: Trajectory (batch_size, seq_len, dim)
            dt: Time step between consecutive points
            mask: Optional mask (batch_size, seq_len) where True = valid position

        Returns:
            log_prob: Log-likelihood (batch_size,)
        """
        batch_size, seq_len, dim = x_path.shape
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"

        if seq_len < 2:
            raise ValueError(f"Path must have at least 2 time points, got {seq_len}")

        # Get parameters
        theta, mu, sigma = self.get_parameters()
        theta_full = self.expand_to_dim(theta)
        mu_full = self.expand_to_dim(mu)
        sigma_full = self.expand_to_dim(sigma)

        # Compute transition statistics for all consecutive pairs
        x_t = x_path[:, :-1]  # (batch, seq_len-1, dim)
        x_next = x_path[:, 1:]  # (batch, seq_len-1, dim)

        # Exact transition mean: μ + (x_t - μ) * exp(-θ * dt)
        exp_neg_theta_dt = torch.exp(-theta_full * dt)
        mean = mu_full + (x_t - mu_full) * exp_neg_theta_dt

        # Exact transition variance: (σ² / 2θ) * (1 - exp(-2θ * dt))
        exp_neg_2theta_dt = torch.exp(-2 * theta_full * dt)
        variance = (sigma_full ** 2) / (2 * theta_full) * (1 - exp_neg_2theta_dt)
        std = torch.sqrt(variance)

        # Compute Gaussian log-likelihood for each transition
        # log N(x_next; mean, std²) = -0.5 * log(2π) - log(std) - 0.5 * ((x_next - mean) / std)²
        log_2pi = math.log(2 * math.pi)
        log_std = torch.log(std)

        # Normalized residuals: (x_next - mean) / std
        normalized_residuals = (x_next - mean) / std

        # Log-likelihood per dimension per transition: (batch, seq_len-1, dim)
        log_prob_per_dim = -0.5 * log_2pi - log_std - 0.5 * (normalized_residuals ** 2)

        # Apply mask if provided
        if mask is not None:
            # mask shape: (batch, seq_len)
            # We need mask for transitions: (batch, seq_len-1)
            transition_mask = mask[:, 1:]  # Mask for x_{t+1}

            # Expand to dimension: (batch, seq_len-1, 1)
            transition_mask = transition_mask.unsqueeze(-1)

            # Zero out masked transitions
            log_prob_per_dim = log_prob_per_dim * transition_mask.float()

        # Sum over dimensions and time: (batch, seq_len-1, dim) → (batch,)
        log_prob = log_prob_per_dim.sum(dim=-1).sum(dim=-1)

        return log_prob

    def stationary_sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the stationary distribution of the OU process.

        The stationary distribution (as t → ∞) is:
            X ~ N(μ, σ² / 2θ)

        Args:
            batch_size: Number of samples
            device: Device to create tensor on

        Returns:
            x: Samples from stationary distribution (batch_size, dim)
        """
        # Get parameters
        theta, mu, sigma = self.get_parameters()

        # Expand to full dimension
        mu_full = self.expand_to_dim(mu)
        sigma_full = self.expand_to_dim(sigma)
        theta_full = self.expand_to_dim(theta)

        # Stationary variance: σ² / 2θ
        stationary_std = sigma_full / torch.sqrt(2 * theta_full)

        # Sample: x ~ N(μ, stationary_std²)
        noise = torch.randn(batch_size, self.dim, device=device)
        x = mu_full.to(device) + stationary_std.to(device) * noise

        return x

    def get_stationary_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and standard deviation of the stationary distribution.

        Returns:
            mean: Stationary mean (dim,)
            std: Stationary standard deviation (dim,)
        """
        theta, mu, sigma = self.get_parameters()

        mu_full = self.expand_to_dim(mu)
        sigma_full = self.expand_to_dim(sigma)
        theta_full = self.expand_to_dim(theta)

        stationary_mean = mu_full
        stationary_std = sigma_full / torch.sqrt(2 * theta_full)

        return stationary_mean, stationary_std

    def forward(self, x_t: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Forward pass: sample next state.

        Args:
            x_t: Current state (batch_size, dim)
            dt: Time step

        Returns:
            x_{t+dt}: Next state (batch_size, dim)
        """
        return self.sample(x_t, dt)
