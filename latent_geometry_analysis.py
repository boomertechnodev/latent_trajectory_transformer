"""
Latent Space Geometry Analysis Tools for Latent Trajectory Transformer
========================================================================

This module provides comprehensive geometric analysis and visualization tools
for understanding and improving the latent space structure in both ODE and
Raccoon models.

Author: Latent Geometry Specialist Agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not installed. Install with: pip install umap-learn")


# ============================================================================
#  INTRINSIC DIMENSIONALITY ESTIMATION
# ============================================================================

class IntrinsicDimensionEstimator:
    """
    Estimates the true intrinsic dimensionality of data lying on a manifold.
    Multiple methods for robustness.
    """

    @staticmethod
    def mle_levina_bickel(data: Tensor, k: int = 10) -> float:
        """
        Maximum Likelihood Estimation (Levina-Bickel 2004).

        Args:
            data: (n_samples, n_features) data tensor
            k: Number of nearest neighbors

        Returns:
            Estimated intrinsic dimension
        """
        n, d = data.shape
        data_np = data.detach().cpu().numpy()

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(data_np, data_np)

        # For each point, get k nearest neighbors
        dim_estimates = []
        for i in range(n):
            # Sort distances and take k+1 nearest (including self)
            sorted_dists = np.sort(distances[i])[1:k+2]  # Exclude self (0 distance)

            # MLE formula
            if sorted_dists[k] > 0:
                log_ratios = np.log(sorted_dists[k] / sorted_dists[:-1])
                dim_est = 1.0 / np.mean(log_ratios)
                dim_estimates.append(dim_est)

        return np.median(dim_estimates)  # Use median for robustness

    @staticmethod
    def correlation_dimension(data: Tensor, r_min: float = 0.01,
                             r_max: float = 1.0, n_points: int = 20) -> float:
        """
        Correlation dimension using box-counting.

        Args:
            data: (n_samples, n_features) data tensor
            r_min, r_max: Range of radii to consider
            n_points: Number of radius values

        Returns:
            Correlation dimension estimate
        """
        n = data.shape[0]
        data_np = data.detach().cpu().numpy()

        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(data_np))

        # Count pairs within radius r
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        counts = []

        for r in radii:
            count = np.sum(distances < r) - n  # Exclude diagonal
            counts.append(count)

        # Fit log-log slope
        log_r = np.log(radii)
        log_counts = np.log(np.array(counts) + 1)  # Add 1 to avoid log(0)

        # Linear regression in log-log space
        slope, _ = np.polyfit(log_r, log_counts, 1)

        return slope

    @staticmethod
    def pca_explained_variance(data: Tensor, threshold: float = 0.95) -> int:
        """
        Estimate dimension by PCA explained variance ratio.

        Args:
            data: (n_samples, n_features) data tensor
            threshold: Cumulative variance threshold

        Returns:
            Number of components explaining threshold variance
        """
        data_np = data.detach().cpu().numpy()

        # Center data
        data_centered = data_np - data_np.mean(axis=0)

        # Compute PCA
        pca = PCA()
        pca.fit(data_centered)

        # Find number of components for threshold variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= threshold) + 1

        return n_components


# ============================================================================
#  DISENTANGLEMENT METRICS
# ============================================================================

class DisentanglementMetrics:
    """
    Measures how well latent dimensions capture independent factors.
    """

    @staticmethod
    def mutual_information_gap(latents: Tensor, factors: Tensor) -> float:
        """
        MIG (Mutual Information Gap) metric.
        Higher is better (more disentangled).

        Args:
            latents: (n_samples, n_latents)
            factors: (n_samples, n_factors) ground truth factors

        Returns:
            MIG score in [0, 1]
        """
        from sklearn.feature_selection import mutual_info_regression

        latents_np = latents.detach().cpu().numpy()
        factors_np = factors.detach().cpu().numpy()

        n_latents = latents_np.shape[1]
        n_factors = factors_np.shape[1]

        # Compute MI matrix
        mi_matrix = np.zeros((n_latents, n_factors))
        for i in range(n_latents):
            for j in range(n_factors):
                mi = mutual_info_regression(
                    latents_np[:, i:i+1],
                    factors_np[:, j],
                    random_state=42
                )[0]
                mi_matrix[i, j] = mi

        # Normalize by entropy of factors
        from scipy.stats import entropy
        for j in range(n_factors):
            factor_entropy = entropy(np.histogram(factors_np[:, j], bins=20)[0] + 1e-10)
            if factor_entropy > 0:
                mi_matrix[:, j] /= factor_entropy

        # Compute gap between top-2 latents for each factor
        gaps = []
        for j in range(n_factors):
            sorted_mi = np.sort(mi_matrix[:, j])[::-1]
            if len(sorted_mi) > 1:
                gap = sorted_mi[0] - sorted_mi[1]
            else:
                gap = sorted_mi[0]
            gaps.append(gap)

        return np.mean(gaps)

    @staticmethod
    def separated_attribute_predictability(latents: Tensor, factors: Tensor) -> float:
        """
        SAP metric - measures if each factor is predicted by one latent.

        Args:
            latents: (n_samples, n_latents)
            factors: (n_samples, n_factors)

        Returns:
            SAP score in [0, 1]
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        latents_np = latents.detach().cpu().numpy()
        factors_np = factors.detach().cpu().numpy()

        n_latents = latents_np.shape[1]
        n_factors = factors_np.shape[1]

        sap_scores = []
        for j in range(n_factors):
            # Train regressor for each latent dimension
            scores = []
            for i in range(n_latents):
                reg = LinearRegression()
                score = cross_val_score(
                    reg,
                    latents_np[:, i:i+1],
                    factors_np[:, j],
                    cv=5,
                    scoring='r2'
                ).mean()
                scores.append(max(0, score))  # Clip negative R²

            # SAP is difference between top-2 scores
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) > 1:
                sap = sorted_scores[0] - sorted_scores[1]
            else:
                sap = sorted_scores[0]
            sap_scores.append(sap)

        return np.mean(sap_scores)

    @staticmethod
    def total_correlation(latents: Tensor) -> float:
        """
        Measures statistical independence between latent dimensions.
        Lower is better (more independent).

        Args:
            latents: (n_samples, n_latents)

        Returns:
            Total correlation (KL divergence from independence)
        """
        # Estimate using sampling approach
        n, d = latents.shape

        # Log p(z) - joint distribution
        mean = latents.mean(0)
        cov = torch.cov(latents.T)

        # Multivariate normal log prob
        diff = latents - mean
        inv_cov = torch.linalg.pinv(cov + 1e-6 * torch.eye(d, device=latents.device))
        log_pz = -0.5 * (diff @ inv_cov * diff).sum(1)

        # Sum of log p(z_i) - marginal distributions
        log_pz_marginal = 0
        for i in range(d):
            z_i = latents[:, i]
            mean_i = z_i.mean()
            std_i = z_i.std() + 1e-6
            log_pz_i = -0.5 * ((z_i - mean_i) / std_i).pow(2) - torch.log(std_i)
            log_pz_marginal += log_pz_i

        # Total correlation = KL(p(z) || prod p(z_i))
        tc = (log_pz - log_pz_marginal).mean()

        return tc.item()


# ============================================================================
#  TRAJECTORY ANALYSIS
# ============================================================================

class TrajectoryGeometry:
    """
    Analyzes geometric properties of latent trajectories.
    """

    @staticmethod
    def trajectory_curvature(trajectory: Tensor) -> Tensor:
        """
        Compute curvature along a trajectory.

        Args:
            trajectory: (n_steps, latent_dim) or (batch, n_steps, latent_dim)

        Returns:
            curvatures: (n_steps-2,) or (batch, n_steps-2)
        """
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)

        # First derivative (velocity)
        v = torch.diff(trajectory, dim=1)

        # Second derivative (acceleration)
        a = torch.diff(v, dim=1)

        # Curvature = ||v × a|| / ||v||^3
        # For high-dimensional space, use: κ = ||a|| / ||v||^2
        v_norm = torch.norm(v[:, :-1, :], dim=-1) + 1e-8
        a_norm = torch.norm(a, dim=-1)

        curvature = a_norm / (v_norm ** 2)

        return curvature.squeeze(0) if trajectory.shape[0] == 1 else curvature

    @staticmethod
    def trajectory_length(trajectory: Tensor) -> float:
        """
        Compute arc length of trajectory.

        Args:
            trajectory: (n_steps, latent_dim)

        Returns:
            Total arc length
        """
        # Sum of distances between consecutive points
        diffs = torch.diff(trajectory, dim=0)
        distances = torch.norm(diffs, dim=-1)
        return distances.sum().item()

    @staticmethod
    def trajectory_smoothness(trajectory: Tensor) -> float:
        """
        Measure trajectory smoothness using total variation.
        Lower is smoother.

        Args:
            trajectory: (n_steps, latent_dim)

        Returns:
            Smoothness score
        """
        # Total variation of velocity
        v = torch.diff(trajectory, dim=0)
        tv = torch.diff(v, dim=0).abs().sum()

        # Normalize by trajectory length
        length = TrajectoryGeometry.trajectory_length(trajectory)

        return (tv / (length + 1e-8)).item()

    @staticmethod
    def geodesic_distance(z1: Tensor, z2: Tensor, flow_model: Optional[nn.Module] = None) -> float:
        """
        Compute geodesic distance between two points.

        Args:
            z1, z2: Points in latent space
            flow_model: Optional normalizing flow for manifold structure

        Returns:
            Geodesic distance
        """
        if flow_model is None:
            # Euclidean distance as approximation
            return torch.norm(z2 - z1).item()

        # Use flow to map to base space where geodesics are straight
        with torch.no_grad():
            # Assuming flow has inverse method
            w1 = flow_model(z1.unsqueeze(0), torch.zeros(1, 1), reverse=True)[0]
            w2 = flow_model(z2.unsqueeze(0), torch.zeros(1, 1), reverse=True)[0]

            # Distance in base space
            return torch.norm(w2 - w1).item()


# ============================================================================
#  LATENT SPACE VISUALIZATION
# ============================================================================

class LatentVisualizer:
    """
    Creates comprehensive visualizations of latent space structure.
    """

    @staticmethod
    def plot_2d_manifold(latents: Tensor, labels: Optional[Tensor] = None,
                         method: str = 'pca', title: str = "Latent Manifold") -> plt.Figure:
        """
        Project high-dimensional latents to 2D for visualization.

        Args:
            latents: (n_samples, latent_dim)
            labels: Optional (n_samples,) for coloring
            method: 'pca', 'tsne', or 'umap'
            title: Plot title

        Returns:
            matplotlib figure
        """
        latents_np = latents.detach().cpu().numpy()

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedded = reducer.fit_transform(latents_np)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = reducer.fit_transform(latents_np)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(latents_np)
        else:
            raise ValueError(f"Method {method} not available")

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                               c=labels_np, cmap='tab10', alpha=0.7, s=20)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.7, s=20)

        ax.set_title(f"{title} ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        return fig

    @staticmethod
    def plot_trajectory_3d(trajectory: Tensor, title: str = "Latent Trajectory") -> plt.Figure:
        """
        Visualize trajectory in 3D (using first 3 principal components).

        Args:
            trajectory: (n_steps, latent_dim)
            title: Plot title

        Returns:
            matplotlib figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        traj_np = trajectory.detach().cpu().numpy()

        # PCA to 3D
        if traj_np.shape[1] > 3:
            pca = PCA(n_components=3)
            traj_3d = pca.fit_transform(traj_np)
        else:
            traj_3d = traj_np

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
               'b-', alpha=0.7, linewidth=2)

        # Mark start and end
        ax.scatter(*traj_3d[0], color='green', s=100, marker='o', label='Start')
        ax.scatter(*traj_3d[-1], color='red', s=100, marker='s', label='End')

        # Add time coloring
        n_steps = len(traj_3d)
        colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
        ax.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                  c=colors, s=20, alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.legend()

        return fig

    @staticmethod
    def plot_latent_traversal(model: nn.Module, z: Tensor, dim_idx: int,
                             decoder_fn: callable, range_vals: Tuple[float, float] = (-3, 3),
                             n_steps: int = 10) -> Tensor:
        """
        Create traversal visualization by varying single latent dimension.

        Args:
            model: Model with decoder
            z: Base latent code (1, latent_dim)
            dim_idx: Dimension to traverse
            decoder_fn: Function to decode latent to data
            range_vals: Range of values to traverse
            n_steps: Number of steps

        Returns:
            Reconstructions tensor (n_steps, ...)
        """
        traversals = []
        z_copy = z.clone()

        for val in torch.linspace(range_vals[0], range_vals[1], n_steps):
            z_copy[:, dim_idx] = val
            with torch.no_grad():
                reconstruction = decoder_fn(z_copy)
            traversals.append(reconstruction)

        return torch.cat(traversals, dim=0)

    @staticmethod
    def plot_correlation_matrix(latents: Tensor) -> plt.Figure:
        """
        Plot correlation matrix of latent dimensions.

        Args:
            latents: (n_samples, latent_dim)

        Returns:
            matplotlib figure
        """
        # Compute correlation matrix
        latents_centered = latents - latents.mean(0)
        cov = torch.mm(latents_centered.T, latents_centered) / (latents.shape[0] - 1)
        std = torch.sqrt(torch.diag(cov))
        corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-8)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.detach().cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)

        ax.set_title("Latent Dimension Correlations")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Latent Dimension")

        return fig


# ============================================================================
#  ENCODER-DECODER ALIGNMENT
# ============================================================================

class EncoderDecoderAlignment:
    """
    Measures and improves encoder-decoder consistency.
    """

    @staticmethod
    def cycle_consistency_loss(encoder: nn.Module, decoder: nn.Module,
                              data: Tensor) -> Tuple[float, float]:
        """
        Compute cycle consistency: x → z → x' → z' consistency.

        Args:
            encoder: Encoder network
            decoder: Decoder network
            data: Input data

        Returns:
            (data_consistency, latent_consistency) losses
        """
        # Encode data
        z = encoder(data)

        # Decode and re-encode
        x_recon = decoder(z)
        z_cycle = encoder(x_recon)

        # Data consistency: x → z → x' ≈ x
        data_consistency = F.mse_loss(x_recon, data)

        # Latent consistency: z → x' → z' ≈ z
        latent_consistency = F.mse_loss(z_cycle, z)

        return data_consistency.item(), latent_consistency.item()

    @staticmethod
    def jacobian_analysis(decoder: nn.Module, z: Tensor) -> Dict[str, float]:
        """
        Analyze decoder Jacobian for local geometry understanding.

        Args:
            decoder: Decoder network
            z: Latent code (batch, latent_dim)

        Returns:
            Dictionary with geometric metrics
        """
        z.requires_grad_(True)
        x = decoder(z)

        # Flatten output
        x_flat = x.view(x.shape[0], -1)

        # Compute Jacobian for first output dimension (as proxy)
        jacobian = []
        for i in range(min(10, x_flat.shape[1])):  # Sample a few output dims
            grad = torch.autograd.grad(
                x_flat[:, i].sum(), z,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian.append(grad)

        jacobian = torch.stack(jacobian, dim=1)  # (batch, sampled_dims, latent_dim)

        # Compute singular values
        U, S, V = torch.svd(jacobian)

        # Metrics
        metrics = {
            'effective_rank': (S > 1e-5).sum(dim=1).float().mean().item(),
            'condition_number': (S.max(dim=1)[0] / (S.min(dim=1)[0] + 1e-8)).mean().item(),
            'max_singular_value': S.max().item(),
            'min_singular_value': S.min().item(),
        }

        return metrics


# ============================================================================
#  IMPROVED LATENT REGULARIZERS
# ============================================================================

class ImprovedLatentRegularizers(nn.Module):
    """
    Advanced regularization techniques for better latent geometry.
    """

    def __init__(self, beta: float = 4.0, gamma: float = 1.0):
        super().__init__()
        self.beta = beta  # Disentanglement weight
        self.gamma = gamma  # Independence weight

    def beta_vae_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        β-VAE loss for disentanglement.

        Args:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)

        Returns:
            Weighted KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.beta * kl_loss / mu.shape[0]

    def factor_vae_loss(self, z: Tensor, mu: Tensor, logvar: Tensor) -> Dict[str, Tensor]:
        """
        Factor-VAE decomposition of KL divergence.

        Args:
            z: Sampled latents (batch, latent_dim)
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)

        Returns:
            Dictionary with loss components
        """
        batch_size = z.shape[0]
        latent_dim = z.shape[1]

        # 1. Index-Code Mutual Information
        # I(z;n) where n is data index
        log_qz_x = self._gaussian_log_pdf(z, mu, logvar).sum(1)
        log_qz = self._gaussian_log_pdf(
            z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0)
        ).sum(2).logsumexp(1) - np.log(batch_size)
        index_code_mi = (log_qz_x - log_qz).mean()

        # 2. Total Correlation
        # KL(q(z) || ∏q(z_i))
        log_qz_marginal = []
        for i in range(latent_dim):
            log_qz_i = self._gaussian_log_pdf(
                z[:, i:i+1].unsqueeze(1),
                mu[:, i:i+1].unsqueeze(0),
                logvar[:, i:i+1].unsqueeze(0)
            ).squeeze(2).logsumexp(1) - np.log(batch_size)
            log_qz_marginal.append(log_qz_i)
        log_qz_marginal = torch.stack(log_qz_marginal, dim=1).sum(1)
        total_correlation = (log_qz - log_qz_marginal).mean()

        # 3. Dimension-wise KL
        dim_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

        return {
            'index_code_mi': self.beta * index_code_mi,
            'total_correlation': self.gamma * total_correlation,
            'dimension_kl': dim_kl,
            'total': self.beta * index_code_mi + self.gamma * total_correlation + dim_kl
        }

    def _gaussian_log_pdf(self, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute log probability under Gaussian."""
        return -0.5 * (
            logvar +
            (x - mu).pow(2) / (logvar.exp() + 1e-8) +
            np.log(2 * np.pi)
        )

    def contrastive_latent_loss(self, z: Tensor, temperature: float = 0.1) -> Tensor:
        """
        Contrastive loss to spread latents uniformly on hypersphere.

        Args:
            z: Latents (batch, latent_dim)
            temperature: Temperature parameter

        Returns:
            Contrastive loss
        """
        # Normalize to unit sphere
        z_norm = F.normalize(z, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.T) / temperature

        # Mask diagonal
        mask = torch.eye(z.shape[0], device=z.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))

        # Softmax over similarities
        probs = F.softmax(sim_matrix, dim=1)

        # Encourage uniform distribution
        uniform_target = torch.ones_like(probs) / (z.shape[0] - 1)
        uniform_target.masked_fill_(mask, 0)

        loss = F.kl_div(probs.log(), uniform_target, reduction='batchmean')

        return loss


# ============================================================================
#  MANIFOLD-AWARE INTERPOLATION
# ============================================================================

class ManifoldInterpolation:
    """
    Smooth interpolation methods respecting manifold structure.
    """

    @staticmethod
    def spherical_interpolation(z1: Tensor, z2: Tensor, t: float) -> Tensor:
        """
        SLERP (Spherical Linear Interpolation) for normalized codes.

        Args:
            z1, z2: Latent codes
            t: Interpolation parameter in [0, 1]

        Returns:
            Interpolated code
        """
        # Normalize
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        # Compute angle
        cos_omega = (z1_norm * z2_norm).sum(-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(cos_omega)

        # Interpolate
        sin_omega = torch.sin(omega) + 1e-8
        w1 = torch.sin((1 - t) * omega) / sin_omega
        w2 = torch.sin(t * omega) / sin_omega

        # Handle edge case where z1 ≈ z2
        nearly_parallel = (omega.abs() < 1e-3).float()
        w1 = nearly_parallel * (1 - t) + (1 - nearly_parallel) * w1
        w2 = nearly_parallel * t + (1 - nearly_parallel) * w2

        return w1 * z1 + w2 * z2

    @staticmethod
    def geodesic_interpolation(z1: Tensor, z2: Tensor, t: float,
                               metric_tensor_fn: Optional[callable] = None) -> Tensor:
        """
        Interpolate along geodesic using learned metric.

        Args:
            z1, z2: Start and end points
            t: Interpolation parameter
            metric_tensor_fn: Function computing metric tensor at point

        Returns:
            Point on geodesic
        """
        if metric_tensor_fn is None:
            # Default to Euclidean
            return (1 - t) * z1 + t * z2

        # Use ODE to compute geodesic (simplified)
        # This would require solving geodesic equation
        # For now, use linear approximation
        z_t = (1 - t) * z1 + t * z2

        # Apply metric correction
        G = metric_tensor_fn(z_t)
        if G is not None:
            # Project onto metric-adjusted direction
            direction = z2 - z1
            adjusted = torch.matmul(G, direction.unsqueeze(-1)).squeeze(-1)
            z_t = z1 + t * adjusted

        return z_t


# ============================================================================
#  COMPREHENSIVE LATENT ANALYSIS PIPELINE
# ============================================================================

def analyze_latent_space(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_plots: bool = True,
    plot_dir: str = "latent_analysis"
) -> Dict:
    """
    Comprehensive analysis of model's latent space geometry.

    Args:
        model: Model with encoder
        dataloader: Data loader
        device: Torch device
        save_plots: Whether to save visualization plots
        plot_dir: Directory to save plots

    Returns:
        Dictionary with all analysis metrics
    """
    import os
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print("\n" + "="*70)
    print("LATENT SPACE GEOMETRY ANALYSIS")
    print("="*70)

    # Collect latent codes
    print("\n📊 Collecting latent representations...")
    latents_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # Limit samples for analysis
                break

            if isinstance(batch, tuple):
                data, labels = batch
            else:
                data = batch
                labels = None

            data = data.to(device)

            # Get latent codes (handle different model types)
            if hasattr(model, 'encode'):
                z = model.encode(data)
                if isinstance(z, tuple):  # VAE-style (mean, logvar)
                    z = z[0]  # Use mean
            elif hasattr(model, 'encoder'):
                z = model.encoder(data)
            else:
                continue

            # Flatten if needed
            if z.dim() > 2:
                z = z.reshape(z.shape[0], -1)

            latents_list.append(z)
            if labels is not None:
                labels_list.append(labels)

    if not latents_list:
        print("⚠️  No latents collected!")
        return {}

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0) if labels_list else None

    print(f"✅ Collected {latents.shape[0]} latent codes of dimension {latents.shape[1]}")

    results = {}

    # 1. Intrinsic Dimensionality
    print("\n🔍 Estimating intrinsic dimensionality...")
    dim_estimator = IntrinsicDimensionEstimator()

    dim_mle = dim_estimator.mle_levina_bickel(latents, k=10)
    dim_corr = dim_estimator.correlation_dimension(latents)
    dim_pca = dim_estimator.pca_explained_variance(latents, threshold=0.95)

    results['intrinsic_dimension'] = {
        'mle': dim_mle,
        'correlation': dim_corr,
        'pca_95': dim_pca
    }

    print(f"  MLE estimate: {dim_mle:.2f}")
    print(f"  Correlation dimension: {dim_corr:.2f}")
    print(f"  PCA 95% variance: {dim_pca} components")

    # 2. Disentanglement (if we have factors)
    if labels is not None and labels.dim() == 1:
        print("\n🎯 Computing disentanglement metrics...")
        metrics = DisentanglementMetrics()

        # Convert labels to one-hot for factor analysis
        n_classes = labels.max().item() + 1
        factors = F.one_hot(labels, n_classes).float()

        mig = metrics.mutual_information_gap(latents, factors)
        sap = metrics.separated_attribute_predictability(latents, factors)
        tc = metrics.total_correlation(latents)

        results['disentanglement'] = {
            'mig': mig,
            'sap': sap,
            'total_correlation': tc
        }

        print(f"  MIG score: {mig:.3f}")
        print(f"  SAP score: {sap:.3f}")
        print(f"  Total correlation: {tc:.3f}")

    # 3. Latent Statistics
    print("\n📈 Computing latent statistics...")

    results['statistics'] = {
        'mean_norm': latents.norm(dim=1).mean().item(),
        'std_norm': latents.norm(dim=1).std().item(),
        'mean_activation': latents.mean().item(),
        'std_activation': latents.std().item(),
        'sparsity': (latents.abs() < 0.1).float().mean().item(),
    }

    print(f"  Mean L2 norm: {results['statistics']['mean_norm']:.3f}")
    print(f"  Activation sparsity: {results['statistics']['sparsity']:.3f}")

    # 4. Visualization
    if save_plots:
        print("\n🎨 Creating visualizations...")
        viz = LatentVisualizer()

        # 2D manifold plots
        for method in ['pca', 'tsne']:
            fig = viz.plot_2d_manifold(latents, labels, method=method)
            fig.savefig(f"{plot_dir}/manifold_{method}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Correlation matrix
        fig = viz.plot_correlation_matrix(latents)
        fig.savefig(f"{plot_dir}/correlation_matrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved visualizations to {plot_dir}/")

    # 5. Trajectory Analysis (if applicable)
    if hasattr(model, 'p_ode') or hasattr(model, 'dynamics'):
        print("\n🌀 Analyzing trajectory geometry...")

        # Generate sample trajectory
        z0 = torch.randn(1, latents.shape[1], device=device)

        if hasattr(model, 'p_ode'):
            # ODE model
            from latent_drift_trajectory import solve_ode
            trajectory = solve_ode(model.p_ode, z0, 0.0, 1.0, 50)
            trajectory = trajectory.squeeze(1)  # Remove batch dim
        elif hasattr(model, 'dynamics'):
            # SDE model
            from latent_drift_trajectory import solve_sde
            t_span = torch.linspace(0, 1, 50, device=device)
            trajectory = solve_sde(model.dynamics, z0, t_span)
            trajectory = trajectory.squeeze(0)  # Remove batch dim

        traj_geom = TrajectoryGeometry()

        curvature = traj_geom.trajectory_curvature(trajectory)
        length = traj_geom.trajectory_length(trajectory)
        smoothness = traj_geom.trajectory_smoothness(trajectory)

        results['trajectory'] = {
            'mean_curvature': curvature.mean().item(),
            'max_curvature': curvature.max().item(),
            'arc_length': length,
            'smoothness': smoothness
        }

        print(f"  Mean curvature: {results['trajectory']['mean_curvature']:.3f}")
        print(f"  Arc length: {results['trajectory']['arc_length']:.3f}")
        print(f"  Smoothness: {results['trajectory']['smoothness']:.3f}")

        if save_plots:
            # 3D trajectory plot
            fig = viz.plot_trajectory_3d(trajectory)
            fig.savefig(f"{plot_dir}/trajectory_3d.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)

    return results


# ============================================================================
#  USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Example usage
    print("\nExample Analysis Pipeline:")
    print("-" * 40)

    # Create dummy data for demonstration
    n_samples = 1000
    latent_dim = 32

    # Simulate latent codes with structure
    latents = torch.randn(n_samples, latent_dim)

    # Add some structure (lower intrinsic dimension)
    basis = torch.randn(latent_dim, 5)  # 5D subspace
    coeffs = torch.randn(n_samples, 5)
    latents = latents * 0.1 + torch.mm(coeffs, basis.T)

    # Simulate labels
    labels = torch.randint(0, 4, (n_samples,))

    # Test intrinsic dimension estimation
    print("\n1. Intrinsic Dimension:")
    estimator = IntrinsicDimensionEstimator()
    dim = estimator.mle_levina_bickel(latents)
    print(f"   Estimated dimension: {dim:.2f} (true: ~5)")

    # Test disentanglement metrics
    print("\n2. Disentanglement Metrics:")
    metrics = DisentanglementMetrics()
    tc = metrics.total_correlation(latents)
    print(f"   Total correlation: {tc:.3f}")

    # Test trajectory analysis
    print("\n3. Trajectory Geometry:")
    trajectory = torch.cumsum(torch.randn(100, latent_dim) * 0.1, dim=0)
    geom = TrajectoryGeometry()
    length = geom.trajectory_length(trajectory)
    smoothness = geom.trajectory_smoothness(trajectory)
    print(f"   Arc length: {length:.3f}")
    print(f"   Smoothness: {smoothness:.3f}")

    print("\n✅ All geometry analysis tools ready!")
    print("Import this module and use analyze_latent_space() for full analysis.")