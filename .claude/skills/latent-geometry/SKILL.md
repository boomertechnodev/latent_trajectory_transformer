# Latent Space Geometry Expert Skill

Comprehensive implementation of latent space analysis, manifold learning, VAE geometry, and disentanglement metrics. This skill provides advanced tools for understanding and optimizing the geometric properties of learned representations.

## Core Implementations

### 1. Disentanglement Metrics Suite

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from typing import Optional, Tuple, Dict, List, Union
import matplotlib.pyplot as plt
import seaborn as sns


class DisentanglementMetrics:
    """Comprehensive suite of disentanglement metrics for latent representations."""

    @staticmethod
    def compute_mig(latents: np.ndarray, factors: np.ndarray, discrete: bool = True) -> float:
        """
        Mutual Information Gap (MIG) metric.

        Args:
            latents: Latent representations (n_samples, n_latents)
            factors: Ground truth factors (n_samples, n_factors)
            discrete: Whether factors are discrete

        Returns:
            MIG score in [0, 1]
        """
        n_factors = factors.shape[1]
        n_latents = latents.shape[1]

        # Discretize continuous latents
        if not discrete:
            latents = DisentanglementMetrics._discretize(latents)
            factors = DisentanglementMetrics._discretize(factors)

        # Compute mutual information matrix
        mi_matrix = np.zeros((n_factors, n_latents))
        factor_entropies = np.zeros(n_factors)

        for j in range(n_factors):
            factor_entropies[j] = entropy(np.unique(factors[:, j], return_counts=True)[1])
            for i in range(n_latents):
                mi_matrix[j, i] = mutual_info_score(factors[:, j], latents[:, i])

        # Normalize by entropy
        mi_matrix = mi_matrix / (factor_entropies[:, None] + 1e-8)

        # Compute gap for each factor
        gaps = []
        for j in range(n_factors):
            mi_sorted = np.sort(mi_matrix[j, :])[::-1]
            gap = mi_sorted[0] - mi_sorted[1] if len(mi_sorted) > 1 else mi_sorted[0]
            gaps.append(gap)

        return np.mean(gaps)

    @staticmethod
    def compute_sap(latents: np.ndarray, factors: np.ndarray, continuous: bool = False,
                   regression_model: str = 'linear') -> float:
        """
        Separated Attribute Predictability (SAP) score.

        Args:
            latents: Latent representations (n_samples, n_latents)
            factors: Ground truth factors (n_samples, n_factors)
            continuous: Whether factors are continuous
            regression_model: Type of regression model ('linear' or 'forest')

        Returns:
            SAP score
        """
        n_factors = factors.shape[1]
        n_latents = latents.shape[1]

        sap_scores = []

        for j in range(n_factors):
            # Train predictor for each latent dimension
            scores = []
            for i in range(n_latents):
                X = latents[:, i:i+1]
                y = factors[:, j]

                # Choose appropriate model
                if continuous:
                    if regression_model == 'forest':
                        model = RandomForestRegressor(n_estimators=10, max_depth=5)
                    else:
                        model = LinearRegression()
                else:
                    model = LogisticRegression(max_iter=1000)

                # Train and evaluate
                split = int(0.8 * len(X))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)

            # SAP is difference between top two scores
            scores_sorted = sorted(scores, reverse=True)
            sap = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else scores_sorted[0]
            sap_scores.append(sap)

        return np.mean(sap_scores)

    @staticmethod
    def compute_dci(latents: np.ndarray, factors: np.ndarray) -> Dict[str, float]:
        """
        Disentanglement, Completeness, and Informativeness (DCI) metrics.

        Returns:
            Dictionary with 'disentanglement', 'completeness', 'informativeness'
        """
        # Train importance matrix using Gradient Boosting
        importance_matrix = DisentanglementMetrics._train_importance_matrix(latents, factors)

        # Normalize importance matrix
        importance_matrix = importance_matrix / (importance_matrix.sum(axis=0, keepdims=True) + 1e-8)

        # Disentanglement: Each latent should encode at most one factor
        disentanglement = DisentanglementMetrics._compute_disentanglement(importance_matrix)

        # Completeness: Each factor should be encoded by at most one latent
        completeness = DisentanglementMetrics._compute_completeness(importance_matrix)

        # Informativeness: Overall prediction accuracy
        informativeness = DisentanglementMetrics._compute_informativeness(latents, factors)

        return {
            'disentanglement': disentanglement,
            'completeness': completeness,
            'informativeness': informativeness,
            'overall_dci': (disentanglement + completeness + informativeness) / 3
        }

    @staticmethod
    def compute_beta_vae_metric(latents: np.ndarray, factors: np.ndarray,
                               n_train: int = 10000, n_eval: int = 5000) -> float:
        """
        Beta-VAE disentanglement metric based on factor classification accuracy.
        """
        n_factors = factors.shape[1]
        accuracies = []

        for _ in range(10):  # Multiple random evaluations
            # Sample training pairs with single factor difference
            idx1 = np.random.choice(n_train, n_train)
            idx2 = np.random.choice(n_train, n_train)

            # Find pairs differing in exactly one factor
            factor_diffs = (factors[idx1] != factors[idx2]).sum(axis=1)
            valid_pairs = np.where(factor_diffs == 1)[0]

            if len(valid_pairs) < 100:
                continue

            # Subsample valid pairs
            valid_pairs = valid_pairs[:min(1000, len(valid_pairs))]
            idx1 = idx1[valid_pairs]
            idx2 = idx2[valid_pairs]

            # Get latent differences
            latent_diffs = np.abs(latents[idx1] - latents[idx2])

            # Find which factor changed
            factor_labels = np.argmax(np.abs(factors[idx1] - factors[idx2]), axis=1)

            # Train classifier
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(latent_diffs, factor_labels)

            # Evaluate
            eval_idx1 = np.random.choice(n_eval, 100)
            eval_idx2 = np.random.choice(n_eval, 100)
            eval_diffs = np.abs(latents[eval_idx1] - latents[eval_idx2])
            eval_labels = np.argmax(np.abs(factors[eval_idx1] - factors[eval_idx2]), axis=1)

            accuracy = classifier.score(eval_diffs, eval_labels)
            accuracies.append(accuracy)

        return np.mean(accuracies) if accuracies else 0.0

    @staticmethod
    def _discretize(data: np.ndarray, n_bins: int = 20) -> np.ndarray:
        """Discretize continuous data into bins."""
        discretized = np.zeros_like(data, dtype=int)
        for i in range(data.shape[1]):
            discretized[:, i] = np.digitize(data[:, i],
                                           bins=np.histogram(data[:, i], bins=n_bins)[1][:-1])
        return discretized

    @staticmethod
    def _train_importance_matrix(latents: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Train importance matrix for DCI metric."""
        n_factors = factors.shape[1]
        n_latents = latents.shape[1]
        importance_matrix = np.zeros((n_factors, n_latents))

        for j in range(n_factors):
            # Use Random Forest to get feature importances
            rf = RandomForestRegressor(n_estimators=100, max_depth=5)
            rf.fit(latents, factors[:, j])
            importance_matrix[j, :] = rf.feature_importances_

        return importance_matrix

    @staticmethod
    def _compute_disentanglement(importance_matrix: np.ndarray) -> float:
        """Compute disentanglement from importance matrix."""
        # For each latent, compute entropy over factors
        per_latent = []
        for i in range(importance_matrix.shape[1]):
            if importance_matrix[:, i].sum() > 0:
                probs = importance_matrix[:, i] / importance_matrix[:, i].sum()
                H = -np.sum(probs * np.log(probs + 1e-8))
                per_latent.append(1 - H / np.log(len(probs)))
            else:
                per_latent.append(0)
        return np.mean(per_latent)

    @staticmethod
    def _compute_completeness(importance_matrix: np.ndarray) -> float:
        """Compute completeness from importance matrix."""
        # For each factor, compute entropy over latents
        per_factor = []
        for j in range(importance_matrix.shape[0]):
            if importance_matrix[j, :].sum() > 0:
                probs = importance_matrix[j, :] / importance_matrix[j, :].sum()
                H = -np.sum(probs * np.log(probs + 1e-8))
                per_factor.append(1 - H / np.log(len(probs)))
            else:
                per_factor.append(0)
        return np.mean(per_factor)

    @staticmethod
    def _compute_informativeness(latents: np.ndarray, factors: np.ndarray) -> float:
        """Compute informativeness as prediction accuracy."""
        scores = []
        for j in range(factors.shape[1]):
            rf = RandomForestRegressor(n_estimators=10, max_depth=5)
            split = int(0.8 * len(latents))
            rf.fit(latents[:split], factors[:split, j])
            score = rf.score(latents[split:], factors[split:, j])
            scores.append(max(0, score))  # Clip negative R² to 0
        return np.mean(scores)


### 2. Manifold Analysis Tools

class ManifoldAnalyzer:
    """Tools for analyzing the geometry of learned manifolds."""

    @staticmethod
    def estimate_intrinsic_dimension(data: torch.Tensor, method: str = 'mle',
                                    k: int = 10) -> float:
        """
        Estimate intrinsic dimensionality of data manifold.

        Args:
            data: Data points (n_samples, n_features)
            method: 'mle', 'correlation', or 'pca'
            k: Number of neighbors for MLE method

        Returns:
            Estimated intrinsic dimension
        """
        data_np = data.detach().cpu().numpy()

        if method == 'mle':
            # Maximum Likelihood Estimation (Levina-Bickel)
            n = len(data_np)
            distances = []

            for i in range(n):
                # Compute distances to all other points
                dists = np.linalg.norm(data_np - data_np[i], axis=1)
                dists = np.sort(dists)[1:k+1]  # Exclude self, take k nearest
                distances.append(dists)

            distances = np.array(distances)

            # MLE estimator
            estimates = []
            for j in range(1, k):
                log_ratio = np.log(distances[:, j] / distances[:, :j].max(axis=1))
                estimate = -1 / np.mean(log_ratio)
                estimates.append(estimate)

            return np.mean(estimates)

        elif method == 'correlation':
            # Correlation dimension
            n = len(data_np)
            max_dist = np.max(np.linalg.norm(data_np - data_np[0], axis=1))
            r_values = np.logspace(-3, np.log10(max_dist), 50)

            counts = []
            for r in r_values:
                count = 0
                for i in range(n):
                    distances = np.linalg.norm(data_np - data_np[i], axis=1)
                    count += np.sum(distances < r) - 1  # Exclude self
                counts.append(count / (n * (n - 1)))

            # Fit slope in log-log space
            valid_idx = np.where((np.array(counts) > 0) & (np.array(counts) < 1))[0]
            if len(valid_idx) > 2:
                slope, _ = np.polyfit(np.log(r_values[valid_idx]),
                                     np.log(np.array(counts)[valid_idx]), 1)
                return slope
            return -1

        elif method == 'pca':
            # PCA-based estimation
            centered = data_np - data_np.mean(axis=0)
            cov = np.dot(centered.T, centered) / len(centered)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]

            # Find elbow in explained variance
            total_var = eigenvalues.sum()
            explained_var = np.cumsum(eigenvalues) / total_var

            # Find dimension capturing 95% variance
            dim = np.argmax(explained_var >= 0.95) + 1
            return float(dim)

    @staticmethod
    def compute_local_tangent_spaces(data: torch.Tensor, k: int = 20) -> List[torch.Tensor]:
        """
        Compute local tangent spaces using PCA on neighborhoods.

        Returns:
            List of tangent space basis vectors for each point
        """
        data_np = data.detach().cpu().numpy()
        n = len(data_np)
        tangent_spaces = []

        for i in range(n):
            # Find k nearest neighbors
            distances = np.linalg.norm(data_np - data_np[i], axis=1)
            neighbors = np.argsort(distances)[1:k+1]

            # Local PCA
            local_data = data_np[neighbors] - data_np[i]
            _, _, V = np.linalg.svd(local_data, full_matrices=False)

            tangent_spaces.append(torch.from_numpy(V).float())

        return tangent_spaces

    @staticmethod
    def compute_geodesic_distance(data: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Approximate geodesic distances using graph shortest paths.

        Returns:
            Geodesic distance matrix (n_samples, n_samples)
        """
        data_np = data.detach().cpu().numpy()
        n = len(data_np)

        # Build k-NN graph
        graph = np.full((n, n), np.inf)
        np.fill_diagonal(graph, 0)

        for i in range(n):
            distances = np.linalg.norm(data_np - data_np[i], axis=1)
            neighbors = np.argsort(distances)[1:k+1]
            for j in neighbors:
                graph[i, j] = distances[j]
                graph[j, i] = distances[j]  # Symmetric

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if graph[i, k] + graph[k, j] < graph[i, j]:
                        graph[i, j] = graph[i, k] + graph[k, j]

        return torch.from_numpy(graph).float()

    @staticmethod
    def compute_curvature(data: torch.Tensor, tangent_spaces: List[torch.Tensor],
                         epsilon: float = 0.01) -> torch.Tensor:
        """
        Estimate local curvature using parallel transport.

        Returns:
            Curvature estimates for each point
        """
        n = len(data)
        curvatures = []

        for i in range(n):
            # Get tangent space at point i
            T_i = tangent_spaces[i]

            # Find nearby points
            distances = torch.norm(data - data[i], dim=1)
            neighbors = torch.where(distances < epsilon)[0]

            if len(neighbors) < 2:
                curvatures.append(0.0)
                continue

            # Estimate curvature from tangent space variation
            curvature_sum = 0.0
            for j in neighbors:
                if i != j:
                    T_j = tangent_spaces[j.item()]
                    # Compute principal angles between tangent spaces
                    U, S, V = torch.svd(torch.mm(T_i.T, T_j))
                    angles = torch.acos(torch.clamp(S, -1, 1))
                    curvature_sum += angles.mean().item() / distances[j].item()

            curvatures.append(curvature_sum / len(neighbors))

        return torch.tensor(curvatures)


### 3. VAE Geometry and Optimization

class VAEGeometry:
    """Advanced VAE architectures and geometric analysis."""

    @staticmethod
    def compute_posterior_collapse_degree(mu: torch.Tensor, logvar: torch.Tensor,
                                         threshold: float = 0.1) -> Dict[str, float]:
        """
        Measure degree of posterior collapse in VAE.

        Args:
            mu: Mean of posterior (batch_size, latent_dim)
            logvar: Log-variance of posterior (batch_size, latent_dim)
            threshold: KL threshold for considering dimension active

        Returns:
            Dictionary with collapse metrics
        """
        # Compute KL divergence per dimension
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = kl_per_dim.mean(dim=0)  # Average over batch

        # Count active dimensions
        active_dims = (kl_per_dim > threshold).sum().item()
        total_dims = kl_per_dim.shape[0]

        # Compute average KL for active vs inactive
        active_mask = kl_per_dim > threshold
        avg_active_kl = kl_per_dim[active_mask].mean().item() if active_mask.any() else 0
        avg_inactive_kl = kl_per_dim[~active_mask].mean().item() if (~active_mask).any() else 0

        return {
            'active_dimensions': active_dims,
            'total_dimensions': total_dims,
            'collapse_ratio': 1 - (active_dims / total_dims),
            'avg_active_kl': avg_active_kl,
            'avg_inactive_kl': avg_inactive_kl,
            'total_kl': kl_per_dim.sum().item()
        }

    @staticmethod
    def beta_vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                     logvar: torch.Tensor, beta: float = 4.0,
                     capacity: Optional[float] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Beta-VAE loss with optional capacity constraint.

        Returns:
            Total loss and dictionary of loss components
        """
        batch_size = x.shape[0]

        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Apply capacity constraint if specified
        if capacity is not None:
            kl_loss = torch.abs(kl_loss - capacity)

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_per_dim': (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0).tolist()
        }

    @staticmethod
    def factor_vae_loss(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                       recon_loss: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
        """
        Factor-VAE loss with total correlation penalty.
        """
        batch_size = z.shape[0]

        # Standard VAE losses
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Total Correlation estimation
        # Sample from aggregate posterior
        z_perm = z[torch.randperm(batch_size)]

        # Discriminator to distinguish q(z) from q(z_1)...q(z_d)
        # This is a simplified version - in practice, use a separate discriminator network
        log_qz = VAEGeometry._log_density_gaussian(z, mu, logvar).sum(1)
        log_qz_perm = VAEGeometry._log_density_gaussian(z_perm, mu, logvar).sum(1)

        # Total correlation as discriminator accuracy
        tc_loss = (log_qz - log_qz_perm).mean()

        return recon_loss + kl_loss + gamma * tc_loss

    @staticmethod
    def _log_density_gaussian(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Log density of Gaussian."""
        normalization = -0.5 * (np.log(2 * np.pi) + logvar)
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * ((x - mu).pow(2) * inv_var)
        return log_density


### 4. Interpolation and Traversal Tools

class LatentInterpolation:
    """Advanced interpolation methods for latent spaces."""

    @staticmethod
    def spherical_interpolation(z1: torch.Tensor, z2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Spherical linear interpolation (SLERP) for normalized latent codes.

        Args:
            z1, z2: Latent codes to interpolate between
            t: Interpolation parameter(s) in [0, 1]

        Returns:
            Interpolated latent codes
        """
        # Normalize
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)

        # Compute angle
        cos_omega = (z1_norm * z2_norm).sum(-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(cos_omega)

        # Compute interpolation weights
        sin_omega = torch.sin(omega) + 1e-8
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1)

        w1 = torch.sin((1 - t) * omega) / sin_omega
        w2 = torch.sin(t * omega) / sin_omega

        # Handle edge case where omega ≈ 0
        linear_mask = omega.squeeze() < 1e-4
        if linear_mask.any():
            w1[linear_mask] = 1 - t[linear_mask]
            w2[linear_mask] = t[linear_mask]

        return w1 * z1 + w2 * z2

    @staticmethod
    def geodesic_interpolation(z1: torch.Tensor, z2: torch.Tensor, t: torch.Tensor,
                              flow_model: nn.Module) -> torch.Tensor:
        """
        Interpolate along geodesic in learned manifold using normalizing flow.

        Args:
            z1, z2: Start and end points
            t: Interpolation parameter(s)
            flow_model: Normalizing flow mapping to base space

        Returns:
            Points along geodesic
        """
        # Map to base space where geodesics are straight lines
        with torch.no_grad():
            w1 = flow_model.inverse(z1)
            w2 = flow_model.inverse(z2)

            # Linear interpolation in base space
            if t.dim() == 0:
                t = t.unsqueeze(0).unsqueeze(1)
            elif t.dim() == 1:
                t = t.unsqueeze(1)

            w_t = (1 - t) * w1.unsqueeze(0) + t * w2.unsqueeze(0)

            # Map back to latent space
            z_t = flow_model.forward(w_t.view(-1, w1.shape[-1]))

        return z_t.view(t.shape[0], -1, z1.shape[-1]).squeeze()

    @staticmethod
    def bezier_interpolation(control_points: List[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        """
        Bezier curve interpolation through multiple control points.
        """
        n = len(control_points) - 1
        result = torch.zeros_like(control_points[0])

        for i, point in enumerate(control_points):
            # Bernstein polynomial coefficient
            coeff = VAEGeometry._binomial(n, i) * (t ** i) * ((1 - t) ** (n - i))
            result += coeff.unsqueeze(-1) * point

        return result

    @staticmethod
    def _binomial(n: int, k: int) -> float:
        """Binomial coefficient."""
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))


### 5. Latent Space Visualization

class LatentVisualization:
    """Advanced visualization tools for latent spaces."""

    @staticmethod
    def plot_latent_traversals(decoder: nn.Module, latent_dim: int, n_samples: int = 10,
                              ranges: Tuple[float, float] = (-3, 3), device: str = 'cpu'):
        """
        Visualize latent dimension traversals.
        """
        fig, axes = plt.subplots(latent_dim, n_samples, figsize=(n_samples * 2, latent_dim * 2))

        for dim in range(latent_dim):
            # Sample base latent
            z = torch.zeros(1, latent_dim, device=device)

            # Traverse dimension
            values = torch.linspace(ranges[0], ranges[1], n_samples)

            for i, val in enumerate(values):
                z_copy = z.clone()
                z_copy[0, dim] = val

                with torch.no_grad():
                    reconstruction = decoder(z_copy)

                # Convert to image
                if reconstruction.shape[1] == 1:  # Grayscale
                    img = reconstruction.squeeze().cpu().numpy()
                else:  # RGB
                    img = reconstruction.squeeze().permute(1, 2, 0).cpu().numpy()

                axes[dim, i].imshow(img, cmap='gray' if reconstruction.shape[1] == 1 else None)
                axes[dim, i].axis('off')
                axes[dim, i].set_title(f'd={dim}, v={val:.1f}', fontsize=8)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_disentanglement_matrix(importance_matrix: np.ndarray, factor_names: List[str] = None,
                                   latent_names: List[str] = None):
        """
        Visualize factor-latent importance matrix.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(importance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=latent_names if latent_names else [f'z{i}' for i in range(importance_matrix.shape[1])],
                   yticklabels=factor_names if factor_names else [f'f{i}' for i in range(importance_matrix.shape[0])],
                   ax=ax)

        ax.set_xlabel('Latent Dimensions')
        ax.set_ylabel('Factors')
        ax.set_title('Factor-Latent Importance Matrix')

        return fig

    @staticmethod
    def plot_manifold_density(latents: torch.Tensor, labels: Optional[torch.Tensor] = None,
                             method: str = 'umap', perplexity: int = 30):
        """
        Plot 2D manifold embedding with density estimation.
        """
        from sklearn.manifold import TSNE
        try:
            import umap
            has_umap = True
        except ImportError:
            has_umap = False

        latents_np = latents.detach().cpu().numpy()

        # Dimensionality reduction
        if method == 'umap' and has_umap:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(latents_np)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embedded = reducer.fit_transform(latents_np)
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(latents_np)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            scatter = axes[0].scatter(embedded[:, 0], embedded[:, 1], c=labels_np,
                                     cmap='tab10', alpha=0.7, s=20)
            axes[0].legend(*scatter.legend_elements(), title="Classes")
        else:
            axes[0].scatter(embedded[:, 0], embedded[:, 1], alpha=0.7, s=20)

        axes[0].set_title(f'Latent Space Embedding ({method.upper()})')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')

        # Density plot
        from scipy.stats import gaussian_kde
        xy = embedded.T
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = embedded[idx, 0], embedded[idx, 1], z[idx]

        scatter = axes[1].scatter(x, y, c=z, s=20, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=axes[1])
        axes[1].set_title('Latent Space Density')
        axes[1].set_xlabel('Component 1')
        axes[1].set_ylabel('Component 2')

        return fig


### 6. Encoder-Decoder Alignment

class EncoderDecoderAlignment:
    """Tools for analyzing and improving encoder-decoder alignment."""

    @staticmethod
    def compute_cycle_consistency(encoder: nn.Module, decoder: nn.Module,
                                 data: torch.Tensor, latent_samples: torch.Tensor) -> Dict:
        """
        Measure cycle consistency in both directions.
        """
        with torch.no_grad():
            # Data -> Latent -> Data cycle
            z_encoded = encoder(data)
            x_reconstructed = decoder(z_encoded)
            z_re_encoded = encoder(x_reconstructed)

            data_consistency = F.mse_loss(x_reconstructed, data)
            latent_consistency_forward = F.mse_loss(z_re_encoded, z_encoded)

            # Latent -> Data -> Latent cycle
            x_generated = decoder(latent_samples)
            z_encoded_gen = encoder(x_generated)
            x_re_generated = decoder(z_encoded_gen)

            latent_consistency_backward = F.mse_loss(z_encoded_gen, latent_samples)
            generation_consistency = F.mse_loss(x_re_generated, x_generated)

        return {
            'data_cycle_loss': data_consistency.item(),
            'latent_cycle_loss_forward': latent_consistency_forward.item(),
            'latent_cycle_loss_backward': latent_consistency_backward.item(),
            'generation_cycle_loss': generation_consistency.item()
        }

    @staticmethod
    def compute_jacobian_metrics(decoder: nn.Module, z: torch.Tensor) -> Dict:
        """
        Analyze decoder Jacobian for local geometry understanding.
        """
        z.requires_grad_(True)
        x = decoder(z)

        batch_size = z.shape[0]
        latent_dim = z.shape[1]
        output_dim = x.numel() // batch_size

        jacobians = []

        for i in range(batch_size):
            jacobian = torch.zeros(output_dim, latent_dim)

            for j in range(output_dim):
                if x.grad is not None:
                    x.grad.zero_()

                x_flat = x.view(batch_size, -1)
                grad = torch.autograd.grad(x_flat[i, j], z, retain_graph=True)[0]
                jacobian[j] = grad[i]

            jacobians.append(jacobian)

        jacobians = torch.stack(jacobians)

        # Compute metrics
        singular_values = []
        ranks = []
        condition_numbers = []

        for jacobian in jacobians:
            U, S, V = torch.svd(jacobian)
            singular_values.append(S)
            ranks.append((S > 1e-5).sum().item())
            condition_numbers.append((S[0] / (S[-1] + 1e-8)).item())

        return {
            'mean_rank': np.mean(ranks),
            'mean_condition_number': np.mean(condition_numbers),
            'singular_value_decay': torch.stack(singular_values).mean(0).tolist()
        }


### 7. Advanced VAE Architectures

class HypersphericalVAE(nn.Module):
    """VAE with von Mises-Fisher posterior for hyperspherical latent space."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to hyperspherical parameters."""
        h = self.encoder(x)

        # Split into mean direction and concentration
        mu = h[:, :self.latent_dim]
        mu = F.normalize(mu, p=2, dim=-1)  # Project to unit sphere

        # Concentration parameter (like inverse variance)
        kappa = F.softplus(h[:, self.latent_dim:].mean(-1, keepdim=True)) + 1

        return mu, kappa

    def sample_vmf(self, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Sample from von Mises-Fisher distribution."""
        batch_size = mu.shape[0]
        dim = mu.shape[1]

        # Sample using rejection sampling (simplified)
        # For full implementation, use specialized VMF sampling
        w = self._sample_weight(dim, kappa)
        v = torch.randn(batch_size, dim - 1, device=mu.device)
        v = F.normalize(v, p=2, dim=-1)

        # Construct sample
        z = torch.cat([w, v * torch.sqrt(1 - w**2)], dim=-1)

        # Rotate to align with mean direction
        # (simplified - full implementation needs Householder reflection)
        z = z * mu

        return z

    def _sample_weight(self, dim: int, kappa: torch.Tensor) -> torch.Tensor:
        """Sample weight component for VMF."""
        # Simplified sampling - use rejection sampling in practice
        return torch.ones_like(kappa) * 0.9  # Placeholder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, kappa = self.encode(x)
        z = self.sample_vmf(mu, kappa)
        recon = self.decoder(z)
        return recon, mu, kappa


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    n_samples = 1000
    latent_dim = 10
    n_factors = 5

    # Simulated latents and factors
    latents = torch.randn(n_samples, latent_dim)
    factors = torch.randint(0, 3, (n_samples, n_factors))

    # Compute disentanglement metrics
    metrics = DisentanglementMetrics()
    mig_score = metrics.compute_mig(latents.numpy(), factors.numpy())
    sap_score = metrics.compute_sap(latents.numpy(), factors.numpy())
    dci_scores = metrics.compute_dci(latents.numpy(), factors.numpy())

    print(f"MIG Score: {mig_score:.3f}")
    print(f"SAP Score: {sap_score:.3f}")
    print(f"DCI Scores: {dci_scores}")

    # Analyze manifold geometry
    analyzer = ManifoldAnalyzer()
    intrinsic_dim = analyzer.estimate_intrinsic_dimension(latents, method='mle')
    print(f"Intrinsic dimension: {intrinsic_dim:.2f}")

    # Visualize latent space
    visualizer = LatentVisualization()
    fig = visualizer.plot_manifold_density(latents, labels=factors[:, 0])
    plt.show()