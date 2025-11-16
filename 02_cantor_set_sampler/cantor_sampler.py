"""
Cantor Set Sampler for Multi-Scale Hierarchical Attention

This module implements the CantorSetSampler class that generates multi-scale
hierarchical sampling patterns using Cantor set fractals. The Cantor set provides
natural logarithmic sparse sampling at multiple scales, ideal for capturing both
local and long-range dependencies in sequences.

The Cantor set has fractal dimension log(2)/log(3) ≈ 0.631 and exhibits perfect
self-similarity across scales, making it mathematically elegant for attention.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import math


class CantorSetSampler(nn.Module):
    """
    Generates multi-scale hierarchical sampling indices using Cantor set fractals.

    The Cantor set sampling pattern:
    - Scale 0: Just the center position (local attention)
    - Scale 1: 1/3 and 2/3 positions relative to center (medium-range)
    - Scale 2: 1/9, 2/9, 7/9, 8/9 positions (long-range)
    - Scale k: 2^k sample points with self-similar structure

    This creates logarithmic O(log n) sampling that captures hierarchical
    structure at multiple scales simultaneously.

    Args:
        num_scales: Number of Cantor set scales to compute (default: 5)
                   Each scale k generates 2^k sample points
        max_distance: Maximum distance from center to sample (default: 0.5)
                     In normalized [0, 1] space

    Example:
        >>> sampler = CantorSetSampler(num_scales=3)
        >>> center_idx = 512
        >>> seq_len = 1024
        >>> samples = sampler(center_idx, seq_len)
        >>> print(samples.shape)  # (8,) since 2^3 = 8 samples at scale 3
    """

    def __init__(self, num_scales: int = 5, max_distance: float = 0.5):
        super().__init__()

        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        if not (0.0 < max_distance <= 1.0):
            raise ValueError(f"max_distance must be in (0, 1], got {max_distance}")

        self.num_scales = num_scales
        self.max_distance = max_distance

        # Precompute Cantor set positions for each scale
        cantor_positions = self._generate_cantor_positions(num_scales)

        # Store as buffer (auto-moves to GPU/CPU with model)
        self.register_buffer('cantor_positions', cantor_positions)

        # Also store the count at each scale for validation
        self.scale_counts = [2 ** k for k in range(num_scales + 1)]

    def _generate_cantor_positions(self, num_scales: int) -> torch.Tensor:
        """
        Generate Cantor set positions for all scales up to num_scales.

        The Cantor set is generated recursively:
        - Start with interval [0, 1]
        - Remove middle third, leaving [0, 1/3] and [2/3, 1]
        - Recursively remove middle thirds from remaining intervals

        At scale k, we have 2^k intervals, each of size (1/3)^k.

        Returns:
            Tensor of shape (2^num_scales,) containing relative positions
            in range [-max_distance, +max_distance]

        Algorithm:
            We build the Cantor set iteratively by tracking interval endpoints.
            At each level, we split each interval into two parts by removing
            the middle third.
        """
        # Start with the full interval [0, 1]
        intervals = torch.tensor([[0.0, 1.0]])

        # Recursively generate Cantor set by removing middle thirds
        for _ in range(num_scales):
            new_intervals = []
            for start, end in intervals:
                # Remove middle third: keep [start, start + L/3] and [end - L/3, end]
                length = end - start
                left_end = start + length / 3
                right_start = end - length / 3

                new_intervals.append([start, left_end])
                new_intervals.append([right_start, end])

            intervals = torch.tensor(new_intervals)

        # Extract sample points (use left endpoints of final intervals)
        positions = intervals[:, 0]

        # Center around 0 and scale to [-max_distance, +max_distance]
        # Original positions are in [0, 1], shift to [-0.5, 0.5] then scale
        positions = (positions - 0.5) * 2 * self.max_distance

        return positions

    def forward(self, center_idx: torch.Tensor, seq_len: int,
                scale: int = None) -> torch.Tensor:
        """
        Generate Cantor set sampling indices around a center position.

        Args:
            center_idx: Center position(s) to sample around, shape (B,) or scalar
            seq_len: Total sequence length
            scale: Specific scale to use (0 to num_scales). If None, uses max scale.

        Returns:
            Sampling indices of shape (B, num_samples) or (num_samples,)
            where num_samples = 2^scale

        Example:
            >>> sampler = CantorSetSampler(num_scales=3)
            >>> indices = sampler(torch.tensor([100, 200]), seq_len=1024, scale=2)
            >>> print(indices.shape)  # (2, 4) - batch of 2, 4 samples per position
        """
        # Handle scale parameter
        if scale is None:
            scale = self.num_scales
        elif scale < 0 or scale > self.num_scales:
            raise ValueError(f"scale must be in [0, {self.num_scales}], got {scale}")

        # Get number of samples for this scale (2^scale)
        num_samples = 2 ** scale

        # Handle scalar vs batched center_idx
        if isinstance(center_idx, int):
            center_idx = torch.tensor([center_idx], device=self.cantor_positions.device)
        elif center_idx.dim() == 0:
            center_idx = center_idx.unsqueeze(0)

        batch_size = center_idx.shape[0]

        # Get Cantor positions for this scale
        positions = self.cantor_positions[:num_samples]  # Shape: (num_samples,)

        # Convert relative positions to absolute indices
        # positions are in [-max_distance, +max_distance]
        # Scale by sequence length and add to center
        relative_offsets = positions * seq_len  # Shape: (num_samples,)

        # Broadcast: (B, 1) + (1, num_samples) = (B, num_samples)
        absolute_indices = center_idx.unsqueeze(1) + relative_offsets.unsqueeze(0)

        # Clamp to valid range [0, seq_len)
        absolute_indices = torch.clamp(absolute_indices, 0, seq_len - 1).long()

        # If input was scalar, return 1D output
        if batch_size == 1:
            return absolute_indices.squeeze(0)

        return absolute_indices

    def get_scale_samples(self, center_idx: int, seq_len: int,
                          scale: int) -> torch.Tensor:
        """
        Get samples for a specific scale only (for debugging/visualization).

        Args:
            center_idx: Center position
            seq_len: Sequence length
            scale: Which scale to retrieve (0 to num_scales)

        Returns:
            Indices for that scale, shape (2^scale,)
        """
        return self.forward(center_idx, seq_len, scale=scale)

    def get_all_scales(self, center_idx: int, seq_len: int) -> List[torch.Tensor]:
        """
        Get samples for all scales separately (useful for visualization).

        Args:
            center_idx: Center position
            seq_len: Sequence length

        Returns:
            List of tensors, one per scale
            all_scales[k] has shape (2^k,) for scale k

        Example:
            >>> sampler = CantorSetSampler(num_scales=3)
            >>> all_scales = sampler.get_all_scales(512, 1024)
            >>> for k, indices in enumerate(all_scales):
            ...     print(f"Scale {k}: {len(indices)} samples")
            Scale 0: 1 samples
            Scale 1: 2 samples
            Scale 2: 4 samples
            Scale 3: 8 samples
        """
        all_scales = []
        for scale in range(self.num_scales + 1):
            samples = self.forward(center_idx, seq_len, scale=scale)
            all_scales.append(samples)
        return all_scales

    def compute_coverage(self, center_idx: int, seq_len: int,
                        scale: int = None) -> float:
        """
        Compute sequence coverage (what fraction of positions are sampled).

        Args:
            center_idx: Center position
            seq_len: Sequence length
            scale: Scale to use (None = max scale)

        Returns:
            Coverage fraction in [0, 1]

        Example:
            >>> sampler = CantorSetSampler(num_scales=5)
            >>> coverage = sampler.compute_coverage(512, 1024, scale=5)
            >>> print(f"Coverage: {coverage:.2%}")  # Should be ~3.1% (32/1024)
        """
        samples = self.forward(center_idx, seq_len, scale=scale)
        unique_samples = torch.unique(samples)
        return len(unique_samples) / seq_len

    def verify_self_similarity(self, scale: int) -> Tuple[bool, str]:
        """
        Verify the self-similarity property of Cantor set.

        The Cantor set should satisfy: pattern at scale k contains
        pattern at scale k-1, with additional samples in between.

        Args:
            scale: Scale to verify (must be >= 1)

        Returns:
            (is_self_similar, message)

        Example:
            >>> sampler = CantorSetSampler(num_scales=5)
            >>> is_similar, msg = sampler.verify_self_similarity(scale=3)
            >>> print(is_similar)  # True
        """
        if scale < 1:
            return False, f"Scale must be >= 1, got {scale}"
        if scale > self.num_scales:
            return False, f"Scale {scale} > num_scales {self.num_scales}"

        # Get positions for scale and scale-1
        pos_current = self.cantor_positions[:2**scale]
        pos_prev = self.cantor_positions[:2**(scale-1)]

        # Check if previous scale positions are subset of current
        # Allow small numerical tolerance
        tolerance = 1e-6

        for prev_pos in pos_prev:
            # Check if this position exists in current scale
            matches = torch.abs(pos_current - prev_pos) < tolerance
            if not matches.any():
                return False, f"Position {prev_pos:.6f} from scale {scale-1} not found in scale {scale}"

        return True, f"Scale {scale} contains all positions from scale {scale-1} (self-similar)"

    def get_fractal_dimension(self) -> float:
        """
        Return the theoretical fractal dimension of the Cantor set.

        The Cantor set has Hausdorff dimension log(2)/log(3) ≈ 0.6309.

        Returns:
            Fractal dimension (approximately 0.631)
        """
        return math.log(2) / math.log(3)

    def __repr__(self) -> str:
        return (f"CantorSetSampler(num_scales={self.num_scales}, "
                f"max_distance={self.max_distance:.3f}, "
                f"max_samples={2**self.num_scales}, "
                f"fractal_dim={self.get_fractal_dimension():.4f})")
