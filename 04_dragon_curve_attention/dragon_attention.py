"""
Dragon Curve Attention: Hierarchical Attention using Fractal Weighting

This module implements attention mechanisms that use dragon curve fractal patterns
to create natural hierarchical weightings. Unlike standard uniform attention or
learned attention patterns, dragon curve attention leverages the self-similar
hierarchical structure of the dragon fractal.

Key Innovation:
    The dragon curve naturally separates global structure (early turns) from local
    details (later turns). By using the curve's hierarchical depth as attention
    weights, we get natural multi-scale attention without explicit multi-head design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
import os

# Add dragon_curve to path
sys.path.insert(0, os.path.dirname(__file__))

from dragon_curve import DragonCurveGenerator


class DragonCurveAttention(nn.Module):
    """
    Attention mechanism using dragon curve fractal patterns for hierarchical weighting.

    Instead of uniform attention weights or pure learned weights, this module
    uses the hierarchical structure of the dragon curve to bias attention:
    - Earlier positions in the curve (global structure) get higher base weights
    - Later positions (local details) get lower base weights
    - The fractal pattern provides natural multi-scale attention

    Args:
        dim_in: Input feature dimension
        dim_qk: Query/Key dimension per head
        dim_v: Value dimension per head
        nb_heads: Number of attention heads
        max_seq_len: Maximum sequence length (determines dragon iteration)
        decay_rate: How quickly hierarchical weights decay (0 < rate < 1)
        use_fractal_weights: Whether to apply dragon curve weighting
        dropout: Dropout probability

    Example:
        >>> attn = DragonCurveAttention(dim_in=64, dim_qk=32, dim_v=32, nb_heads=4)
        >>> x = torch.randn(8, 128, 64)
        >>> out = attn(x)  # Applies dragon curve hierarchical weighting
    """

    def __init__(
        self,
        dim_in: int,
        dim_qk: int,
        dim_v: int,
        nb_heads: int = 4,
        max_seq_len: int = 1024,
        decay_rate: float = 0.85,
        use_fractal_weights: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.nb_heads = nb_heads
        self.decay_rate = decay_rate
        self.use_fractal_weights = use_fractal_weights

        # Total dimensions for multi-head
        self.total_qk_dim = dim_qk * nb_heads
        self.total_v_dim = dim_v * nb_heads

        # Projections for Q, K, V
        self.W_q = nn.Linear(dim_in, self.total_qk_dim, bias=False)
        self.W_k = nn.Linear(dim_in, self.total_qk_dim, bias=False)
        self.W_v = nn.Linear(dim_in, self.total_v_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(self.total_v_dim, dim_in, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Dragon curve generator
        # Determine iteration based on max_seq_len: find smallest iteration where 2^iter >= max_seq_len
        import math
        self.dragon_iteration = max(1, math.ceil(math.log2(max_seq_len)))
        self.dragon_generator = DragonCurveGenerator(max_iterations=self.dragon_iteration)

    def _get_fractal_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get dragon curve hierarchical weights for given sequence length.

        Args:
            seq_len: Actual sequence length
            device: Device to create tensors on

        Returns:
            Weight tensor of shape (seq_len,) for modulating attention
        """
        # Find appropriate dragon iteration for this sequence length
        import math
        iteration = max(1, math.ceil(math.log2(seq_len)))
        iteration = min(iteration, self.dragon_iteration)

        # Get hierarchical weights from dragon curve
        weights = self.dragon_generator.get_hierarchical_weights(
            iteration=iteration,
            decay_rate=self.decay_rate
        )

        # Truncate or pad to match seq_len
        if len(weights) > seq_len:
            weights = weights[:seq_len]
        elif len(weights) < seq_len:
            # Pad with uniform weights
            padding = torch.ones(seq_len - len(weights)) / (seq_len - len(weights))
            weights = torch.cat([weights, padding])
            # Renormalize
            weights = weights / weights.sum()

        return weights.to(device)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply dragon curve modulated attention.

        Args:
            x: Input tensor (batch_size, seq_len, dim_in)
            mask: Optional attention mask

        Returns:
            Output tensor (batch_size, seq_len, dim_in)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, total_qk_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.nb_heads, self.dim_qk).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nb_heads, self.dim_qk).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nb_heads, self.dim_v).transpose(1, 2)

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_qk ** 0.5)
        # Shape: (batch, nb_heads, seq_len, seq_len)

        # Apply fractal weights if enabled
        if self.use_fractal_weights:
            # Get dragon curve hierarchical weights
            fractal_weights = self._get_fractal_weights(seq_len, device)

            # Modulate attention scores by fractal weights
            # fractal_weights[i] modulates how much position i is attended TO
            # Broadcasting: (seq_len,) -> (1, 1, 1, seq_len)
            fractal_weights = fractal_weights.view(1, 1, 1, seq_len)
            scores = scores * fractal_weights

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.total_v_dim)

        # Apply output projection
        output = self.W_o(attn_output)

        return output

    def get_fractal_info(self, seq_len: int) -> dict:
        """
        Get information about the dragon curve fractal used for this sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Dictionary with fractal statistics
        """
        import math
        iteration = max(1, math.ceil(math.log2(seq_len)))
        iteration = min(iteration, self.dragon_iteration)

        pattern = self.dragon_generator.get_pattern(iteration)
        coords = self.dragon_generator.get_coordinates(iteration)
        weights = self.dragon_generator.get_hierarchical_weights(iteration, self.decay_rate)

        return {
            'iteration': iteration,
            'num_turns': len(pattern),
            'num_points': len(coords),
            'fractal_dimension': self.dragon_generator.get_fractal_dimension(),
            'weight_range': (weights.min().item(), weights.max().item()),
            'weight_entropy': -(weights * torch.log(weights + 1e-10)).sum().item()
        }
