"""
Time-Aware Transform: Multi-Scale Sinusoidal Time Embeddings

This module implements time embeddings for continuous-time models, inspired by
Transformer positional encodings but adapted for continuous time in [0, 1].

Key Innovation:
    Instead of discrete position indices, we embed continuous time values using
    exponentially-spaced frequency bands. This provides multi-scale temporal
    resolution similar to wavelets or Fourier analysis.

Frequency Design:
    - Frequencies span from freq_min (e.g., 1 Hz) to freq_max (e.g., 1000 Hz)
    - Exponential spacing: freq_i = freq_min * (freq_max/freq_min)^(i/num_freqs)
    - Low frequencies capture slow changes (long-term trends)
    - High frequencies capture fast changes (local variations)

Embedding Construction:
    For time t ∈ [0, 1] and frequency ω:
        emb = [sin(2π * ω * t), cos(2π * ω * t)]

    Output dimension = 2 * num_frequencies (sin and cos for each frequency)

Properties:
    - Unique: Different times have different embeddings (injective mapping)
    - Smooth: Nearby times have similar embeddings (Lipschitz continuous)
    - Multi-scale: Captures both slow and fast temporal variations
    - Rotation-invariant: sin/cos pairs provide phase information

Use Cases:
    - Condition normalizing flows on time: p(z|t) = flow(noise; emb(t))
    - Time-dependent SDEs: dz = f(z, emb(t))dt + σ(z, emb(t))dW
    - Temporal attention: attention weights based on time distance in embedding space
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    """
    Multi-scale sinusoidal time embedding for continuous time in [0, 1].

    Creates a time-dependent feature representation using exponentially-spaced
    frequency bands, providing multi-scale temporal resolution from slow to fast
    changes.

    Args:
        time_dim: Output dimension (must be even, since we use sin/cos pairs)
        freq_min: Minimum frequency in Hz (default: 1.0)
        freq_max: Maximum frequency in Hz (default: 1000.0)

    Shape:
        Input: (batch_size,) or (batch_size, 1) - time values in [0, 1]
        Output: (batch_size, time_dim) - time embeddings

    Example:
        >>> time_emb = TimeEmbedding(time_dim=64, freq_min=1.0, freq_max=1000.0)
        >>> t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        >>> emb = time_emb(t)
        >>> print(emb.shape)  # torch.Size([5, 64])
    """

    def __init__(
        self,
        time_dim: int,
        freq_min: float = 1.0,
        freq_max: float = 1000.0
    ):
        super().__init__()

        if time_dim % 2 != 0:
            raise ValueError(f"time_dim must be even (for sin/cos pairs), got {time_dim}")

        if freq_min <= 0:
            raise ValueError(f"freq_min must be positive, got {freq_min}")

        if freq_max <= freq_min:
            raise ValueError(f"freq_max must be > freq_min, got {freq_max} <= {freq_min}")

        self.time_dim = time_dim
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Number of frequency bands (half of time_dim since we use sin and cos)
        self.num_frequencies = time_dim // 2

        # Compute exponentially-spaced frequencies
        # freq_i = freq_min * (freq_max / freq_min)^(i / (num_freqs - 1))
        # This spans from freq_min to freq_max in log-space
        if self.num_frequencies == 1:
            frequencies = torch.tensor([freq_min])
        else:
            # Log-space interpolation
            log_freq_min = math.log(freq_min)
            log_freq_max = math.log(freq_max)
            log_frequencies = torch.linspace(log_freq_min, log_freq_max, self.num_frequencies)
            frequencies = torch.exp(log_frequencies)

        # Register frequencies as buffer (will move with model to GPU/CPU automatically)
        # Shape: (num_frequencies,)
        self.register_buffer('frequencies', frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embeddings for given time values.

        Args:
            t: Time values, shape (batch_size,) or (batch_size, 1)
               Values should be in [0, 1] range

        Returns:
            Time embeddings, shape (batch_size, time_dim)
        """
        # Ensure t is 2D: (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # t shape: (batch_size, 1)
        # frequencies shape: (num_frequencies,)

        # Compute angles: 2π * freq * t
        # Broadcasting: (batch_size, 1) * (num_frequencies,) = (batch_size, num_frequencies)
        angles = 2.0 * math.pi * self.frequencies * t

        # Compute sin and cos
        sin_emb = torch.sin(angles)  # (batch_size, num_frequencies)
        cos_emb = torch.cos(angles)  # (batch_size, num_frequencies)

        # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
        # Shape: (batch_size, 2 * num_frequencies) = (batch_size, time_dim)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)

        return emb

    def get_frequencies(self) -> torch.Tensor:
        """
        Get the frequency bands used in the embedding.

        Returns:
            Frequencies tensor, shape (num_frequencies,)
        """
        return self.frequencies

    def get_frequency_info(self) -> dict:
        """
        Get information about the frequency spectrum.

        Returns:
            Dictionary with frequency statistics
        """
        return {
            'num_frequencies': self.num_frequencies,
            'freq_min': self.freq_min,
            'freq_max': self.freq_max,
            'frequencies': self.frequencies.tolist(),
            'freq_spacing': 'exponential',
            'time_dim': self.time_dim
        }


class LearnableTimeEmbedding(nn.Module):
    """
    Learnable time embedding that projects sinusoidal features through an MLP.

    This combines fixed sinusoidal embeddings (good inductive bias) with learned
    transformations (adaptable to task-specific temporal structure).

    Args:
        time_dim: Dimension of sinusoidal embedding (must be even)
        output_dim: Output dimension after MLP projection
        hidden_dim: Hidden layer dimension (default: time_dim * 2)
        freq_min: Minimum frequency for sinusoidal embedding
        freq_max: Maximum frequency for sinusoidal embedding

    Shape:
        Input: (batch_size,) or (batch_size, 1) - time values in [0, 1]
        Output: (batch_size, output_dim) - learned time features

    Example:
        >>> time_emb = LearnableTimeEmbedding(time_dim=64, output_dim=128)
        >>> t = torch.linspace(0, 1, 100)
        >>> emb = time_emb(t)
        >>> print(emb.shape)  # torch.Size([100, 128])
    """

    def __init__(
        self,
        time_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        freq_min: float = 1.0,
        freq_max: float = 1000.0
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = time_dim * 2

        self.time_dim = time_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Fixed sinusoidal embedding
        self.sinusoidal_emb = TimeEmbedding(
            time_dim=time_dim,
            freq_min=freq_min,
            freq_max=freq_max
        )

        # Learnable MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute learned time embeddings.

        Args:
            t: Time values, shape (batch_size,) or (batch_size, 1)

        Returns:
            Learned time embeddings, shape (batch_size, output_dim)
        """
        # Get sinusoidal embedding
        sin_emb = self.sinusoidal_emb(t)  # (batch_size, time_dim)

        # Project through MLP
        emb = self.mlp(sin_emb)  # (batch_size, output_dim)

        return emb

    def get_sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the intermediate sinusoidal embedding (before MLP).

        Args:
            t: Time values

        Returns:
            Sinusoidal embedding, shape (batch_size, time_dim)
        """
        return self.sinusoidal_emb(t)
