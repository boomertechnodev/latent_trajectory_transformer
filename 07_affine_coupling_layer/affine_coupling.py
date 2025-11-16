"""
Affine Coupling Layer: RealNVP-style transformations for normalizing flows

This module implements affine coupling layers, the fundamental building block
of RealNVP (Real-valued Non-Volume Preserving) normalizing flows.

Key Idea:
    Split input into two parts using a binary mask, use the first part (unchanged)
    to compute scale and shift parameters via a neural network, then transform
    the second part with an affine transformation.

Transformation:
    Given input x and binary mask m:
        x_frozen = x * m          # Unchanged part
        x_active = x * (1 - m)    # Part to transform

        scale, shift = NN(x_frozen, time)  # Conditioned on frozen part and time

        # Forward (data → latent):
        y_active = x_active * exp(scale) + shift
        y = x_frozen + y_active

        # Inverse (latent → data):
        x_active = (y_active - shift) / exp(scale)

Jacobian Determinant:
    The log-determinant is simply sum(scale), making likelihood computation efficient.

Time Conditioning:
    Time features are concatenated to the NN input, allowing the transformation
    to smoothly vary over continuous time t ∈ [0, 1].

Stability:
    - Scale is clamped: tanh(raw_scale / 2) * scale_limit
    - Final layer initialized to zeros (starts near identity)
    - Optional batch normalization for input stability
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class ScaleShiftNetwork(nn.Module):
    """
    Neural network that computes scale and shift parameters for affine coupling.

    Takes frozen input dimensions (plus optional time features) and outputs
    scale and shift for the active dimensions.

    Args:
        input_dim: Dimension of frozen input
        output_dim: Dimension of active output (scale and shift)
        hidden_dim: Hidden layer dimension
        time_dim: Dimension of time features (0 = no time conditioning)
        num_layers: Number of hidden layers (default: 3)
        scale_limit: Maximum absolute scale value (default: 2.0)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        time_dim: int = 0,
        num_layers: int = 3,
        scale_limit: float = 2.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.scale_limit = scale_limit

        # Input: frozen dimensions + optional time features
        net_input_dim = input_dim + time_dim

        # Build MLP
        layers = []

        # Input layer
        layers.append(nn.Linear(net_input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer: produces both scale and shift
        # Output dimension is 2 * output_dim (scale and shift)
        output_layer = nn.Linear(hidden_dim, 2 * output_dim)

        # Initialize output layer to zeros (start near identity transformation)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_frozen: torch.Tensor,
        time_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale and shift parameters.

        Args:
            x_frozen: Frozen input dimensions (batch_size, input_dim)
            time_features: Optional time features (batch_size, time_dim)

        Returns:
            scale: Scale parameters (batch_size, output_dim)
            shift: Shift parameters (batch_size, output_dim)
        """
        # Concatenate inputs
        if time_features is not None:
            net_input = torch.cat([x_frozen, time_features], dim=-1)
        else:
            net_input = x_frozen

        # Forward through network
        output = self.net(net_input)  # (batch_size, 2 * output_dim)

        # Split into scale and shift
        scale, shift = output.chunk(2, dim=-1)

        # Clamp scale for numerical stability
        # tanh(scale / 2) * scale_limit keeps scale in [-scale_limit, scale_limit]
        scale = torch.tanh(scale / 2.0) * self.scale_limit

        return scale, shift


class AffineCouplingLayer(nn.Module):
    """
    Single affine coupling layer with optional time conditioning.

    Implements the RealNVP coupling transformation:
        y = x_frozen + (x_active * exp(scale) + shift)

    where scale and shift are computed from x_frozen (and optional time).

    Args:
        dim: Total dimension of input/output
        mask: Binary mask (1 = frozen, 0 = active), shape (dim,)
        hidden_dim: Hidden dimension for scale/shift network
        time_dim: Dimension of time features (0 = no time conditioning)
        num_layers: Number of layers in scale/shift network
        scale_limit: Maximum absolute scale value

    Shape:
        Forward: (batch_size, dim) → (batch_size, dim), log_det (batch_size,)
        Inverse: (batch_size, dim) → (batch_size, dim)
    """

    def __init__(
        self,
        dim: int,
        mask: torch.Tensor,
        hidden_dim: int,
        time_dim: int = 0,
        num_layers: int = 3,
        scale_limit: float = 2.0
    ):
        super().__init__()

        if mask.shape != (dim,):
            raise ValueError(f"Mask shape must be ({dim},), got {mask.shape}")

        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Mask must be binary (0 or 1)")

        self.dim = dim
        self.time_dim = time_dim

        # Register mask as buffer (moves with model to GPU/CPU)
        self.register_buffer('mask', mask.float())

        # Compute dimensions
        self.frozen_dim = int(mask.sum().item())
        self.active_dim = dim - self.frozen_dim

        if self.frozen_dim == 0 or self.active_dim == 0:
            raise ValueError("Mask must split dimensions (neither all 0 nor all 1)")

        # Scale/shift network
        self.scale_shift_net = ScaleShiftNetwork(
            input_dim=self.frozen_dim,
            output_dim=self.active_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            scale_limit=scale_limit
        )

    def forward(
        self,
        x: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        inverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or inverse transformation.

        Args:
            x: Input tensor (batch_size, dim)
            time_features: Optional time features (batch_size, time_dim)
            inverse: If True, compute inverse transformation

        Returns:
            y: Output tensor (batch_size, dim)
            log_det: Log-determinant of Jacobian (batch_size,)
        """
        batch_size, dim = x.shape
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"

        # Split into frozen and active parts
        x_frozen = x * self.mask  # (batch_size, dim)
        x_active = x * (1 - self.mask)  # (batch_size, dim)

        # Extract non-zero elements
        # frozen_indices and active_indices are complementary
        frozen_vals = x_frozen[:, self.mask.bool()]  # (batch_size, frozen_dim)
        active_vals = x_active[:, (~self.mask.bool())]  # (batch_size, active_dim)

        # Compute scale and shift from frozen part
        scale, shift = self.scale_shift_net(frozen_vals, time_features)
        # scale: (batch_size, active_dim)
        # shift: (batch_size, active_dim)

        if not inverse:
            # Forward: y_active = x_active * exp(scale) + shift
            y_active = active_vals * torch.exp(scale) + shift

            # Log-determinant: sum of scale (from exp Jacobian)
            log_det = scale.sum(dim=-1)  # (batch_size,)
        else:
            # Inverse: x_active = (y_active - shift) / exp(scale)
            y_active = (active_vals - shift) * torch.exp(-scale)

            # Log-determinant: -sum of scale (inverse transformation)
            log_det = -scale.sum(dim=-1)  # (batch_size,)

        # Reconstruct full output
        y = x_frozen.clone()
        y[:, (~self.mask.bool())] = y_active

        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        time_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inverse transformation (convenience method).

        Args:
            y: Latent tensor (batch_size, dim)
            time_features: Optional time features

        Returns:
            x: Data tensor (batch_size, dim)
        """
        x, _ = self.forward(y, time_features, inverse=True)
        return x


def create_checkerboard_mask(dim: int, invert: bool = False) -> torch.Tensor:
    """
    Create checkerboard binary mask for coupling layers.

    Alternates 1, 0, 1, 0, ... to split dimensions evenly.

    Args:
        dim: Total dimension
        invert: If True, start with 0 instead of 1

    Returns:
        Binary mask tensor (dim,)
    """
    mask = torch.zeros(dim)
    if invert:
        mask[1::2] = 1  # Odd indices
    else:
        mask[0::2] = 1  # Even indices
    return mask
