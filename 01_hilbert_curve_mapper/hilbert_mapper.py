"""
Hilbert Curve Mapper for Locality-Preserving 1D to 2D Sequence Mapping

This module implements the HilbertCurveMapper class that transforms 1D sequence
positions into 2D coordinates using a Hilbert space-filling curve. The Hilbert
curve is a fractal that preserves locality: nearby positions in 1D space remain
nearby in 2D space, which is critical for efficient O(n×w²) attention mechanisms.

The implementation uses bit manipulation based on John Skilling's 2004 algorithm
for efficient Hilbert curve generation.
"""

import torch
import torch.nn as nn
from typing import Tuple


class HilbertCurveMapper(nn.Module):
    """
    Maps 1D sequence positions to 2D Hilbert curve coordinates.

    The Hilbert curve is a continuous fractal space-filling curve that maps
    a 1D line onto a 2D plane while preserving locality. This property is
    crucial for attention mechanisms: tokens that are close in the sequence
    will also be close in 2D Euclidean space.

    Args:
        max_seq_len: Maximum sequence length to support (default: 4096)
        order: Hilbert curve order (default: auto-computed from max_seq_len)
               The grid size will be 2^order × 2^order

    Example:
        >>> mapper = HilbertCurveMapper(max_seq_len=256)
        >>> indices = torch.arange(10)
        >>> coords = mapper(indices)  # Shape: (10, 2) with values in [0, 1]
        >>> print(coords)
        tensor([[0.0000, 0.0000],
                [0.0625, 0.0000],
                [0.0625, 0.0625],
                ...])
    """

    def __init__(self, max_seq_len: int = 4096, order: int = None):
        super().__init__()

        # Auto-compute order if not provided
        if order is None:
            # Need 2^(2*order) >= max_seq_len, so order >= log2(sqrt(max_seq_len))
            # Using ceiling division: (bits + 1) // 2
            import math
            bits = math.ceil(math.log2(max_seq_len))
            order = max(1, (bits + 1) // 2)

        self.order = order
        self.grid_size = 2 ** order
        self.max_positions = self.grid_size ** 2

        # Precompute all Hilbert curve coordinates at initialization
        coords = self._generate_hilbert_curve(order)

        # Normalize coordinates to [0, 1] range
        coords_normalized = coords.float() / (self.grid_size - 1)

        # Register as buffer so it moves with model to GPU/CPU automatically
        self.register_buffer('hilbert_coords', coords_normalized)

    def _hilbert_index_to_xy(self, index: int, order: int) -> Tuple[int, int]:
        """
        Convert Hilbert curve index to (x, y) coordinates using bit manipulation.

        Based on corrected algorithm for Hilbert curve generation.

        Args:
            index: 1D position along the Hilbert curve (0 to 2^(2*order) - 1)
            order: Order of the Hilbert curve

        Returns:
            Tuple of (x, y) coordinates in grid space [0, 2^order - 1]

        Algorithm:
            The Hilbert curve is built recursively. At each level, we extract
            2 bits to determine which quadrant, apply appropriate rotations,
            then accumulate offsets.
        """
        x = y = 0
        s = 1  # side length of current square

        for _ in range(order):
            # Extract 2 bits: determines which quadrant (0, 1, 2, or 3)
            rx = 1 & (index >> 1)
            ry = 1 & (index ^ rx)

            # Apply rotation for this quadrant
            if ry == 0:
                if rx == 1:
                    # Reflect vertically and horizontally
                    x = s - 1 - x
                    y = s - 1 - y
                # Swap x and y
                x, y = y, x

            # Add offset for this quadrant
            x += s * rx
            y += s * ry

            # Move to next level (square size doubles)
            index >>= 2
            s *= 2

        return x, y

    def _generate_hilbert_curve(self, order: int) -> torch.Tensor:
        """
        Generate all coordinates for a Hilbert curve of given order.

        Args:
            order: Order of the Hilbert curve (creates 2^order × 2^order grid)

        Returns:
            Tensor of shape (2^(2*order), 2) containing (x, y) coordinates

        Example:
            For order=2 (4×4 grid, 16 positions):
            [[0,0], [1,0], [1,1], [0,1], [0,2], [0,3], [1,3], [1,2],
             [2,2], [2,3], [3,3], [3,2], [3,1], [2,1], [2,0], [3,0]]
        """
        num_positions = 2 ** (2 * order)
        coords = torch.zeros(num_positions, 2, dtype=torch.long)

        for i in range(num_positions):
            x, y = self._hilbert_index_to_xy(i, order)
            coords[i, 0] = x
            coords[i, 1] = y

        return coords

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Map 1D sequence indices to 2D Hilbert curve coordinates.

        Args:
            indices: 1D tensor of sequence positions, shape (N,) or (B, N)
                    Values should be in range [0, max_positions)

        Returns:
            2D coordinates tensor of shape (*indices.shape, 2)
            Coordinates are normalized to [0, 1] range

        Example:
            >>> mapper = HilbertCurveMapper(max_seq_len=64)
            >>> indices = torch.tensor([0, 1, 2, 3])
            >>> coords = mapper(indices)
            >>> # coords[i] gives (x, y) for sequence position i
            >>> print(coords.shape)  # torch.Size([4, 2])
        """
        # Clamp indices to valid range
        indices_clamped = torch.clamp(indices, 0, self.max_positions - 1)

        # Look up precomputed coordinates
        coords = self.hilbert_coords[indices_clamped]

        return coords

    def get_2d_distances(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise 2D Euclidean distances between sequence positions.

        This is useful for finding k-nearest neighbors in fractal attention.

        Args:
            indices: Sequence position indices, shape (N,)

        Returns:
            Pairwise distance matrix, shape (N, N)
            distances[i, j] = Euclidean distance in 2D between positions i and j

        Example:
            >>> mapper = HilbertCurveMapper()
            >>> indices = torch.arange(100)
            >>> dists = mapper.get_2d_distances(indices)
            >>> # Find 5 nearest neighbors for position 50
            >>> neighbors = torch.topk(dists[50], k=5, largest=False)
        """
        coords = self.forward(indices)  # Shape: (N, 2)

        # Compute pairwise distances: sqrt((x1-x2)^2 + (y1-y2)^2)
        # Broadcasting: (N, 1, 2) - (1, N, 2) = (N, N, 2)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=-1))

        return distances

    def visualize_curve(self, max_points: int = None) -> torch.Tensor:
        """
        Get coordinates for visualizing the Hilbert curve path.

        Args:
            max_points: Maximum number of points to return (default: all)

        Returns:
            Coordinates tensor shape (num_points, 2) for plotting

        Example:
            >>> import matplotlib.pyplot as plt
            >>> mapper = HilbertCurveMapper(max_seq_len=64)
            >>> coords = mapper.visualize_curve()
            >>> plt.plot(coords[:, 0], coords[:, 1], '-o', markersize=2)
            >>> plt.title('Hilbert Curve Order 3')
            >>> plt.show()
        """
        if max_points is None:
            max_points = self.max_positions
        else:
            max_points = min(max_points, self.max_positions)

        indices = torch.arange(max_points, device=self.hilbert_coords.device)
        coords = self.forward(indices)

        return coords
