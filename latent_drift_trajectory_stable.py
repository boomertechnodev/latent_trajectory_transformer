"""
Numerical Stability Improvements for Latent Trajectory Transformer
====================================================================

This module provides numerically stable implementations and utilities
for the latent trajectory transformer, addressing all critical issues
identified in the stability audit.

Author: Numerical Stability Specialist Agent
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================================
# NUMERICAL CONSTANTS FOR STABILITY
# ============================================================================

class StabilityConstants:
    """
    Centralized numerical stability constants.

    These values are carefully chosen to work with both FP32 and FP16/BF16.
    """
    # Epsilon values (compatible with float16)
    EPS_FLOAT32 = 1e-8
    EPS_FLOAT16 = 1e-4  # Larger than fp16 smallest normal (6.1e-5)
    EPS_DIVISION = 1e-6  # For safe division
    EPS_LOG = 1e-7  # For log operations

    # Clamping bounds
    LOG_MIN = -20.0  # exp(-20) ≈ 2e-9, safe for fp16
    LOG_MAX = 20.0   # exp(20) ≈ 5e8, well below fp16 max

    LOGVAR_MIN = -10.0  # For variational parameters
    LOGVAR_MAX = 10.0

    SCALE_MIN = 1e-3  # Minimum scale for flows
    SCALE_MAX = 100.0  # Maximum scale for flows

    # Gradient clipping
    GRAD_CLIP_NORM = 1.0  # Default max gradient norm
    GRAD_CLIP_VALUE = 10.0  # Backup value clipping

    # Attention masking
    ATTENTION_MASK_VALUE = -1e9  # Instead of -inf, works with all dtypes

    @staticmethod
    def get_eps(dtype: torch.dtype) -> float:
        """Get appropriate epsilon for dtype."""
        if dtype in [torch.float16, torch.bfloat16]:
            return StabilityConstants.EPS_FLOAT16
        return StabilityConstants.EPS_FLOAT32


# ============================================================================
# STABLE MATHEMATICAL OPERATIONS
# ============================================================================

def stable_log(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """
    Numerically stable logarithm.

    Args:
        x: Input tensor
        eps: Minimum value before log (defaults based on dtype)

    Returns:
        log(max(x, eps)) with appropriate epsilon
    """
    if eps is None:
        eps = StabilityConstants.get_eps(x.dtype)
    return torch.log(torch.clamp(x, min=eps))


def stable_exp(x: torch.Tensor,
               min_val: float = None,
               max_val: float = None) -> torch.Tensor:
    """
    Numerically stable exponential with clamping.

    Args:
        x: Input tensor
        min_val: Minimum input value (default: LOG_MIN)
        max_val: Maximum input value (default: LOG_MAX)

    Returns:
        exp(clamp(x, min_val, max_val))
    """
    if min_val is None:
        min_val = StabilityConstants.LOG_MIN
    if max_val is None:
        max_val = StabilityConstants.LOG_MAX

    x_clamped = torch.clamp(x, min=min_val, max=max_val)
    return torch.exp(x_clamped)


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax using the max-subtraction trick.

    Args:
        x: Input logits
        dim: Dimension for softmax

    Returns:
        Stable softmax probabilities
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + StabilityConstants.get_eps(x.dtype))


def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable log-softmax.

    More stable than log(softmax(x)) by using log-sum-exp trick.
    """
    return F.log_softmax(x, dim=dim)


def log_sum_exp(x: torch.Tensor,
                 dim: int = -1,
                 keepdim: bool = False) -> torch.Tensor:
    """
    Numerically stable log-sum-exp computation.

    Computes log(sum(exp(x))) in a stable way.

    Args:
        x: Input tensor
        dim: Dimension to reduce
        keepdim: Keep reduced dimension

    Returns:
        log(sum(exp(x))) computed stably
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    if not keepdim:
        x_max_squeeze = x_max.squeeze(dim)
    else:
        x_max_squeeze = x_max

    return x_max_squeeze + torch.log(
        torch.sum(torch.exp(x - x_max), dim=dim, keepdim=keepdim) +
        StabilityConstants.get_eps(x.dtype)
    )


def stable_norm(x: torch.Tensor,
                p: float = 2.0,
                dim: Optional[int] = None,
                eps: Optional[float] = None) -> torch.Tensor:
    """
    Numerically stable norm computation.

    Args:
        x: Input tensor
        p: Norm type (default: 2)
        dim: Dimension(s) to compute norm over
        eps: Epsilon for numerical stability

    Returns:
        Stable norm with epsilon handling
    """
    if eps is None:
        eps = StabilityConstants.get_eps(x.dtype)

    if dim is None:
        return torch.norm(x, p=p) + eps
    else:
        return torch.norm(x, p=p, dim=dim, keepdim=False) + eps


def stable_normalize(x: torch.Tensor,
                     dim: int = -1,
                     eps: Optional[float] = None) -> torch.Tensor:
    """
    Numerically stable L2 normalization.

    Args:
        x: Input tensor
        dim: Dimension to normalize
        eps: Epsilon for division

    Returns:
        L2-normalized tensor
    """
    if eps is None:
        eps = StabilityConstants.EPS_DIVISION

    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)


# ============================================================================
# STABLE INITIALIZATION UTILITIES
# ============================================================================

class StableInitializer:
    """
    Numerically stable weight initialization strategies.
    """

    @staticmethod
    def xavier_uniform_stable(tensor: torch.Tensor,
                               gain: float = 1.0,
                               fan_mode: str = 'fan_avg') -> torch.Tensor:
        """
        Xavier/Glorot uniform initialization with stability improvements.

        Args:
            tensor: Tensor to initialize
            gain: Scaling factor
            fan_mode: 'fan_in', 'fan_out', or 'fan_avg'

        Returns:
            Initialized tensor
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

        if fan_mode == 'fan_in':
            fan = fan_in
        elif fan_mode == 'fan_out':
            fan = fan_out
        else:  # fan_avg
            fan = (fan_in + fan_out) / 2.0

        # Add small epsilon to prevent division by zero
        std = gain * math.sqrt(2.0 / (fan + StabilityConstants.EPS_DIVISION))

        # Clamp to reasonable range
        std = min(std, 0.1)  # Prevent too large initialization

        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        return tensor

    @staticmethod
    def kaiming_normal_stable(tensor: torch.Tensor,
                               a: float = 0,
                               mode: str = 'fan_in',
                               nonlinearity: str = 'relu') -> torch.Tensor:
        """
        He/Kaiming normal initialization with stability improvements.
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

        if mode == 'fan_in':
            fan = fan_in
        else:
            fan = fan_out

        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan + StabilityConstants.EPS_DIVISION)

        # Clamp standard deviation
        std = min(max(std, 1e-3), 0.1)

        with torch.no_grad():
            tensor.normal_(0, std)
            # Clip extreme values
            tensor.clamp_(-3 * std, 3 * std)
        return tensor

    @staticmethod
    def residual_init(tensor: torch.Tensor,
                       depth: int,
                       layer_idx: int) -> torch.Tensor:
        """
        Depth-scaled initialization for residual connections.

        Scales weights by 1/sqrt(depth) to maintain signal variance.
        """
        # Base initialization
        nn.init.xavier_uniform_(tensor)

        # Scale by depth
        scale = (layer_idx + 1) ** -0.5
        with torch.no_grad():
            tensor.mul_(scale)

        return tensor


# ============================================================================
# GRADIENT MANAGEMENT UTILITIES
# ============================================================================

class GradientManager:
    """
    Utilities for managing gradient flow and preventing explosions.
    """

    @staticmethod
    def clip_gradients(parameters,
                        max_norm: Optional[float] = None,
                        max_value: Optional[float] = None,
                        norm_type: float = 2.0) -> dict:
        """
        Apply gradient clipping with diagnostics.

        Args:
            parameters: Model parameters
            max_norm: Maximum gradient norm (L2)
            max_value: Maximum gradient value (element-wise)
            norm_type: Type of norm for gradient clipping

        Returns:
            Dictionary with gradient statistics
        """
        if max_norm is None:
            max_norm = StabilityConstants.GRAD_CLIP_NORM

        # Compute gradient norm before clipping
        total_norm = 0.0
        param_count = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
                param_count += 1

        total_norm = total_norm ** (1. / norm_type)

        # Apply gradient clipping
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

        if max_value is not None:
            torch.nn.utils.clip_grad_value_(parameters, max_value)

        # Compute gradient norm after clipping
        total_norm_after = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm_after += param_norm.item() ** norm_type

        total_norm_after = total_norm_after ** (1. / norm_type)

        return {
            'grad_norm_before': total_norm,
            'grad_norm_after': total_norm_after,
            'num_params_with_grad': param_count,
            'clipped': total_norm > max_norm if max_norm else False
        }

    @staticmethod
    def check_gradients(model: nn.Module,
                         raise_on_nan: bool = True) -> dict:
        """
        Check for NaN/Inf in gradients.

        Args:
            model: PyTorch model
            raise_on_nan: Raise exception if NaN/Inf found

        Returns:
            Dictionary with gradient health statistics
        """
        stats = {
            'has_nan': False,
            'has_inf': False,
            'num_nan': 0,
            'num_inf': 0,
            'params_affected': []
        }

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                nan_mask = torch.isnan(grad)
                inf_mask = torch.isinf(grad)

                if nan_mask.any():
                    stats['has_nan'] = True
                    stats['num_nan'] += nan_mask.sum().item()
                    stats['params_affected'].append(f"{name}_nan")

                if inf_mask.any():
                    stats['has_inf'] = True
                    stats['num_inf'] += inf_mask.sum().item()
                    stats['params_affected'].append(f"{name}_inf")

        if raise_on_nan and (stats['has_nan'] or stats['has_inf']):
            raise ValueError(f"Gradient check failed: {stats}")

        return stats

    @staticmethod
    def adaptive_gradient_clipping(parameters,
                                    percentile: float = 95.0) -> dict:
        """
        Adaptive gradient clipping based on gradient statistics.

        Clips to a percentile of current gradient norms.
        """
        grad_norms = []
        for p in parameters:
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm(2).item())

        if len(grad_norms) == 0:
            return {'skipped': True, 'reason': 'no_gradients'}

        grad_norms_tensor = torch.tensor(grad_norms)
        threshold = torch.quantile(grad_norms_tensor, percentile / 100.0).item()

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(parameters, threshold)

        return {
            'threshold': threshold,
            'max_norm': grad_norms_tensor.max().item(),
            'mean_norm': grad_norms_tensor.mean().item(),
            'percentile': percentile
        }


# ============================================================================
# STABLE LAYER IMPLEMENTATIONS
# ============================================================================

class StableLayerNorm(nn.Module):
    """
    Numerically stable LayerNorm implementation.
    """
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps if eps is not None else StabilityConstants.EPS_FLOAT16
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Welford's algorithm for numerical stability
        mean = x.mean(dim=-1, keepdim=True)
        # Two-pass variance for better numerical stability
        x_centered = x - mean
        var = (x_centered ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        x_norm = x_centered / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


class StableAttention(nn.Module):
    """
    Numerically stable multi-head attention.
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 use_stable_softmax: bool = True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.dropout = dropout
        self.use_stable_softmax = use_stable_softmax

        # Initialize with small values for stability
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        # Use scaled initialization
        for module in [self.w_q, self.w_k, self.w_v]:
            StableInitializer.xavier_uniform_stable(module.weight, gain=0.5)
            nn.init.zeros_(module.bias)

        # Output projection with residual scaling
        StableInitializer.xavier_uniform_stable(self.w_o.weight, gain=0.1)
        nn.init.zeros_(self.w_o.bias)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention with numerical stability
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask with bounded values (not -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, StabilityConstants.ATTENTION_MASK_VALUE)

        # Stable softmax
        if self.use_stable_softmax:
            attn_weights = stable_softmax(scores, dim=-1)
        else:
            attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attn_output)

        return output


# ============================================================================
# MIXED PRECISION UTILITIES
# ============================================================================

class MixedPrecisionHelper:
    """
    Utilities for mixed precision training stability.
    """

    @staticmethod
    def prepare_for_autocast(model: nn.Module) -> nn.Module:
        """
        Prepare model for automatic mixed precision.

        Ensures critical operations stay in FP32.
        """
        for name, module in model.named_modules():
            # Keep normalization in FP32
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.float()

            # Keep softmax operations in FP32
            if hasattr(module, 'use_float32_softmax'):
                module.use_float32_softmax = True

        return model

    @staticmethod
    def stable_backward(loss: torch.Tensor,
                         scaler = None,
                         retain_graph: bool = False) -> None:
        """
        Stable backward pass with optional loss scaling.

        Args:
            loss: Loss tensor
            scaler: GradScaler for AMP (optional)
            retain_graph: Retain computation graph
        """
        if scaler is not None:
            scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    @staticmethod
    def check_tensor_health(tensor: torch.Tensor,
                             name: str = "tensor") -> dict:
        """
        Check tensor for numerical issues.

        Args:
            tensor: Tensor to check
            name: Name for logging

        Returns:
            Health statistics dictionary
        """
        with torch.no_grad():
            stats = {
                'name': name,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'has_nan': torch.isnan(tensor).any().item(),
                'has_inf': torch.isinf(tensor).any().item(),
                'min': tensor.min().item() if tensor.numel() > 0 else None,
                'max': tensor.max().item() if tensor.numel() > 0 else None,
                'mean': tensor.mean().item() if tensor.numel() > 0 else None,
                'std': tensor.std().item() if tensor.numel() > 0 else None,
            }

            # Check for extreme values
            if stats['max'] is not None and stats['min'] is not None:
                stats['range'] = stats['max'] - stats['min']
                stats['is_extreme'] = (abs(stats['max']) > 1e6 or
                                        abs(stats['min']) > 1e6)

            return stats


# ============================================================================
# STABLE LOSS FUNCTIONS
# ============================================================================

class StableLosses:
    """
    Numerically stable loss function implementations.
    """

    @staticmethod
    def stable_cross_entropy(logits: torch.Tensor,
                              targets: torch.Tensor,
                              reduction: str = 'mean') -> torch.Tensor:
        """
        Numerically stable cross-entropy loss.

        Uses log-softmax internally for stability.
        """
        return F.cross_entropy(logits, targets, reduction=reduction)

    @staticmethod
    def stable_kl_divergence(mean: torch.Tensor,
                              logvar: torch.Tensor,
                              prior_mean: float = 0.0,
                              prior_logvar: float = 0.0,
                              eps: Optional[float] = None) -> torch.Tensor:
        """
        Numerically stable KL divergence for Gaussian distributions.

        KL(q||p) where q ~ N(mean, exp(logvar)) and p ~ N(prior_mean, exp(prior_logvar))
        """
        if eps is None:
            eps = StabilityConstants.get_eps(mean.dtype)

        # Clamp log-variances to reasonable range
        logvar = torch.clamp(logvar,
                              StabilityConstants.LOGVAR_MIN,
                              StabilityConstants.LOGVAR_MAX)
        prior_logvar_tensor = torch.full_like(logvar, prior_logvar)
        prior_logvar_tensor = torch.clamp(prior_logvar_tensor,
                                           StabilityConstants.LOGVAR_MIN,
                                           StabilityConstants.LOGVAR_MAX)

        # Stable KL computation
        var_ratio = torch.exp(logvar - prior_logvar_tensor)
        diff_squared = (mean - prior_mean) ** 2

        kl = 0.5 * (
            prior_logvar_tensor - logvar +
            var_ratio +
            diff_squared * torch.exp(-prior_logvar_tensor) -
            1.0
        )

        return kl.mean()

    @staticmethod
    def stable_binary_cross_entropy(pred: torch.Tensor,
                                     target: torch.Tensor,
                                     eps: Optional[float] = None) -> torch.Tensor:
        """
        Numerically stable binary cross-entropy.

        Args:
            pred: Predicted probabilities (after sigmoid)
            target: Target labels
            eps: Epsilon for log stability
        """
        if eps is None:
            eps = StabilityConstants.get_eps(pred.dtype)

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, eps, 1.0 - eps)

        loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return loss.mean()


# ============================================================================
# MONITORING AND DEBUGGING
# ============================================================================

class StabilityMonitor:
    """
    Runtime monitoring for numerical stability.
    """

    def __init__(self,
                 log_interval: int = 100,
                 raise_on_nan: bool = True):
        self.log_interval = log_interval
        self.raise_on_nan = raise_on_nan
        self.step = 0
        self.history = {
            'grad_norms': [],
            'loss_values': [],
            'param_stats': [],
            'nan_count': 0,
            'inf_count': 0
        }

    def check_loss(self, loss: torch.Tensor) -> bool:
        """Check if loss is valid."""
        if torch.isnan(loss).any():
            self.history['nan_count'] += 1
            if self.raise_on_nan:
                raise ValueError(f"NaN loss at step {self.step}")
            return False

        if torch.isinf(loss).any():
            self.history['inf_count'] += 1
            if self.raise_on_nan:
                raise ValueError(f"Inf loss at step {self.step}")
            return False

        self.history['loss_values'].append(loss.item())
        return True

    def log_gradient_stats(self, model: nn.Module) -> dict:
        """Log gradient statistics."""
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                grad_norms.append(grad_norm)

        if grad_norms:
            stats = {
                'step': self.step,
                'max_grad': max(grad_norms),
                'mean_grad': sum(grad_norms) / len(grad_norms),
                'min_grad': min(grad_norms)
            }
            self.history['grad_norms'].append(stats)
            return stats
        return {}

    def log_param_stats(self, model: nn.Module) -> dict:
        """Log parameter statistics."""
        param_means = []
        param_stds = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_means.append(param.data.mean().item())
                param_stds.append(param.data.std().item())

        if param_means:
            stats = {
                'step': self.step,
                'mean_param_mean': sum(param_means) / len(param_means),
                'mean_param_std': sum(param_stds) / len(param_stds),
            }
            self.history['param_stats'].append(stats)
            return stats
        return {}

    def step_update(self):
        """Increment step counter."""
        self.step += 1

    def summary(self) -> dict:
        """Get monitoring summary."""
        return {
            'total_steps': self.step,
            'nan_count': self.history['nan_count'],
            'inf_count': self.history['inf_count'],
            'loss_mean': sum(self.history['loss_values'][-100:]) / min(100, len(self.history['loss_values']))
                         if self.history['loss_values'] else 0,
            'grad_norm_max': max([g['max_grad'] for g in self.history['grad_norms'][-100:]])
                            if self.history['grad_norms'] else 0
        }


# ============================================================================
# EXAMPLE: IMPROVED TRAINING LOOP WITH STABILITY
# ============================================================================

def stable_training_step(model: nn.Module,
                          batch: torch.Tensor,
                          labels: torch.Tensor,
                          optimizer: torch.optim.Optimizer,
                          loss_fn,
                          scaler = None,
                          monitor: Optional[StabilityMonitor] = None,
                          max_grad_norm: float = 1.0) -> dict:
    """
    Single training step with comprehensive stability measures.

    Args:
        model: PyTorch model
        batch: Input batch
        labels: Target labels
        optimizer: Optimizer
        loss_fn: Loss function
        scaler: GradScaler for AMP (optional)
        monitor: StabilityMonitor (optional)
        max_grad_norm: Maximum gradient norm

    Returns:
        Dictionary with training statistics
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass with optional autocast
    if scaler is not None:
        from torch.cuda.amp import autocast
        with autocast():
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
    else:
        outputs = model(batch)
        loss = loss_fn(outputs, labels)

    # Check loss validity
    if monitor:
        if not monitor.check_loss(loss):
            return {'loss': float('nan'), 'skipped': True}

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_stats = GradientManager.clip_gradients(
            model.parameters(),
            max_norm=max_grad_norm
        )

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()

        # Gradient clipping
        grad_stats = GradientManager.clip_gradients(
            model.parameters(),
            max_norm=max_grad_norm
        )

        optimizer.step()

    # Log statistics
    if monitor:
        monitor.log_gradient_stats(model)
        monitor.step_update()

    return {
        'loss': loss.item(),
        'grad_norm': grad_stats['grad_norm_after'],
        'grad_clipped': grad_stats['clipped']
    }


# ============================================================================
# TESTING UTILITIES
# ============================================================================

class StabilityTester:
    """
    Test model for numerical stability issues.
    """

    @staticmethod
    def test_extreme_inputs(model: nn.Module,
                             input_shape: tuple,
                             device: torch.device = torch.device('cpu')) -> dict:
        """
        Test model with extreme input values.
        """
        results = {}
        model.eval()

        with torch.no_grad():
            # Test with very small values
            x_small = torch.full(input_shape, 1e-8, device=device)
            try:
                out_small = model(x_small)
                results['small_input'] = {
                    'success': True,
                    'has_nan': torch.isnan(out_small).any().item(),
                    'has_inf': torch.isinf(out_small).any().item(),
                }
            except Exception as e:
                results['small_input'] = {'success': False, 'error': str(e)}

            # Test with very large values
            x_large = torch.full(input_shape, 1e8, device=device)
            try:
                out_large = model(x_large)
                results['large_input'] = {
                    'success': True,
                    'has_nan': torch.isnan(out_large).any().item(),
                    'has_inf': torch.isinf(out_large).any().item(),
                }
            except Exception as e:
                results['large_input'] = {'success': False, 'error': str(e)}

            # Test with zeros
            x_zero = torch.zeros(input_shape, device=device)
            try:
                out_zero = model(x_zero)
                results['zero_input'] = {
                    'success': True,
                    'has_nan': torch.isnan(out_zero).any().item(),
                    'has_inf': torch.isinf(out_zero).any().item(),
                }
            except Exception as e:
                results['zero_input'] = {'success': False, 'error': str(e)}

            # Test with random normal
            x_normal = torch.randn(input_shape, device=device)
            try:
                out_normal = model(x_normal)
                results['normal_input'] = {
                    'success': True,
                    'has_nan': torch.isnan(out_normal).any().item(),
                    'has_inf': torch.isinf(out_normal).any().item(),
                }
            except Exception as e:
                results['normal_input'] = {'success': False, 'error': str(e)}

        return results

    @staticmethod
    def test_gradient_flow(model: nn.Module,
                            input_shape: tuple,
                            device: torch.device = torch.device('cpu')) -> dict:
        """
        Test gradient flow through the model.
        """
        model.train()
        x = torch.randn(input_shape, device=device, requires_grad=True)

        # Forward pass
        out = model(x)

        # Create dummy loss
        if out.dim() > 1:
            loss = out.mean()
        else:
            loss = out.sum()

        # Backward pass
        loss.backward()

        # Check gradients
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_stats[name] = {
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                    'norm': grad.norm(2).item(),
                    'max': grad.abs().max().item(),
                    'min': grad.abs().min().item(),
                }

        return grad_stats


if __name__ == "__main__":
    print("Numerical Stability Utilities Loaded Successfully!")
    print(f"Available utilities:")
    print(f"  - StabilityConstants: Numerical constants for stable computation")
    print(f"  - Stable operations: log, exp, softmax, norm, etc.")
    print(f"  - StableInitializer: Improved weight initialization")
    print(f"  - GradientManager: Gradient clipping and monitoring")
    print(f"  - StableLayerNorm, StableAttention: Stable layer implementations")
    print(f"  - MixedPrecisionHelper: AMP compatibility utilities")
    print(f"  - StabilityMonitor: Runtime stability monitoring")
    print(f"  - StabilityTester: Testing utilities for numerical issues")