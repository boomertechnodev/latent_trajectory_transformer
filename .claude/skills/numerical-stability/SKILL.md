# Numerical Stability Expert Skill

Expert implementation of numerical stability techniques, gradient flow optimization, and debugging tools for deep learning systems. This skill provides battle-tested implementations for preventing and fixing numerical issues in neural networks.

## Core Implementations

### 1. Stable Mathematical Operations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings


class StableMath:
    """Collection of numerically stable mathematical operations."""

    @staticmethod
    def log_sum_exp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """
        Compute log(sum(exp(x))) in a numerically stable way.

        Args:
            x: Input tensor
            dim: Dimension to reduce
            keepdim: Keep reduced dimension

        Returns:
            Stable log-sum-exp result
        """
        max_x = x.max(dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - max_x)
        sum_exp_x = exp_x.sum(dim=dim, keepdim=keepdim)
        return max_x.squeeze(dim) + torch.log(sum_exp_x + 1e-30)

    @staticmethod
    def stable_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute softmax with numerical stability and temperature scaling.

        Args:
            x: Logits tensor
            dim: Dimension for softmax
            temperature: Temperature for scaling

        Returns:
            Stable softmax probabilities
        """
        x = x / temperature
        x_max = x.max(dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / (exp_x.sum(dim=dim, keepdim=True) + 1e-30)

    @staticmethod
    def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute log(softmax(x)) stably.
        """
        return x - StableMath.log_sum_exp(x, dim=dim, keepdim=True)

    @staticmethod
    def stable_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                            reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
        """
        Numerically stable cross-entropy with label smoothing.

        Args:
            logits: Model predictions (B, num_classes)
            targets: Target labels (B,) or one-hot (B, num_classes)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor

        Returns:
            Cross-entropy loss
        """
        num_classes = logits.shape[-1]
        log_probs = StableMath.stable_log_softmax(logits, dim=-1)

        # Handle both index and one-hot targets
        if targets.dim() == 1:
            # Convert to one-hot
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = one_hot

        # Apply label smoothing
        if label_smoothing > 0:
            targets = targets * (1 - label_smoothing) + label_smoothing / num_classes

        # Compute loss
        loss = -(targets * log_probs).sum(dim=-1)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    @staticmethod
    def stable_binary_cross_entropy(pred: torch.Tensor, target: torch.Tensor,
                                   reduction: str = 'mean') -> torch.Tensor:
        """
        Stable binary cross-entropy using log-sigmoid.
        """
        max_val = torch.clamp(pred, max=0)
        loss = pred - pred * target + max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred - max_val) + 1e-30)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    @staticmethod
    def stable_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor,
                           reduction: str = 'batchmean') -> torch.Tensor:
        """
        Compute KL(P||Q) with numerical stability.

        Args:
            p_logits: Logits for distribution P
            q_logits: Logits for distribution Q
            reduction: Reduction method

        Returns:
            KL divergence
        """
        p_log_probs = StableMath.stable_log_softmax(p_logits)
        q_log_probs = StableMath.stable_log_softmax(q_logits)

        kl = torch.exp(p_log_probs) * (p_log_probs - q_log_probs)
        kl = kl.sum(dim=-1)

        if reduction == 'batchmean':
            return kl.mean()
        elif reduction == 'sum':
            return kl.sum()
        return kl

    @staticmethod
    def stable_gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Stable Gumbel-Softmax for differentiable sampling.
        """
        # Sample Gumbel noise
        eps = 1e-20
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + eps) + eps)

        # Add noise and apply softmax
        y = StableMath.stable_softmax((logits + gumbel) / tau, dim=-1)

        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1)
            y = (y_hard - y).detach() + y

        return y


### 2. Advanced Initialization Methods

class SmartInitializer:
    """Advanced initialization strategies for deep networks."""

    @staticmethod
    def xavier_uniform_modified(tensor: torch.Tensor, gain: float = 1.0,
                               mode: str = 'fan_avg') -> torch.Tensor:
        """
        Modified Xavier initialization with flexible fan mode.

        Args:
            tensor: Weight tensor to initialize
            gain: Scaling factor
            mode: 'fan_in', 'fan_out', or 'fan_avg'
        """
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Xavier initialization requires at least 2D tensor")

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

        if mode == 'fan_in':
            std = gain * np.sqrt(2.0 / fan_in)
        elif mode == 'fan_out':
            std = gain * np.sqrt(2.0 / fan_out)
        else:  # fan_avg
            std = gain * np.sqrt(4.0 / (fan_in + fan_out))

        bound = np.sqrt(3.0) * std
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
        return tensor

    @staticmethod
    def he_normal_modified(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in',
                          nonlinearity: str = 'relu') -> torch.Tensor:
        """
        Modified He initialization for various activation functions.

        Args:
            tensor: Weight tensor
            a: Negative slope for leaky ReLU
            mode: 'fan_in' or 'fan_out'
            nonlinearity: Activation function name
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        fan = fan_in if mode == 'fan_in' else fan_out

        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain * np.sqrt(2.0 / fan)

        with torch.no_grad():
            tensor.normal_(0, std)
        return tensor

    @staticmethod
    def lsuv_init(model: nn.Module, data_batch: torch.Tensor, target_mean: float = 0.0,
                 target_var: float = 1.0, max_iters: int = 10, tolerance: float = 0.01) -> nn.Module:
        """
        Layer-Sequential Unit-Variance initialization.

        Args:
            model: Model to initialize
            data_batch: Sample batch for activation statistics
            target_mean: Target activation mean
            target_var: Target activation variance
            max_iters: Maximum adjustment iterations
            tolerance: Convergence tolerance
        """
        device = next(model.parameters()).device
        data_batch = data_batch.to(device)

        def adjust_layer(module, input, output):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                with torch.no_grad():
                    # Compute statistics
                    mean = output.mean()
                    var = output.var()

                    # Adjust bias for mean
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data -= (mean - target_mean)

                    # Adjust weights for variance
                    if abs(var - target_var) > tolerance:
                        module.weight.data *= np.sqrt(target_var / (var + 1e-8))

        # Register hooks and run forward pass
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(adjust_layer))

        # Multiple passes for convergence
        for _ in range(max_iters):
            with torch.no_grad():
                _ = model(data_batch)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return model

    @staticmethod
    def fixup_init(model: nn.Module, depth: int) -> nn.Module:
        """
        FixUp initialization for residual networks without normalization.

        Args:
            model: Residual network model
            depth: Number of residual blocks
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=np.sqrt(2 / module.weight.shape[0]))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=np.sqrt(2 / (module.weight.shape[0] * np.prod(module.weight.shape[2:]))))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale residual branches
        for module in model.modules():
            if hasattr(module, 'is_residual_branch') and module.is_residual_branch:
                for submodule in module.modules():
                    if isinstance(submodule, (nn.Linear, nn.Conv2d)):
                        submodule.weight.data *= depth ** (-0.5)

        return model

    @staticmethod
    def spectral_init(tensor: torch.Tensor, target_spectral_norm: float = 1.0) -> torch.Tensor:
        """
        Initialize weights with controlled spectral norm.
        """
        with torch.no_grad():
            # Compute current spectral norm
            U, S, V = torch.svd(tensor.view(tensor.shape[0], -1))

            # Scale to target spectral norm
            tensor.data = (tensor.data / S[0]) * target_spectral_norm

        return tensor


### 3. Gradient Flow Optimization

class GradientFlowOptimizer:
    """Tools for optimizing and monitoring gradient flow."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = []
        self.activation_history = []
        self.hooks = []

    def register_hooks(self):
        """Register hooks to monitor gradients and activations."""
        self.clear_hooks()

        for name, module in self.model.named_modules():
            # Gradient hook
            def grad_hook(module, grad_input, grad_output, name=name):
                if grad_output[0] is not None:
                    grad_norm = grad_output[0].detach().norm().item()
                    self.gradient_history.append({
                        'name': name,
                        'grad_norm': grad_norm,
                        'grad_mean': grad_output[0].detach().mean().item(),
                        'grad_std': grad_output[0].detach().std().item()
                    })

            # Activation hook
            def activation_hook(module, input, output, name=name):
                if isinstance(output, torch.Tensor):
                    self.activation_history.append({
                        'name': name,
                        'act_mean': output.detach().mean().item(),
                        'act_std': output.detach().std().item(),
                        'act_norm': output.detach().norm().item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item()
                    })

            if len(list(module.children())) == 0:  # Leaf modules only
                self.hooks.append(module.register_backward_hook(grad_hook))
                self.hooks.append(module.register_forward_hook(activation_hook))

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradient_history = []
        self.activation_history = []

    def analyze_gradient_flow(self) -> Dict:
        """Analyze collected gradient statistics."""
        if not self.gradient_history:
            return {}

        analysis = {
            'dead_layers': [],
            'exploding_layers': [],
            'healthy_layers': [],
            'gradient_norm_ratio': None
        }

        # Group by layer
        layer_stats = {}
        for item in self.gradient_history:
            name = item['name']
            if name not in layer_stats:
                layer_stats[name] = []
            layer_stats[name].append(item['grad_norm'])

        # Analyze each layer
        for name, norms in layer_stats.items():
            mean_norm = np.mean(norms)

            if mean_norm < 1e-8:
                analysis['dead_layers'].append(name)
            elif mean_norm > 100:
                analysis['exploding_layers'].append(name)
            else:
                analysis['healthy_layers'].append(name)

        # Compute gradient norm ratio (first/last layer)
        all_norms = [item['grad_norm'] for item in self.gradient_history]
        if len(all_norms) > 1:
            analysis['gradient_norm_ratio'] = all_norms[0] / (all_norms[-1] + 1e-8)

        return analysis

    def suggest_fixes(self, analysis: Dict) -> List[str]:
        """Suggest fixes based on gradient flow analysis."""
        suggestions = []

        if analysis.get('dead_layers'):
            suggestions.append("Dead neurons detected. Consider:")
            suggestions.append("- Using LeakyReLU or ELU instead of ReLU")
            suggestions.append("- Reducing learning rate")
            suggestions.append("- Better initialization (He or LSUV)")

        if analysis.get('exploding_layers'):
            suggestions.append("Exploding gradients detected. Consider:")
            suggestions.append("- Gradient clipping (norm or value)")
            suggestions.append("- Batch normalization or layer normalization")
            suggestions.append("- Reducing learning rate")
            suggestions.append("- Spectral normalization")

        ratio = analysis.get('gradient_norm_ratio')
        if ratio and (ratio > 1000 or ratio < 0.001):
            suggestions.append("Poor gradient flow detected. Consider:")
            suggestions.append("- Adding skip connections")
            suggestions.append("- Using gradient normalization")
            suggestions.append("- Pre-activation normalization")

        return suggestions


### 4. Mixed Precision Training

class MixedPrecisionTrainer:
    """Advanced mixed precision training with dynamic loss scaling."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 init_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        )
        self.overflow_count = 0
        self.successful_steps = 0

    def train_step(self, data_batch: torch.Tensor, target_batch: torch.Tensor,
                  loss_fn: callable) -> Tuple[float, bool]:
        """
        Perform one training step with mixed precision.

        Returns:
            loss: Loss value
            overflow: Whether gradient overflow occurred
        """
        self.optimizer.zero_grad()

        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            predictions = self.model(data_batch)
            loss = loss_fn(predictions, target_batch)

        # Backward pass with scaling
        self.scaler.scale(loss).backward()

        # Check for overflow before stepping
        overflow = self._check_gradients()

        if not overflow:
            # Unscale and clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.successful_steps += 1
        else:
            # Skip update on overflow
            self.overflow_count += 1
            self.scaler.update()

        return loss.item(), overflow

    def _check_gradients(self) -> bool:
        """Check for NaN or Inf in gradients."""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False

    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            'current_scale': self.scaler.get_scale(),
            'overflow_count': self.overflow_count,
            'successful_steps': self.successful_steps,
            'overflow_rate': self.overflow_count / max(1, self.overflow_count + self.successful_steps)
        }


### 5. Numerical Debugging Tools

class NumericalDebugger:
    """Comprehensive debugging tools for numerical issues."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.watch_tensors = {}
        self.anomaly_log = []

    def check_model_numerics(self) -> Dict[str, List]:
        """Check all model parameters and buffers for numerical issues."""
        issues = {
            'nan_params': [],
            'inf_params': [],
            'zero_grad_params': [],
            'large_params': [],
            'denormal_params': []
        }

        for name, param in self.model.named_parameters():
            if param is not None:
                # Check for NaN
                if torch.isnan(param).any():
                    issues['nan_params'].append(name)

                # Check for Inf
                if torch.isinf(param).any():
                    issues['inf_params'].append(name)

                # Check for zero gradients
                if param.grad is not None and param.grad.abs().max() < 1e-8:
                    issues['zero_grad_params'].append(name)

                # Check for large values
                if param.abs().max() > 1e6:
                    issues['large_params'].append(name)

                # Check for denormal numbers
                if (param.abs() < 1e-30).any() and (param != 0).any():
                    issues['denormal_params'].append(name)

        return issues

    def monitor_forward_pass(self, input_batch: torch.Tensor) -> Dict:
        """Monitor a forward pass for numerical issues."""
        stats = {
            'layer_stats': [],
            'anomalies': []
        }

        def hook_fn(module, input, output):
            layer_name = str(module.__class__.__name__)

            if isinstance(output, torch.Tensor):
                stat = {
                    'layer': layer_name,
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item(),
                    'output_min': output.min().item(),
                    'output_max': output.max().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                    'sparsity': (output == 0).float().mean().item()
                }
                stats['layer_stats'].append(stat)

                # Check for anomalies
                if stat['has_nan'] or stat['has_inf']:
                    stats['anomalies'].append(f"NaN/Inf in {layer_name}")
                if stat['output_std'] < 1e-6:
                    stats['anomalies'].append(f"Collapsed activations in {layer_name}")
                if stat['sparsity'] > 0.99:
                    stats['anomalies'].append(f"Dead neurons in {layer_name}")

        # Register hooks
        hooks = []
        for module in self.model.modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_batch)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return stats

    def compute_condition_number(self, layer_name: str) -> float:
        """Compute condition number of a linear layer."""
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                weight = module.weight.detach()
                U, S, V = torch.svd(weight)
                condition = S[0] / S[-1]
                return condition.item()
        return -1

    def gradient_magnitude_analysis(self, loss: torch.Tensor) -> Dict:
        """Analyze gradient magnitudes throughout the network."""
        loss.backward(retain_graph=True)

        analysis = {
            'min_grad': float('inf'),
            'max_grad': 0,
            'mean_grad': 0,
            'gradient_norm_per_layer': {}
        }

        total_params = 0
        total_grad_sum = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                analysis['gradient_norm_per_layer'][name] = grad_norm

                analysis['min_grad'] = min(analysis['min_grad'], grad_norm)
                analysis['max_grad'] = max(analysis['max_grad'], grad_norm)

                total_grad_sum += param.grad.abs().sum().item()
                total_params += param.numel()

        analysis['mean_grad'] = total_grad_sum / max(1, total_params)
        analysis['gradient_ratio'] = analysis['max_grad'] / max(1e-8, analysis['min_grad'])

        return analysis


### 6. Adaptive Clipping and Regularization

class AdaptiveGradientClipper:
    """Adaptive gradient clipping based on gradient statistics."""

    def __init__(self, percentile: float = 95.0, history_size: int = 100):
        self.percentile = percentile
        self.history_size = history_size
        self.gradient_history = []

    def clip(self, model: nn.Module) -> float:
        """
        Adaptively clip gradients based on historical statistics.

        Returns:
            clip_value: The value used for clipping
        """
        # Collect current gradient norms
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if not grad_norms:
            return 0.0

        current_norm = np.mean(grad_norms)
        self.gradient_history.append(current_norm)

        # Maintain history size
        if len(self.gradient_history) > self.history_size:
            self.gradient_history.pop(0)

        # Compute adaptive threshold
        if len(self.gradient_history) >= 10:
            threshold = np.percentile(self.gradient_history, self.percentile)
        else:
            threshold = 10.0  # Default threshold

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=threshold)

        return threshold


class NumericalRegularizer(nn.Module):
    """Regularization terms for numerical stability."""

    def __init__(self, spectral_reg: float = 0.01, activation_reg: float = 0.001,
                 gradient_penalty: float = 0.001):
        super().__init__()
        self.spectral_reg = spectral_reg
        self.activation_reg = activation_reg
        self.gradient_penalty = gradient_penalty

    def spectral_regularization(self, model: nn.Module) -> torch.Tensor:
        """Regularize spectral norms of weight matrices."""
        reg_loss = 0.0

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.view(module.weight.shape[0], -1)
                U, S, V = torch.svd(weight)
                reg_loss += self.spectral_reg * S[0]  # Penalize largest singular value

        return reg_loss

    def activation_regularization(self, activations: torch.Tensor) -> torch.Tensor:
        """Regularize activation statistics to prevent saturation."""
        # Encourage unit variance
        var_loss = (activations.var() - 1.0) ** 2

        # Prevent extreme values
        extreme_loss = torch.mean(F.relu(activations.abs() - 10.0))

        return self.activation_reg * (var_loss + extreme_loss)

    def gradient_penalty_loss(self, real_data: torch.Tensor, fake_data: torch.Tensor,
                            discriminator: nn.Module) -> torch.Tensor:
        """WGAN-GP style gradient penalty for stable training."""
        batch_size = real_data.shape[0]

        # Random interpolation
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Forward pass
        output = discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        return self.gradient_penalty * gradient_penalty


### 7. Robustness Testing

class RobustnessTest:
    """Test model robustness to numerical perturbations."""

    def __init__(self, model: nn.Module):
        self.model = model

    def test_input_sensitivity(self, input_batch: torch.Tensor,
                              epsilon_range: List[float] = [1e-6, 1e-4, 1e-2]) -> Dict:
        """Test sensitivity to input perturbations."""
        results = {}

        with torch.no_grad():
            original_output = self.model(input_batch)

            for epsilon in epsilon_range:
                # Add noise
                noise = torch.randn_like(input_batch) * epsilon
                perturbed_input = input_batch + noise

                # Forward pass
                perturbed_output = self.model(perturbed_input)

                # Measure difference
                output_diff = (perturbed_output - original_output).abs().mean().item()
                relative_diff = output_diff / (original_output.abs().mean().item() + 1e-8)

                results[f'epsilon_{epsilon}'] = {
                    'absolute_diff': output_diff,
                    'relative_diff': relative_diff
                }

        return results

    def test_weight_quantization(self, bits: List[int] = [16, 8, 4]) -> Dict:
        """Test robustness to weight quantization."""
        results = {}

        for num_bits in bits:
            # Clone model
            quantized_model = copy.deepcopy(self.model)

            # Quantize weights
            for param in quantized_model.parameters():
                param.data = self._quantize_tensor(param.data, num_bits)

            # Compare outputs
            test_input = torch.randn(1, *self.model.input_shape)
            with torch.no_grad():
                original_output = self.model(test_input)
                quantized_output = quantized_model(test_input)

            diff = (original_output - quantized_output).abs().mean().item()
            results[f'{num_bits}_bit'] = diff

        return results

    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize tensor to specified bit width."""
        min_val = tensor.min()
        max_val = tensor.max()

        # Scale to [0, 2^bits - 1]
        scale = (max_val - min_val) / (2 ** bits - 1)
        quantized = torch.round((tensor - min_val) / scale)

        # Scale back
        return quantized * scale + min_val


### 8. Training Stability Monitor

class TrainingStabilityMonitor:
    """Monitor and ensure training stability."""

    def __init__(self, model: nn.Module, patience: int = 10):
        self.model = model
        self.patience = patience
        self.loss_history = []
        self.gradient_explosion_count = 0
        self.nan_count = 0
        self.early_stop = False

    def update(self, loss: float, gradients_ok: bool = True) -> Dict:
        """Update monitoring statistics."""
        self.loss_history.append(loss)

        # Check for NaN loss
        if np.isnan(loss) or np.isinf(loss):
            self.nan_count += 1
            if self.nan_count >= self.patience:
                self.early_stop = True
                return {'status': 'stop', 'reason': 'NaN loss'}

        # Check for gradient explosion
        if not gradients_ok:
            self.gradient_explosion_count += 1
            if self.gradient_explosion_count >= self.patience:
                self.early_stop = True
                return {'status': 'stop', 'reason': 'Gradient explosion'}

        # Check for loss explosion
        if len(self.loss_history) > 10:
            recent_mean = np.mean(self.loss_history[-10:])
            if loss > 100 * recent_mean:
                return {'status': 'warning', 'reason': 'Potential loss explosion'}

        return {'status': 'ok'}

    def suggest_recovery(self, issue: str) -> List[str]:
        """Suggest recovery strategies for different issues."""
        suggestions = []

        if 'NaN' in issue:
            suggestions.extend([
                "Reduce learning rate by 10x",
                "Check for division by zero",
                "Add epsilon terms to denominators",
                "Enable gradient clipping",
                "Check data for invalid values"
            ])

        if 'explosion' in issue:
            suggestions.extend([
                "Enable gradient clipping",
                "Reduce learning rate",
                "Use gradient accumulation",
                "Add batch/layer normalization",
                "Check initialization"
            ])

        return suggestions

    def get_summary(self) -> Dict:
        """Get training stability summary."""
        return {
            'total_steps': len(self.loss_history),
            'nan_count': self.nan_count,
            'gradient_explosions': self.gradient_explosion_count,
            'should_stop': self.early_stop,
            'loss_trend': 'stable' if len(self.loss_history) < 2 else
                         'decreasing' if self.loss_history[-1] < self.loss_history[0] else 'increasing'
        }


# Usage Example
if __name__ == "__main__":
    # Example model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Initialize with LSUV
    sample_batch = torch.randn(32, 100)
    SmartInitializer.lsuv_init(model, sample_batch)

    # Setup gradient flow monitoring
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    grad_monitor = GradientFlowOptimizer(model)
    grad_monitor.register_hooks()

    # Training with stability monitoring
    stability_monitor = TrainingStabilityMonitor(model)

    for epoch in range(10):
        # Forward pass
        output = model(sample_batch)
        target = torch.randint(0, 10, (32,))
        loss = F.cross_entropy(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check stability
        status = stability_monitor.update(loss.item())
        if status['status'] == 'stop':
            print(f"Training stopped: {status['reason']}")
            break

        # Optimize
        optimizer.step()

    # Analyze gradient flow
    analysis = grad_monitor.analyze_gradient_flow()
    print(f"Gradient flow analysis: {analysis}")

    # Get suggestions
    suggestions = grad_monitor.suggest_fixes(analysis)
    for suggestion in suggestions:
        print(suggestion)