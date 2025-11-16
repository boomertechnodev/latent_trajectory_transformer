#!/usr/bin/env python3
"""
Numerical Stability Testing Suite for Latent Trajectory Transformer
====================================================================

Comprehensive tests to identify and debug numerical stability issues.

Usage:
    python test_stability.py [--model ode|raccoon] [--device cpu|cuda|mps]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import argparse
import sys
import warnings

# Import the models
from latent_drift_trajectory import (
    DeterministicLatentODE,
    RaccoonLogClassifier,
    vocab_size,
    log_vocab_size,
    NUM_LOG_CLASSES,
    SyntheticTargetDataset,
    LogDataset,
)

# Import stability utilities
from latent_drift_trajectory_stable import (
    StabilityConstants,
    GradientManager,
    StabilityMonitor,
    StabilityTester,
    MixedPrecisionHelper,
    stable_exp,
    stable_log,
    stable_softmax,
)

from torch.utils.data import DataLoader


class ComprehensiveStabilityTester:
    """
    Comprehensive testing suite for numerical stability.
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.results = {}

    def test_initialization_stability(self, model: nn.Module) -> Dict[str, Any]:
        """Test if model initialization is numerically stable."""
        print("\n" + "="*60)
        print("TESTING INITIALIZATION STABILITY")
        print("="*60)

        results = {
            'total_params': 0,
            'zero_params': 0,
            'large_params': 0,
            'small_params': 0,
            'nan_params': 0,
            'inf_params': 0,
            'param_stats': {}
        }

        for name, param in model.named_parameters():
            data = param.data
            results['total_params'] += param.numel()

            # Check for problematic values
            zeros = (data == 0).sum().item()
            nans = torch.isnan(data).sum().item()
            infs = torch.isinf(data).sum().item()
            large = (data.abs() > 10).sum().item()
            small = (data.abs() < 1e-6).sum().item()

            results['zero_params'] += zeros
            results['nan_params'] += nans
            results['inf_params'] += infs
            results['large_params'] += large
            results['small_params'] += small

            # Compute statistics
            if param.numel() > 0:
                results['param_stats'][name] = {
                    'mean': data.mean().item(),
                    'std': data.std().item(),
                    'min': data.min().item(),
                    'max': data.max().item(),
                    'has_nan': nans > 0,
                    'has_inf': infs > 0,
                    'percent_zero': 100 * zeros / param.numel()
                }

                # Print warnings
                if nans > 0:
                    print(f"⚠️ NaN found in {name}: {nans} values")
                if infs > 0:
                    print(f"⚠️ Inf found in {name}: {infs} values")
                if large > param.numel() * 0.1:  # More than 10% large values
                    print(f"⚠️ Large values in {name}: {large}/{param.numel()}")

        # Summary
        print(f"\n📊 Initialization Summary:")
        print(f"  Total parameters: {results['total_params']:,}")
        print(f"  Zero parameters: {results['zero_params']:,} ({100*results['zero_params']/results['total_params']:.2f}%)")
        print(f"  Large parameters (>10): {results['large_params']:,}")
        print(f"  Small parameters (<1e-6): {results['small_params']:,}")
        print(f"  NaN parameters: {results['nan_params']}")
        print(f"  Inf parameters: {results['inf_params']}")

        return results

    def test_forward_pass_stability(self, model: nn.Module, input_batch: torch.Tensor) -> Dict[str, Any]:
        """Test forward pass with various input conditions."""
        print("\n" + "="*60)
        print("TESTING FORWARD PASS STABILITY")
        print("="*60)

        model.eval()
        results = {}

        # Test cases
        test_cases = [
            ("normal", torch.randn_like(input_batch)),
            ("zeros", torch.zeros_like(input_batch)),
            ("ones", torch.ones_like(input_batch)),
            ("small", torch.full_like(input_batch, 1e-6)),
            ("large", torch.full_like(input_batch, 100)),
            ("mixed", torch.randn_like(input_batch) * 100),
        ]

        for name, test_input in test_cases:
            print(f"\nTesting with {name} input...")

            try:
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        # For ODE model that expects loss_weights
                        if isinstance(model, DeterministicLatentODE):
                            output, stats = model(test_input, loss_weights=(1.0, 0.1, 1.0))
                            results[name] = {
                                'success': True,
                                'loss': output.item(),
                                'has_nan': torch.isnan(output).any().item(),
                                'has_inf': torch.isinf(output).any().item(),
                                'stats': {k: v.item() for k, v in stats.items()}
                            }
                        # For Raccoon model that expects tokens and labels
                        elif isinstance(model, RaccoonLogClassifier):
                            labels = torch.randint(0, NUM_LOG_CLASSES, (test_input.shape[0],), device=self.device)
                            output, stats = model(test_input, labels)
                            results[name] = {
                                'success': True,
                                'loss': output.item() if output.numel() > 0 else 0,
                                'has_nan': torch.isnan(output).any().item() if output.numel() > 0 else False,
                                'has_inf': torch.isinf(output).any().item() if output.numel() > 0 else False,
                                'stats': {k: v.item() if torch.is_tensor(v) else v for k, v in stats.items()}
                            }
                        else:
                            output = model(test_input)
                            results[name] = {
                                'success': True,
                                'has_nan': torch.isnan(output).any().item(),
                                'has_inf': torch.isinf(output).any().item(),
                            }

                    if results[name].get('has_nan'):
                        print(f"  ❌ NaN detected in output!")
                    elif results[name].get('has_inf'):
                        print(f"  ❌ Inf detected in output!")
                    else:
                        print(f"  ✅ Output is numerically stable")

            except Exception as e:
                print(f"  ❌ Forward pass failed: {str(e)}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def test_gradient_stability(self, model: nn.Module, input_batch: torch.Tensor) -> Dict[str, Any]:
        """Test gradient flow and stability."""
        print("\n" + "="*60)
        print("TESTING GRADIENT STABILITY")
        print("="*60)

        model.train()
        results = {}

        # Enable anomaly detection
        with torch.autograd.set_detect_anomaly(True):
            try:
                # Forward pass
                if isinstance(model, DeterministicLatentODE):
                    loss, _ = model(input_batch, loss_weights=(1.0, 0.1, 1.0))
                elif isinstance(model, RaccoonLogClassifier):
                    labels = torch.randint(0, NUM_LOG_CLASSES, (input_batch.shape[0],), device=self.device)
                    loss, _ = model(input_batch, labels)
                else:
                    output = model(input_batch)
                    loss = output.mean() if output.dim() > 0 else output

                # Backward pass
                loss.backward()

                # Check gradients
                grad_norms = []
                params_with_grad = 0
                params_with_nan_grad = []
                params_with_inf_grad = []
                params_with_large_grad = []

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad
                        grad_norm = grad.norm(2).item()
                        grad_norms.append(grad_norm)
                        params_with_grad += 1

                        if torch.isnan(grad).any():
                            params_with_nan_grad.append(name)
                        if torch.isinf(grad).any():
                            params_with_inf_grad.append(name)
                        if grad_norm > 100:
                            params_with_large_grad.append(name)

                # Statistics
                if grad_norms:
                    results['grad_stats'] = {
                        'mean': np.mean(grad_norms),
                        'std': np.std(grad_norms),
                        'min': np.min(grad_norms),
                        'max': np.max(grad_norms),
                        'num_params_with_grad': params_with_grad,
                        'params_with_nan': params_with_nan_grad,
                        'params_with_inf': params_with_inf_grad,
                        'params_with_large_grad': params_with_large_grad
                    }

                    print(f"\n📊 Gradient Statistics:")
                    print(f"  Parameters with gradients: {params_with_grad}")
                    print(f"  Mean gradient norm: {results['grad_stats']['mean']:.6f}")
                    print(f"  Max gradient norm: {results['grad_stats']['max']:.6f}")
                    print(f"  Min gradient norm: {results['grad_stats']['min']:.6f}")

                    if params_with_nan_grad:
                        print(f"  ❌ NaN gradients in: {params_with_nan_grad[:5]}")
                    if params_with_inf_grad:
                        print(f"  ❌ Inf gradients in: {params_with_inf_grad[:5]}")
                    if params_with_large_grad:
                        print(f"  ⚠️ Large gradients (>100) in: {params_with_large_grad[:5]}")
                    if not (params_with_nan_grad or params_with_inf_grad):
                        print(f"  ✅ All gradients are finite")

                results['success'] = True

            except Exception as e:
                print(f"  ❌ Gradient computation failed: {str(e)}")
                results['success'] = False
                results['error'] = str(e)

        # Clean gradients
        model.zero_grad()

        return results

    def test_mixed_precision_compatibility(self, model: nn.Module, input_batch: torch.Tensor) -> Dict[str, Any]:
        """Test model with different precision settings."""
        print("\n" + "="*60)
        print("TESTING MIXED PRECISION COMPATIBILITY")
        print("="*60)

        results = {}

        # Test float32 (baseline)
        print("\nTesting with float32...")
        model_fp32 = model.float()
        input_fp32 = input_batch.float()

        try:
            with torch.no_grad():
                if isinstance(model, DeterministicLatentODE):
                    output_fp32, _ = model_fp32(input_fp32)
                elif isinstance(model, RaccoonLogClassifier):
                    labels = torch.randint(0, NUM_LOG_CLASSES, (input_fp32.shape[0],), device=self.device)
                    output_fp32, _ = model_fp32(input_fp32, labels)
                else:
                    output_fp32 = model_fp32(input_fp32)

            results['fp32'] = {
                'success': True,
                'has_nan': torch.isnan(output_fp32).any().item(),
                'has_inf': torch.isinf(output_fp32).any().item(),
            }
            print(f"  ✅ Float32 test passed")
        except Exception as e:
            results['fp32'] = {'success': False, 'error': str(e)}
            print(f"  ❌ Float32 test failed: {e}")

        # Test float16 (if CUDA available)
        if torch.cuda.is_available() and self.device.type == 'cuda':
            print("\nTesting with float16...")
            try:
                model_fp16 = model.half().cuda()
                input_fp16 = input_batch.half().cuda()

                with torch.no_grad():
                    if isinstance(model, DeterministicLatentODE):
                        output_fp16, _ = model_fp16(input_fp16)
                    elif isinstance(model, RaccoonLogClassifier):
                        labels = torch.randint(0, NUM_LOG_CLASSES, (input_fp16.shape[0],), device='cuda')
                        output_fp16, _ = model_fp16(input_fp16, labels)
                    else:
                        output_fp16 = model_fp16(input_fp16)

                results['fp16'] = {
                    'success': True,
                    'has_nan': torch.isnan(output_fp16).any().item(),
                    'has_inf': torch.isinf(output_fp16).any().item(),
                }
                print(f"  ✅ Float16 test passed")
            except Exception as e:
                results['fp16'] = {'success': False, 'error': str(e)}
                print(f"  ❌ Float16 test failed: {e}")
                print(f"     This likely indicates numerical instability issues!")

        return results

    def test_training_step_stability(self, model: nn.Module, dataloader: DataLoader, num_steps: int = 10) -> Dict[str, Any]:
        """Test stability over multiple training steps."""
        print("\n" + "="*60)
        print(f"TESTING TRAINING STABILITY ({num_steps} steps)")
        print("="*60)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        monitor = StabilityMonitor(raise_on_nan=False)

        results = {
            'steps': [],
            'nan_count': 0,
            'inf_count': 0,
            'gradient_explosions': 0
        }

        data_iter = iter(dataloader)

        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Handle different batch formats
            if isinstance(batch, tuple):
                tokens, labels = batch
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
            else:
                tokens = batch.to(self.device)
                labels = None

            optimizer.zero_grad()

            # Forward pass
            if isinstance(model, DeterministicLatentODE):
                loss, stats = model(tokens, loss_weights=(1.0, 0.1, 1.0))
            elif isinstance(model, RaccoonLogClassifier) and labels is not None:
                loss, stats = model(tokens, labels)
            else:
                output = model(tokens)
                loss = output.mean() if output.dim() > 0 else output
                stats = {}

            # Check loss
            if not monitor.check_loss(loss):
                results['nan_count'] += torch.isnan(loss).sum().item()
                results['inf_count'] += torch.isinf(loss).sum().item()
                print(f"  Step {step}: ❌ Invalid loss detected!")
                continue

            # Backward pass
            loss.backward()

            # Check and clip gradients
            grad_info = GradientManager.clip_gradients(
                model.parameters(),
                max_norm=1.0
            )

            if grad_info['grad_norm_before'] > 100:
                results['gradient_explosions'] += 1
                print(f"  Step {step}: ⚠️ Gradient explosion detected (norm={grad_info['grad_norm_before']:.2f})")

            optimizer.step()

            # Record step info
            step_info = {
                'loss': loss.item(),
                'grad_norm': grad_info['grad_norm_after'],
                'clipped': grad_info['clipped']
            }
            results['steps'].append(step_info)

            # Progress
            if step % 2 == 0:
                print(f"  Step {step}: loss={loss.item():.6f}, grad_norm={grad_info['grad_norm_after']:.4f}")

        # Summary
        print(f"\n📊 Training Stability Summary:")
        print(f"  NaN losses: {results['nan_count']}")
        print(f"  Inf losses: {results['inf_count']}")
        print(f"  Gradient explosions: {results['gradient_explosions']}")

        if results['steps']:
            losses = [s['loss'] for s in results['steps']]
            print(f"  Mean loss: {np.mean(losses):.6f}")
            print(f"  Loss std: {np.std(losses):.6f}")

        return results

    def run_all_tests(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """Run all stability tests."""
        print("\n" + "🔬"*30)
        print("COMPREHENSIVE NUMERICAL STABILITY TESTING")
        print("🔬"*30)

        all_results = {}

        # Get sample batch
        batch = next(iter(dataloader))
        if isinstance(batch, tuple):
            input_batch, _ = batch
        else:
            input_batch = batch
        input_batch = input_batch.to(self.device)

        # 1. Test initialization
        all_results['initialization'] = self.test_initialization_stability(model)

        # 2. Test forward pass
        all_results['forward_pass'] = self.test_forward_pass_stability(model, input_batch)

        # 3. Test gradients
        all_results['gradients'] = self.test_gradient_stability(model, input_batch)

        # 4. Test mixed precision
        all_results['mixed_precision'] = self.test_mixed_precision_compatibility(model, input_batch)

        # 5. Test training stability
        all_results['training'] = self.test_training_step_stability(model, dataloader, num_steps=10)

        # Generate report
        self.generate_report(all_results)

        return all_results

    def generate_report(self, results: Dict[str, Any]):
        """Generate a comprehensive stability report."""
        print("\n" + "="*60)
        print("NUMERICAL STABILITY REPORT")
        print("="*60)

        # Count issues
        issues = []

        # Check initialization
        if results['initialization']['nan_params'] > 0:
            issues.append("❌ NaN parameters at initialization")
        if results['initialization']['inf_params'] > 0:
            issues.append("❌ Inf parameters at initialization")

        # Check forward pass
        for test_name, test_result in results['forward_pass'].items():
            if isinstance(test_result, dict):
                if test_result.get('has_nan'):
                    issues.append(f"❌ NaN in forward pass ({test_name})")
                if test_result.get('has_inf'):
                    issues.append(f"❌ Inf in forward pass ({test_name})")

        # Check gradients
        if 'grad_stats' in results['gradients']:
            if results['gradients']['grad_stats']['params_with_nan']:
                issues.append("❌ NaN gradients detected")
            if results['gradients']['grad_stats']['params_with_inf']:
                issues.append("❌ Inf gradients detected")

        # Check training
        if results['training']['nan_count'] > 0:
            issues.append(f"❌ {results['training']['nan_count']} NaN losses during training")
        if results['training']['gradient_explosions'] > 0:
            issues.append(f"⚠️ {results['training']['gradient_explosions']} gradient explosions")

        # Report
        if not issues:
            print("\n✅ MODEL IS NUMERICALLY STABLE!")
            print("All tests passed without critical issues.")
        else:
            print("\n⚠️ NUMERICAL STABILITY ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")

            print("\n💡 RECOMMENDATIONS:")
            print("  1. Apply gradient clipping (max_norm=1.0)")
            print("  2. Use larger epsilon values (1e-6 instead of 1e-12)")
            print("  3. Clamp exponential inputs (-10, 10)")
            print("  4. Initialize weights with smaller values")
            print("  5. Use stable softmax implementation")
            print("  6. Consider using float32 for critical operations")


def main():
    parser = argparse.ArgumentParser(description='Test numerical stability')
    parser.add_argument('--model', choices=['ode', 'raccoon'], default='ode',
                        help='Model to test')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"🖥️ Using device: {device}")

    # Create model and dataset
    if args.model == 'ode':
        print("\n📦 Testing ODE Model")
        model = DeterministicLatentODE(
            vocab_size=vocab_size,
            latent_size=64,
            hidden_size=128,
            embed_size=64,
            num_slices=512
        ).to(device)

        dataset = SyntheticTargetDataset(n_samples=1000)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    else:  # raccoon
        print("\n🦝 Testing Raccoon Model")
        model = RaccoonLogClassifier(
            vocab_size=log_vocab_size,
            num_classes=NUM_LOG_CLASSES,
            latent_dim=32,
            hidden_dim=64,
            embed_dim=32,
            memory_size=100,
            adaptation_rate=1e-4
        ).to(device)

        dataset = LogDataset(n_samples=1000, seq_len=50)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Run tests
    tester = ComprehensiveStabilityTester(device)
    results = tester.run_all_tests(model, dataloader)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()