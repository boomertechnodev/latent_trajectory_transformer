#!/usr/bin/env python3
"""
Test script to verify latent_drift_trajectory.py runs efficiently on CPU.
This validates the core trajectory planning mechanism before building the search app.
"""

import torch
import time
from latent_drift_trajectory import (
    RaccoonLogClassifier,
    LogDataset,
    train_raccoon_classifier,
    continuous_learning_phase,
    log_vocab_size  # Import the actual vocab size
)
from torch.utils.data import DataLoader

print("="*70)
print("LATENT TRAJECTORY TRANSFORMER - CPU BASELINE TEST")
print("="*70)

# Force CPU
device = torch.device("cpu")
print(f"\n✓ Device: {device}")

# Reduced configuration for fast CPU testing
config = {
    'vocab_size': log_vocab_size,  # Use actual log vocab size (39)
    'num_classes': 4,   # ERROR, WARNING, INFO, DEBUG
    'latent_dim': 16,   # Reduced from 32
    'hidden_dim': 32,   # Reduced from 64
    'embed_dim': 16,    # Reduced from 32
    'memory_size': 500, # Reduced from 2000
    'adaptation_rate': 1e-4,
}

print(f"\n📊 Configuration (CPU-optimized):")
print(f"  log_vocab_size: {log_vocab_size}")
for k, v in config.items():
    print(f"  {k:20s}: {v}")

# Create model
print(f"\n🦝 Initializing Raccoon model...")
model = RaccoonLogClassifier(**config).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {param_count:,}")

# Create small dataset for testing
print(f"\n📝 Creating test datasets...")
train_ds = LogDataset(n_samples=500, seq_len=50, drift_point=None)
test_ds = LogDataset(n_samples=100, seq_len=50, drift_point=None)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# Phase 1: Quick training test
print(f"\n🏋️ Phase 1: Training test (100 steps)...")
start_time = time.time()

train_raccoon_classifier(
    model=model,
    dataloader=train_loader,
    n_iter=100,
    device=device,
    loss_weights=(1.0, 0.1, 0.01),
)

train_time = time.time() - start_time
print(f"\n  Training time: {train_time:.2f}s ({train_time/100*1000:.1f}ms/step)")

# Test evaluation
print(f"\n📊 Evaluating on test set...")
model.eval()
correct = 0
total = 0

eval_start = time.time()
with torch.no_grad():
    for tokens, labels in test_loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        mean, logvar = model.encode(tokens)
        z = model.sample_latent(mean, logvar)
        logits = model.classify(z)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

eval_time = time.time() - eval_start
accuracy = correct / total

print(f"  Test Accuracy: {accuracy:.3f} ({correct}/{total})")
print(f"  Inference time: {eval_time:.2f}s ({eval_time/len(test_loader)*1000:.1f}ms/batch)")

# Test continual learning
print(f"\n🔄 Phase 2: Continual learning test (50 samples)...")
drift_ds = LogDataset(n_samples=50, seq_len=50, drift_point=25)
drift_loader = DataLoader(drift_ds, batch_size=1, shuffle=True)

cl_start = time.time()
continuous_learning_phase(
    model=model,
    dataloader=drift_loader,
    n_samples=50,
    device=device,
)
cl_time = time.time() - cl_start

print(f"\n  Continual learning time: {cl_time:.2f}s ({cl_time/50*1000:.1f}ms/sample)")
print(f"  Memory buffer size: {len(model.memory)}")

# Test latent trajectory visualization
print(f"\n🎯 Testing latent trajectory planning...")
with torch.no_grad():
    # Get a test sequence
    test_tokens, _ = test_ds[0]
    test_tokens = test_tokens.unsqueeze(0).to(device)

    # Encode to latent trajectory
    mean, logvar = model.encode(test_tokens)
    z0 = model.sample_latent(mean, logvar)

    # Plan trajectory with SDE (short rollout)
    t_span = torch.linspace(0.0, 0.1, 5, device=device)
    from latent_drift_trajectory import solve_sde
    z_trajectory = solve_sde(model.dynamics, z0, t_span)

    print(f"  Initial latent z0 shape: {z0.shape}")
    print(f"  Trajectory shape: {z_trajectory.shape}")
    print(f"  Trajectory norm: {z_trajectory.norm(dim=-1).mean():.3f}")

    # Apply normalizing flow
    z_flow, log_det = model.flow(z_trajectory[:, -1], t_span[-1:])
    print(f"  After flow: {z_flow.shape}, log_det: {log_det.item():.3f}")

print("\n" + "="*70)
print("✅ BASELINE TEST COMPLETE")
print("="*70)
print("\nKey Findings:")
print(f"  • Training: {train_time/100*1000:.1f}ms/step on CPU")
print(f"  • Inference: {eval_time/len(test_loader)*1000:.1f}ms/batch on CPU")
print(f"  • Continual learning: {cl_time/50*1000:.1f}ms/sample on CPU")
print(f"  • Model params: {param_count:,}")
print(f"  • Memory buffer: {len(model.memory)} experiences")
print("\nReady for Phase 2: Flow matching enhancement! 🚀")
