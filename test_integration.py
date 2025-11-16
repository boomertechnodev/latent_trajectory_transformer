#!/usr/bin/env python3
"""
Integration Test Suite - End-to-End Pipeline Testing

Tests that verify components work together correctly in realistic scenarios.
Unlike unit tests, these test full workflows from input to output.

Coverage:
- Raccoon training → continuous learning → evaluation pipeline
- ODE encoding → trajectory → decoding → sampling pipeline
- Checkpoint save → load → resume training pipeline
- Memory replay + gradient updates integration
- Multi-epoch training with concept drift handling
"""

import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Import from main file
sys.path.insert(0, '/home/user/latent_trajectory_transformer')
from latent_drift_trajectory import (
    # Raccoon components
    RaccoonLogClassifier,
    RaccoonDynamics,
    RaccoonMemory,
    RaccoonFlow,
    solve_sde,
    LogDataset,
    log_vocab_size,
    NUM_LOG_CLASSES,
    train_raccoon_classifier,
    continuous_learning_phase,
    # ODE components
    DeterministicLatentODE,
    SyntheticTargetDataset,
    sample_sequences_ode,
    train_ode,
    vocab_size,
    decode,
    # Statistical tests
    FastEppsPulley,
    SlicingUnivariateTest,
)

print("=" * 80)
print("INTEGRATION TEST SUITE - End-to-End Pipeline Testing")
print("=" * 80)

# ============================================================================
# TEST 1: Raccoon Full Training Pipeline (Initial → Continuous → Eval)
# ============================================================================
print("\n[TEST 1] Raccoon Full Training Pipeline")
print("-" * 80)

# Create model
model = RaccoonLogClassifier(
    vocab_size=log_vocab_size,
    num_classes=NUM_LOG_CLASSES,
    latent_dim=16,
    hidden_dim=32,
    embed_dim=16,
    memory_size=100,
    adaptation_rate=1e-3,
)

# Create datasets
train_dataset = LogDataset(n_samples=200, seq_len=30, drift_point=None)
test_dataset = LogDataset(n_samples=50, seq_len=30, drift_point=None)
drift_dataset = LogDataset(n_samples=100, seq_len=30, drift_point=50)

# Phase 1: Initial supervised training
print("Phase 1: Initial supervised training (50 steps)...")
initial_loss, initial_stats = train_raccoon_classifier(
    model,
    train_dataset,
    num_steps=50,
    batch_size=16,
    lr=1e-3,
    loss_weights=(1.0, 0.1, 0.01),
    device='cpu',
)

assert not torch.isnan(initial_loss), "Training loss is NaN"
assert initial_loss > 0, "Training loss should be positive"
assert 'accuracy' in initial_stats, "Missing accuracy in stats"
assert initial_stats['accuracy'] > 0.0, "Accuracy should be > 0"

print(f"✓ Phase 1 complete: loss={initial_loss:.4f}, acc={initial_stats['accuracy']:.2%}")

# Evaluate after Phase 1
print("\nEvaluating after Phase 1...")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for i in range(len(test_dataset)):
        tokens = test_dataset[i][0].unsqueeze(0)
        label = test_dataset[i][1].unsqueeze(0)

        loss, stats = model(tokens, label)

        # Get predictions
        mean, _ = model.encode(tokens)
        t_span = torch.linspace(0.0, 0.1, 3)
        z_traj = solve_sde(model.dynamics, mean, t_span)
        z_final = z_traj[:, -1]
        t_final = t_span[-1:].expand(z_final.size(0)).unsqueeze(1)
        z_semantic, _ = model.flow(z_final, t_final)
        logits = model.classifier(z_semantic)
        pred = logits.argmax(dim=-1)

        test_correct += (pred == label).sum().item()
        test_total += label.size(0)

phase1_accuracy = test_correct / test_total
print(f"✓ Phase 1 test accuracy: {phase1_accuracy:.2%}")
assert phase1_accuracy > 0.2, "Phase 1 accuracy too low (should be > 20%)"

# Phase 2: Continuous learning with concept drift
print("\nPhase 2: Continuous learning (100 samples with drift)...")
rolling_accuracies = continuous_learning_phase(
    model,
    drift_dataset,
    num_samples=100,
    device='cpu',
)

assert len(rolling_accuracies) > 0, "No accuracies recorded"
final_rolling_acc = rolling_accuracies[-1]
print(f"✓ Phase 2 complete: final rolling accuracy={final_rolling_acc:.2%}")
assert final_rolling_acc > 0.1, "Continuous learning accuracy too low"

# Verify memory buffer grew
assert len(model.memory) > 0, "Memory buffer should not be empty"
print(f"✓ Memory buffer size: {len(model.memory)}")

# Phase 3: Final evaluation
print("\nPhase 3: Final evaluation after continuous learning...")
model.eval()
final_correct = 0
final_total = 0

with torch.no_grad():
    for i in range(len(test_dataset)):
        tokens = test_dataset[i][0].unsqueeze(0)
        label = test_dataset[i][1].unsqueeze(0)

        loss, stats = model(tokens, label)

        # Get predictions
        mean, _ = model.encode(tokens)
        t_span = torch.linspace(0.0, 0.1, 3)
        z_traj = solve_sde(model.dynamics, mean, t_span)
        z_final = z_traj[:, -1]
        t_final = t_span[-1:].expand(z_final.size(0)).unsqueeze(1)
        z_semantic, _ = model.flow(z_final, t_final)
        logits = model.classifier(z_semantic)
        pred = logits.argmax(dim=-1)

        final_correct += (pred == label).sum().item()
        final_total += label.size(0)

final_accuracy = final_correct / final_total
print(f"✓ Final test accuracy: {final_accuracy:.2%}")

# Check for catastrophic forgetting
forgetting = phase1_accuracy - final_accuracy
print(f"✓ Forgetting: {forgetting:.2%} (negative means improvement)")
assert forgetting < 0.5, "Catastrophic forgetting detected (>50% accuracy drop)"

print("\n[TEST 1] ✓ PASSED - Raccoon full pipeline works correctly")

# ============================================================================
# TEST 2: ODE Full Generation Pipeline (Encode → Trajectory → Decode → Sample)
# ============================================================================
print("\n[TEST 2] ODE Full Generation Pipeline")
print("-" * 80)

# Create ODE model
ode_model = DeterministicLatentODE(
    vocab_size=vocab_size,
    latent_size=32,
    hidden_size=64,
    embed_size=32,
    num_slices=128,
)

# Create dataset
ode_dataset = SyntheticTargetDataset(n_samples=500)

# Train briefly
print("Training ODE model (100 steps)...")
ode_losses = []
optimizer = torch.optim.AdamW(ode_model.parameters(), lr=1e-3)

for step in range(100):
    tokens = torch.stack([ode_dataset[i] for i in range(16)])

    ode_model.train()
    loss, stats = ode_model(tokens, loss_weights=(1.0, 0.05, 1.0))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ode_model.parameters(), max_norm=1.0)
    optimizer.step()

    ode_losses.append(loss.item())

    if (step + 1) % 25 == 0:
        print(f"  Step {step+1}: loss={loss.item():.4f}")

print(f"✓ Training complete: initial loss={ode_losses[0]:.4f}, final loss={ode_losses[-1]:.4f}")
assert ode_losses[-1] < ode_losses[0], "Loss should decrease during training"

# Test full generation pipeline
print("\nTesting full generation pipeline...")
ode_model.eval()

with torch.no_grad():
    # Generate samples
    samples_fixed, samples_random = sample_sequences_ode(
        ode_model,
        seq_len=66,
        n_samples=4,
        device='cpu',
    )

    assert samples_fixed.shape == (4, 66), f"Fixed samples wrong shape: {samples_fixed.shape}"
    assert samples_random.shape == (4, 66), f"Random samples wrong shape: {samples_random.shape}"

    # Verify token values are valid
    assert samples_fixed.min() >= 0, "Fixed samples have negative token IDs"
    assert samples_fixed.max() < vocab_size, f"Fixed samples have out-of-vocab tokens: max={samples_fixed.max()}"
    assert samples_random.min() >= 0, "Random samples have negative token IDs"
    assert samples_random.max() < vocab_size, f"Random samples have out-of-vocab tokens: max={samples_random.max()}"

    # Decode and verify they're valid strings
    print("\nGenerated samples (fixed Z):")
    for i in range(2):
        decoded = decode(samples_fixed[i])
        print(f"  {i}: {decoded}")
        assert len(decoded) == 66, f"Decoded length wrong: {len(decoded)}"
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_!>?" for c in decoded), "Invalid characters in decoded sample"

    print("\nGenerated samples (random Z):")
    for i in range(2):
        decoded = decode(samples_random[i])
        print(f"  {i}: {decoded}")
        assert len(decoded) == 66, f"Decoded length wrong: {len(decoded)}"

print("\n[TEST 2] ✓ PASSED - ODE full generation pipeline works correctly")

# ============================================================================
# TEST 3: Checkpoint Save/Load/Resume Pipeline
# ============================================================================
print("\n[TEST 3] Checkpoint Save/Load/Resume Pipeline")
print("-" * 80)

# Create temporary directory for checkpoints
temp_dir = tempfile.mkdtemp(prefix="test_checkpoints_")
print(f"Using temp dir: {temp_dir}")

try:
    # Create model
    checkpoint_model = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=NUM_LOG_CLASSES,
        latent_dim=16,
        hidden_dim=32,
        embed_dim=16,
        memory_size=50,
    )

    # Create dataset
    ckpt_dataset = LogDataset(n_samples=100, seq_len=30)

    # Train for N steps
    print("Training for 25 steps before checkpoint...")
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=1e-3)

    losses_before = []
    for step in range(25):
        tokens = torch.stack([ckpt_dataset[i][0] for i in range(8)])
        labels = torch.stack([ckpt_dataset[i][1] for i in range(8)])

        checkpoint_model.train()
        loss, stats = checkpoint_model(tokens, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_before.append(loss.item())

    print(f"✓ Pre-checkpoint training: loss {losses_before[0]:.4f} → {losses_before[-1]:.4f}")

    # Save checkpoint
    checkpoint_path = Path(temp_dir) / "checkpoint.pt"
    print(f"\nSaving checkpoint to {checkpoint_path}...")

    checkpoint = {
        'model_state': checkpoint_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': 25,
        'loss': losses_before[-1],
        'memory_state': checkpoint_model.memory.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    assert checkpoint_path.exists(), "Checkpoint file not created"
    checkpoint_size = checkpoint_path.stat().st_size
    print(f"✓ Checkpoint saved ({checkpoint_size:,} bytes)")

    # Create new model and load checkpoint
    print("\nLoading checkpoint into new model...")
    resumed_model = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=NUM_LOG_CLASSES,
        latent_dim=16,
        hidden_dim=32,
        embed_dim=16,
        memory_size=50,
    )

    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)

    loaded_checkpoint = torch.load(checkpoint_path)
    resumed_model.load_state_dict(loaded_checkpoint['model_state'])
    resumed_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
    resumed_model.memory.load_state_dict(loaded_checkpoint['memory_state'])

    print(f"✓ Checkpoint loaded from step {loaded_checkpoint['step']}")

    # Verify model parameters match
    for (name1, p1), (name2, p2) in zip(
        checkpoint_model.named_parameters(),
        resumed_model.named_parameters()
    ):
        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
        assert torch.allclose(p1, p2), f"Parameter mismatch in {name1}"

    print("✓ Model parameters match exactly")

    # Verify memory state matches
    assert len(resumed_model.memory) == len(checkpoint_model.memory), "Memory size mismatch"
    print(f"✓ Memory buffer size matches: {len(resumed_model.memory)}")

    # Resume training
    print("\nResuming training for another 25 steps...")
    losses_after = []

    for step in range(25, 50):
        tokens = torch.stack([ckpt_dataset[i][0] for i in range(8)])
        labels = torch.stack([ckpt_dataset[i][1] for i in range(8)])

        resumed_model.train()
        loss, stats = resumed_model(tokens, labels)

        resumed_optimizer.zero_grad()
        loss.backward()
        resumed_optimizer.step()

        losses_after.append(loss.item())

    print(f"✓ Post-checkpoint training: loss {losses_after[0]:.4f} → {losses_after[-1]:.4f}")

    # Verify training continued smoothly (no jump in loss)
    loss_jump = abs(losses_after[0] - losses_before[-1])
    print(f"✓ Loss continuity: jump={loss_jump:.4f}")
    assert loss_jump < 2.0, f"Loss jumped too much after resume: {loss_jump}"

    print("\n[TEST 3] ✓ PASSED - Checkpoint save/load/resume works correctly")

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"✓ Cleaned up temp dir: {temp_dir}")

# ============================================================================
# TEST 4: Memory Replay + Gradient Updates Integration
# ============================================================================
print("\n[TEST 4] Memory Replay + Gradient Updates Integration")
print("-" * 80)

# Create model with memory
mem_model = RaccoonLogClassifier(
    vocab_size=log_vocab_size,
    num_classes=NUM_LOG_CLASSES,
    latent_dim=16,
    hidden_dim=32,
    embed_dim=16,
    memory_size=50,
)

# Add items to memory
print("Populating memory buffer...")
mem_dataset = LogDataset(n_samples=60, seq_len=30)

for i in range(30):
    tokens = mem_dataset[i][0]
    label = mem_dataset[i][1]

    # Add to memory with increasing scores
    mem_model.memory.add({
        'tokens': tokens.unsqueeze(0).cpu(),
        'label': label.unsqueeze(0).cpu()
    }, score=float(i))

print(f"✓ Memory buffer size: {len(mem_model.memory)}")
assert len(mem_model.memory) == 30, "Memory should have 30 items"

# Sample from memory
print("\nSampling from memory...")
sampled = mem_model.memory.sample(10, device='cpu')
assert len(sampled) == 10, f"Expected 10 samples, got {len(sampled)}"
print(f"✓ Sampled {len(sampled)} items")

# Verify sampled items have correct structure
for item in sampled:
    assert 'tokens' in item, "Missing tokens in sampled item"
    assert 'label' in item, "Missing label in sampled item"
    assert item['tokens'].shape[1] == 30, f"Wrong token shape: {item['tokens'].shape}"
    assert item['label'].shape[0] == 1, f"Wrong label shape: {item['label'].shape}"

print("✓ Sampled items have correct structure")

# Test gradient updates with memory replay
print("\nTesting gradient updates with memory replay...")
optimizer = torch.optim.SGD(mem_model.parameters(), lr=1e-3)

# New sample
new_tokens = mem_dataset[40][0].unsqueeze(0)
new_label = mem_dataset[40][1].unsqueeze(0)

# Get initial parameters
initial_params = {name: p.clone() for name, p in mem_model.named_parameters()}

# Single update step (uses memory internally)
mem_model.continuous_update(new_tokens, new_label)

# Verify parameters changed
params_changed = False
for name, p in mem_model.named_parameters():
    if not torch.allclose(p, initial_params[name]):
        params_changed = True
        break

assert params_changed, "Parameters should change after gradient update"
print("✓ Gradient updates modified model parameters")

# Verify memory grew
assert len(mem_model.memory) > 30, "Memory should grow after continuous_update"
print(f"✓ Memory grew to {len(mem_model.memory)} items")

print("\n[TEST 4] ✓ PASSED - Memory replay + gradient updates integration works")

# ============================================================================
# TEST 5: Multi-Epoch Training with Concept Drift
# ============================================================================
print("\n[TEST 5] Multi-Epoch Training with Concept Drift")
print("-" * 80)

# Create model
drift_model = RaccoonLogClassifier(
    vocab_size=log_vocab_size,
    num_classes=NUM_LOG_CLASSES,
    latent_dim=16,
    hidden_dim=32,
    embed_dim=16,
    memory_size=100,
)

# Create datasets with drift
pre_drift = LogDataset(n_samples=100, seq_len=30, drift_point=None)
post_drift = LogDataset(n_samples=100, seq_len=30, drift_point=50)

# Epoch 1: Pre-drift training
print("Epoch 1: Training on pre-drift data...")
epoch1_loss, epoch1_stats = train_raccoon_classifier(
    drift_model,
    pre_drift,
    num_steps=30,
    batch_size=16,
    lr=1e-3,
    loss_weights=(1.0, 0.1, 0.01),
    device='cpu',
)

print(f"✓ Epoch 1: loss={epoch1_loss:.4f}, acc={epoch1_stats['accuracy']:.2%}")
assert epoch1_stats['accuracy'] > 0.0, "Epoch 1 should achieve some accuracy"

# Epoch 2: Post-drift training
print("\nEpoch 2: Training on post-drift data...")
epoch2_loss, epoch2_stats = train_raccoon_classifier(
    drift_model,
    post_drift,
    num_steps=30,
    batch_size=16,
    lr=1e-3,
    loss_weights=(1.0, 0.1, 0.01),
    device='cpu',
)

print(f"✓ Epoch 2: loss={epoch2_loss:.4f}, acc={epoch2_stats['accuracy']:.2%}")
assert epoch2_stats['accuracy'] > 0.0, "Epoch 2 should achieve some accuracy"

# Test on both pre and post drift data
print("\nTesting on pre-drift data...")
pre_drift_correct = 0
pre_drift_total = 0

drift_model.eval()
with torch.no_grad():
    for i in range(min(20, len(pre_drift))):
        tokens = pre_drift[i][0].unsqueeze(0)
        label = pre_drift[i][1].unsqueeze(0)

        mean, _ = drift_model.encode(tokens)
        t_span = torch.linspace(0.0, 0.1, 3)
        z_traj = solve_sde(drift_model.dynamics, mean, t_span)
        z_final = z_traj[:, -1]
        t_final = t_span[-1:].expand(z_final.size(0)).unsqueeze(1)
        z_semantic, _ = drift_model.flow(z_final, t_final)
        logits = drift_model.classifier(z_semantic)
        pred = logits.argmax(dim=-1)

        pre_drift_correct += (pred == label).sum().item()
        pre_drift_total += label.size(0)

pre_drift_acc = pre_drift_correct / pre_drift_total
print(f"✓ Pre-drift accuracy: {pre_drift_acc:.2%}")

print("\nTesting on post-drift data...")
post_drift_correct = 0
post_drift_total = 0

with torch.no_grad():
    for i in range(min(20, len(post_drift))):
        tokens = post_drift[i][0].unsqueeze(0)
        label = post_drift[i][1].unsqueeze(0)

        mean, _ = drift_model.encode(tokens)
        t_span = torch.linspace(0.0, 0.1, 3)
        z_traj = solve_sde(drift_model.dynamics, mean, t_span)
        z_final = z_traj[:, -1]
        t_final = t_span[-1:].expand(z_final.size(0)).unsqueeze(1)
        z_semantic, _ = drift_model.flow(z_final, t_final)
        logits = drift_model.classifier(z_semantic)
        pred = logits.argmax(dim=-1)

        post_drift_correct += (pred == label).sum().item()
        post_drift_total += label.size(0)

post_drift_acc = post_drift_correct / post_drift_total
print(f"✓ Post-drift accuracy: {post_drift_acc:.2%}")

# Model should handle both distributions reasonably
assert pre_drift_acc > 0.1, "Should maintain some accuracy on pre-drift data"
assert post_drift_acc > 0.1, "Should achieve some accuracy on post-drift data"

print("\n[TEST 5] ✓ PASSED - Multi-epoch training with concept drift works")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL 5 INTEGRATION TESTS PASSED!")
print("=" * 80)
print("""
✅ TEST 1: Raccoon Full Training Pipeline (Init → Continuous → Eval)
✅ TEST 2: ODE Full Generation Pipeline (Encode → Trajectory → Decode → Sample)
✅ TEST 3: Checkpoint Save/Load/Resume Pipeline
✅ TEST 4: Memory Replay + Gradient Updates Integration
✅ TEST 5: Multi-Epoch Training with Concept Drift

CONCLUSION:
All end-to-end integrations work correctly. Components integrate properly
without crashes or data flow issues. Ready for production deployment.
""")
print("=" * 80)
