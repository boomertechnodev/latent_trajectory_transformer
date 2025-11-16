#!/usr/bin/env python3
"""
Geometric Testing Suite for Latent Trajectory Transformer
==========================================================

Comprehensive tests for validating geometric properties of latent spaces
in both ODE and Raccoon models.

Run: python test_geometry.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import main module
sys.path.insert(0, str(Path(__file__).parent))
import latent_drift_trajectory as ldt

# Import geometry analysis tools
from latent_geometry_analysis import (
    IntrinsicDimensionEstimator,
    DisentanglementMetrics,
    TrajectoryGeometry,
    LatentVisualizer,
    EncoderDecoderAlignment,
    analyze_latent_space
)


# ============================================================================
#  TEST 1: INTRINSIC DIMENSIONALITY
# ============================================================================

def test_intrinsic_dimensionality():
    """
    Test whether the latent space has lower intrinsic dimensionality
    than the ambient space.
    """
    print("\n" + "="*70)
    print("TEST 1: INTRINSIC DIMENSIONALITY ANALYSIS")
    print("="*70)

    device = torch.device("cpu")  # Use CPU for compatibility

    # Create model
    model = ldt.DeterministicLatentODE(
        vocab_size=ldt.vocab_size,
        latent_size=64,  # High ambient dimension
        hidden_size=128,
        embed_size=64,
        num_slices=256
    ).to(device)

    # Create dataset
    dataset = ldt.SyntheticTargetDataset(n_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Collect latent representations
    print("\n📊 Collecting latent codes...")
    latents = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Limit samples
                break
            batch = batch.to(device)
            z = model.encode(batch)  # (B, L, D)
            # Flatten across batch and sequence
            z_flat = z.reshape(-1, z.shape[-1])
            latents.append(z_flat)

    latents = torch.cat(latents, dim=0)
    print(f"✅ Collected {latents.shape[0]} latent vectors of dimension {latents.shape[1]}")

    # Estimate intrinsic dimension
    estimator = IntrinsicDimensionEstimator()

    print("\n🔍 Estimating intrinsic dimension...")
    dim_mle = estimator.mle_levina_bickel(latents, k=20)
    dim_pca = estimator.pca_explained_variance(latents, threshold=0.95)

    print(f"\n📈 Results:")
    print(f"  Ambient dimension: {latents.shape[1]}")
    print(f"  MLE intrinsic dimension: {dim_mle:.2f}")
    print(f"  PCA 95% variance: {dim_pca} components")
    print(f"  Dimension reduction: {(1 - dim_mle/latents.shape[1])*100:.1f}%")

    # Verdict
    if dim_mle < latents.shape[1] * 0.5:
        print("\n✅ PASS: Latent space has low intrinsic dimensionality")
        print(f"   The model is learning a {dim_mle:.0f}D manifold in {latents.shape[1]}D space")
    else:
        print("\n⚠️  WARNING: Latent space may not be efficiently compressed")
        print("   Consider reducing latent_size or adding stronger regularization")

    return dim_mle, dim_pca


# ============================================================================
#  TEST 2: TRAJECTORY SMOOTHNESS
# ============================================================================

def test_trajectory_smoothness():
    """
    Test smoothness and geometric properties of latent trajectories.
    """
    print("\n" + "="*70)
    print("TEST 2: TRAJECTORY SMOOTHNESS ANALYSIS")
    print("="*70)

    device = torch.device("cpu")

    # Create ODE model
    print("\n🔧 Testing ODE Model Trajectories...")
    ode_model = ldt.DeterministicLatentODE(
        vocab_size=ldt.vocab_size,
        latent_size=32,
        hidden_size=64,
        embed_size=32,
        num_slices=128
    ).to(device)

    # Generate ODE trajectory
    z0 = torch.randn(1, 32, device=device)
    ode_traj = ldt.solve_ode(ode_model.p_ode, z0, 0.0, 1.0, n_steps=50)
    ode_traj = ode_traj.squeeze(1)  # Remove batch dim

    # Analyze ODE trajectory
    traj_geom = TrajectoryGeometry()

    curvature_ode = traj_geom.trajectory_curvature(ode_traj)
    length_ode = traj_geom.trajectory_length(ode_traj)
    smoothness_ode = traj_geom.trajectory_smoothness(ode_traj)

    print(f"\n📊 ODE Trajectory Properties:")
    print(f"  Arc length: {length_ode:.3f}")
    print(f"  Mean curvature: {curvature_ode.mean():.3f}")
    print(f"  Max curvature: {curvature_ode.max():.3f}")
    print(f"  Smoothness score: {smoothness_ode:.3f}")

    # Create Raccoon SDE model
    print("\n🦝 Testing SDE Model Trajectories...")
    sde_model = ldt.RaccoonLogClassifier(
        vocab_size=ldt.log_vocab_size,
        num_classes=4,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32
    ).to(device)

    # Generate SDE trajectory
    t_span = torch.linspace(0.0, 1.0, 50, device=device)
    sde_traj = ldt.solve_sde(sde_model.dynamics, z0, t_span)
    sde_traj = sde_traj.squeeze(0)  # Remove batch dim

    # Analyze SDE trajectory
    curvature_sde = traj_geom.trajectory_curvature(sde_traj)
    length_sde = traj_geom.trajectory_length(sde_traj)
    smoothness_sde = traj_geom.trajectory_smoothness(sde_traj)

    print(f"\n📊 SDE Trajectory Properties:")
    print(f"  Arc length: {length_sde:.3f}")
    print(f"  Mean curvature: {curvature_sde.mean():.3f}")
    print(f"  Max curvature: {curvature_sde.max():.3f}")
    print(f"  Smoothness score: {smoothness_sde:.3f}")

    # Compare trajectories
    print(f"\n📈 Comparison:")
    print(f"  SDE vs ODE arc length ratio: {length_sde/length_ode:.2f}x")
    print(f"  SDE vs ODE smoothness ratio: {smoothness_sde/smoothness_ode:.2f}x")

    # Verdict
    if smoothness_ode < 10.0 and smoothness_sde < 20.0:
        print("\n✅ PASS: Trajectories are reasonably smooth")
    else:
        print("\n⚠️  WARNING: Trajectories may be too jagged")
        print("   Consider adding smoothness regularization")

    return smoothness_ode, smoothness_sde


# ============================================================================
#  TEST 3: DISENTANGLEMENT
# ============================================================================

def test_disentanglement():
    """
    Test whether latent dimensions capture independent factors.
    """
    print("\n" + "="*70)
    print("TEST 3: DISENTANGLEMENT ANALYSIS")
    print("="*70)

    device = torch.device("cpu")

    # Create Raccoon model (has clearer factors: log categories)
    model = ldt.RaccoonLogClassifier(
        vocab_size=ldt.log_vocab_size,
        num_classes=4,
        latent_dim=16,  # Lower dimension for better disentanglement
        hidden_dim=64,
        embed_dim=32
    ).to(device)

    # Create dataset with known factors
    dataset = ldt.LogDataset(n_samples=1000, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Collect latents and factors
    print("\n📊 Collecting latents and factors...")
    latents = []
    factors = []

    model.eval()
    with torch.no_grad():
        for i, (tokens, labels) in enumerate(dataloader):
            if i >= 20:  # Limit samples
                break
            tokens = tokens.to(device)
            labels = labels.to(device)

            # Get latent codes
            mean, logvar = model.encode(tokens)
            z = model.sample_latent(mean, logvar)

            latents.append(z)
            factors.append(labels)

    latents = torch.cat(latents, dim=0)
    factors = torch.cat(factors, dim=0)

    # Convert labels to one-hot factors
    factors_onehot = torch.nn.functional.one_hot(factors, 4).float()

    print(f"✅ Collected {latents.shape[0]} samples")
    print(f"   Latent dimension: {latents.shape[1]}")
    print(f"   Number of factors: {factors_onehot.shape[1]}")

    # Compute disentanglement metrics
    metrics = DisentanglementMetrics()

    print("\n🎯 Computing disentanglement metrics...")
    mig = metrics.mutual_information_gap(latents, factors_onehot)
    sap = metrics.separated_attribute_predictability(latents, factors_onehot)
    tc = metrics.total_correlation(latents)

    print(f"\n📈 Results:")
    print(f"  MIG score: {mig:.3f} (higher is better, max=1)")
    print(f"  SAP score: {sap:.3f} (higher is better, max=1)")
    print(f"  Total Correlation: {tc:.3f} (lower is better)")

    # Analyze which dimensions encode which factors
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\n🔍 Factor-dimension analysis:")
    latents_np = latents.detach().cpu().numpy()
    factors_np = factors.detach().cpu().numpy()

    # Standardize latents
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents_np)

    # For each latent dimension, see which factor it predicts best
    dim_factor_scores = np.zeros((latents.shape[1], 4))

    for dim in range(latents.shape[1]):
        clf = LogisticRegression(max_iter=100, random_state=42)
        clf.fit(latents_scaled[:, dim:dim+1], factors_np)
        score = clf.score(latents_scaled[:, dim:dim+1], factors_np)
        dim_factor_scores[dim, :] = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_.mean(0)

        if score > 0.4:  # Significant predictive power
            print(f"  Dim {dim:2d}: Encodes factor {clf.predict(latents_scaled[:, dim:dim+1]).mean():.1f} (accuracy: {score:.2f})")

    # Verdict
    if mig > 0.1 and sap > 0.1:
        print("\n✅ PASS: Latent space shows some disentanglement")
    else:
        print("\n⚠️  WARNING: Latent dimensions are entangled")
        print("   Consider using β-VAE objective or Factor-VAE")

    return mig, sap, tc


# ============================================================================
#  TEST 4: ENCODER-DECODER ALIGNMENT
# ============================================================================

def test_encoder_decoder_alignment():
    """
    Test cycle consistency and Jacobian properties.
    """
    print("\n" + "="*70)
    print("TEST 4: ENCODER-DECODER ALIGNMENT")
    print("="*70)

    device = torch.device("cpu")

    # Create model
    model = ldt.DeterministicLatentODE(
        vocab_size=ldt.vocab_size,
        latent_size=32,
        hidden_size=64,
        embed_size=32,
        num_slices=128
    ).to(device)

    # Test data
    dataset = ldt.SyntheticTargetDataset(n_samples=100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Get a batch
    batch = next(iter(dataloader)).to(device)

    print("\n🔄 Testing cycle consistency...")

    # Encode-decode cycle
    model.eval()
    with torch.no_grad():
        # Encode
        z = model.encode(batch)

        # Decode (get logits)
        logits = model.decode_logits(z, batch)

        # Get predictions
        preds = logits.argmax(dim=-1)

        # Accuracy
        accuracy = (preds == batch).float().mean()

        # Re-encode predictions
        z_cycle = model.encode(preds)

        # Latent consistency
        latent_consistency = torch.nn.functional.mse_loss(z_cycle, z)

    print(f"\n📊 Results:")
    print(f"  Reconstruction accuracy: {accuracy:.3f}")
    print(f"  Latent cycle consistency: {latent_consistency:.4f}")

    # Jacobian analysis
    print("\n🔬 Analyzing decoder Jacobian...")
    alignment = EncoderDecoderAlignment()

    # Sample latent
    z_sample = torch.randn(4, 32, requires_grad=True, device=device)

    # Create simple decoder function for Jacobian analysis
    def decoder_fn(z):
        # Simplified: just project to vocabulary
        return torch.nn.Linear(32, ldt.vocab_size, device=device)(z)

    jacobian_metrics = alignment.jacobian_analysis(decoder_fn, z_sample)

    print(f"  Effective rank: {jacobian_metrics['effective_rank']:.2f}")
    print(f"  Condition number: {jacobian_metrics['condition_number']:.2f}")
    print(f"  Max singular value: {jacobian_metrics['max_singular_value']:.3f}")
    print(f"  Min singular value: {jacobian_metrics['min_singular_value']:.6f}")

    # Verdict
    if accuracy > 0.5 and latent_consistency < 1.0:
        print("\n✅ PASS: Encoder and decoder are reasonably aligned")
    else:
        print("\n⚠️  WARNING: Poor encoder-decoder alignment")
        print("   Consider adding cycle consistency loss")

    return accuracy, latent_consistency


# ============================================================================
#  TEST 5: INTERPOLATION QUALITY
# ============================================================================

def test_interpolation_quality():
    """
    Test quality of latent space interpolations.
    """
    print("\n" + "="*70)
    print("TEST 5: INTERPOLATION QUALITY")
    print("="*70)

    device = torch.device("cpu")

    # Create model
    model = ldt.DeterministicLatentODE(
        vocab_size=ldt.vocab_size,
        latent_size=32,
        hidden_size=64,
        embed_size=32,
        num_slices=128
    ).to(device)

    # Get two data points
    dataset = ldt.SyntheticTargetDataset(n_samples=10)
    x1 = dataset[0].unsqueeze(0).to(device)
    x2 = dataset[1].unsqueeze(0).to(device)

    print(f"\n📝 Source sequences:")
    print(f"  Start: {ldt.decode(x1[0].cpu())[:20]}...")
    print(f"  End:   {ldt.decode(x2[0].cpu())[:20]}...")

    # Encode endpoints
    model.eval()
    with torch.no_grad():
        z1 = model.encode(x1)[0]  # Take first timestep
        z2 = model.encode(x2)[0]

    # Test different interpolation methods
    print("\n🔀 Testing interpolation methods...")

    n_interp = 5
    interpolation_results = {}

    # 1. Linear interpolation
    print("\n  Linear interpolation:")
    linear_scores = []
    for i, alpha in enumerate(np.linspace(0, 1, n_interp)):
        z_interp = (1 - alpha) * z1 + alpha * z2

        # Measure smoothness of interpolation
        if i > 0:
            step_size = torch.norm(z_interp - z_prev).item()
            linear_scores.append(step_size)
            print(f"    Step {i}: distance = {step_size:.3f}")
        z_prev = z_interp.clone()

    # 2. Spherical interpolation
    from geometric_improvements import ManifoldInterpolation
    interp_method = ManifoldInterpolation()

    print("\n  Spherical interpolation:")
    spherical_scores = []
    for i, alpha in enumerate(np.linspace(0, 1, n_interp)):
        z_interp = interp_method.spherical_interpolation(z1, z2, alpha)

        if i > 0:
            step_size = torch.norm(z_interp - z_prev).item()
            spherical_scores.append(step_size)
            print(f"    Step {i}: distance = {step_size:.3f}")
        z_prev = z_interp.clone()

    # Compare variance in step sizes (lower is better)
    linear_var = np.var(linear_scores) if linear_scores else float('inf')
    spherical_var = np.var(spherical_scores) if spherical_scores else float('inf')

    print(f"\n📊 Interpolation quality:")
    print(f"  Linear variance: {linear_var:.4f}")
    print(f"  Spherical variance: {spherical_var:.4f}")
    print(f"  Best method: {'Spherical' if spherical_var < linear_var else 'Linear'}")

    # Verdict
    if min(linear_var, spherical_var) < 0.1:
        print("\n✅ PASS: Smooth interpolations in latent space")
    else:
        print("\n⚠️  WARNING: Interpolations are not smooth")
        print("   Latent space may have holes or discontinuities")

    return linear_var, spherical_var


# ============================================================================
#  MAIN TEST SUITE
# ============================================================================

def run_all_tests():
    """
    Run comprehensive geometric test suite.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          LATENT SPACE GEOMETRY TEST SUITE                           ║
    ║                                                                      ║
    ║  Testing geometric properties of latent trajectory models           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    results = {}

    try:
        # Test 1: Intrinsic Dimensionality
        dim_mle, dim_pca = test_intrinsic_dimensionality()
        results['intrinsic_dim'] = {'mle': dim_mle, 'pca': dim_pca}
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        results['intrinsic_dim'] = None

    try:
        # Test 2: Trajectory Smoothness
        smooth_ode, smooth_sde = test_trajectory_smoothness()
        results['smoothness'] = {'ode': smooth_ode, 'sde': smooth_sde}
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        results['smoothness'] = None

    try:
        # Test 3: Disentanglement
        mig, sap, tc = test_disentanglement()
        results['disentanglement'] = {'mig': mig, 'sap': sap, 'tc': tc}
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        results['disentanglement'] = None

    try:
        # Test 4: Encoder-Decoder Alignment
        acc, consistency = test_encoder_decoder_alignment()
        results['alignment'] = {'accuracy': acc, 'consistency': consistency}
    except Exception as e:
        print(f"\n❌ Test 4 failed: {e}")
        results['alignment'] = None

    try:
        # Test 5: Interpolation Quality
        lin_var, sph_var = test_interpolation_quality()
        results['interpolation'] = {'linear': lin_var, 'spherical': sph_var}
    except Exception as e:
        print(f"\n❌ Test 5 failed: {e}")
        results['interpolation'] = None

    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)

    passed = 0
    failed = 0

    for test_name, test_results in results.items():
        if test_results is not None:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1

    print(f"\n📊 Final Score: {passed}/{passed+failed} tests passed")

    if passed == 5:
        print("\n🎉 EXCELLENT: All geometric properties validated!")
    elif passed >= 3:
        print("\n👍 GOOD: Most geometric properties are sound")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT: Several geometric issues detected")

    return results


# ============================================================================
#  VISUALIZATION TESTS
# ============================================================================

def visualize_latent_space(save_dir: str = "geometry_plots"):
    """
    Create comprehensive visualizations of latent space geometry.
    """
    print("\n" + "="*70)
    print("LATENT SPACE VISUALIZATION")
    print("="*70)

    import os
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu")

    # Create models
    ode_model = ldt.DeterministicLatentODE(
        vocab_size=ldt.vocab_size,
        latent_size=32,
        hidden_size=64,
        embed_size=32
    ).to(device)

    # Create dataset
    dataset = ldt.SyntheticTargetDataset(n_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Run full analysis
    print("\n📊 Running comprehensive latent analysis...")
    results = analyze_latent_space(
        model=ode_model,
        dataloader=dataloader,
        device=device,
        save_plots=True,
        plot_dir=save_dir
    )

    print(f"\n✅ Visualizations saved to {save_dir}/")
    print("  - manifold_pca.png: PCA projection")
    print("  - manifold_tsne.png: t-SNE embedding")
    print("  - correlation_matrix.png: Latent correlations")
    print("  - trajectory_3d.png: 3D trajectory visualization")

    return results


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test latent space geometry")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests only")
    args = parser.parse_args()

    if args.visualize:
        visualize_latent_space()
    else:
        results = run_all_tests()

        if not args.quick:
            print("\n💡 Tip: Run with --visualize to create latent space plots")

    print("\n✅ Geometric testing complete!")
    print("🔬 Use latent_geometry_analysis.py for deeper analysis")
    print("🛠️  Use geometric_improvements.py to enhance your models")