---
name: normalizing-flows
description: Comprehensive technical knowledge for normalizing flows, including mathematical foundations of change-of-variables, Jacobian determinants, modern flow architectures (NICE, RealNVP, Glow, MAF, IAF, spline flows, FFJORD), invertible neural networks, and advanced implementation patterns in PyTorch.
---

# Normalizing Flows - Deep Technical Knowledge

## Mathematical Foundations

### Change of Variables Formula

**Basic Theorem:**
If X ~ p_X and Y = f(X) where f is bijective with f⁻¹ = g, then:
```
p_Y(y) = p_X(f⁻¹(y)) |det J_{f⁻¹}(y)|
     = p_X(g(y)) |det J_g(y)|
```

**Log-Probability Form** (used in training):
```
log p_Y(y) = log p_X(g(y)) + log |det J_g(y)|
```

**For flows (X → Z):**
```
log p_X(x) = log p_Z(f(x)) + log |det J_f(x)|
```

**Key Insight**: We need:
1. Bijective f (invertible transformation)
2. Tractable base distribution p_Z (usually Gaussian)
3. Efficient log-det Jacobian computation

### Jacobian Determinant Properties

**Determinant of product:**
```
det(AB) = det(A) · det(B)
```

**Determinant of inverse:**
```
det(A⁻¹) = 1 / det(A)
log |det(A⁻¹)| = -log |det(A)|
```

**Triangular matrix:**
If J is triangular (upper or lower), then:
```
det(J) = ∏ J_{ii} (product of diagonal elements)
log |det(J)| = ∑ log |J_{ii}|
```

**Block triangular:**
If J = [[A, B], [0, C]], then:
```
det(J) = det(A) · det(C)
```

### Composing Flows

**Sequential composition:**
If z = f_K ∘ ... ∘ f_2 ∘ f_1(x), then:
```
log p(x) = log p(z) + ∑_{i=1}^K log |det J_{f_i}|
```

**Stacking Coupling Layers:**
Each layer has simple log-det, accumulate:
```python
log_det_total = 0
z = x
for layer in layers:
    z, log_det_i = layer(z)
    log_det_total += log_det_i

log_prob = log_prior(z) + log_det_total
```

## Flow Architecture Zoo

### 1. NICE (Nonlinear Independent Components Estimation)

**Paper**: Dinh et al., ICLR 2015

**Coupling Function:**
```
Given partition x = [x_A, x_B]:
y_A = x_A
y_B = x_B + m(x_A)
```
where m is an arbitrary neural network.

**Jacobian:**
```
J = [[I,  0   ],
     [∂m/∂x_A, I]]
```
Triangular → det(J) = 1 → log |det(J)| = 0

**Pros:**
- Exact inverse: y_B - m(y_A) = x_B
- Zero computational cost for log-det
- Volume-preserving transformation

**Cons:**
- Limited expressivity (only translation)
- Needs many layers for complex distributions
- No learned variance

### 2. RealNVP (Real-valued Non-Volume Preserving)

**Paper**: Dinh et al., ICLR 2017

**Affine Coupling:**
```
y_A = x_A
y_B = x_B ⊙ exp(s(x_A)) + t(x_A)
```
where s = scale network, t = translation network

**Jacobian:**
```
det(J) = exp(∑ s_i(x_A))
log |det(J)| = ∑ s_i(x_A)
```

**Pros:**
- Efficient log-det (just sum of scales)
- Easy inverse: (y_B - t) ⊙ exp(-s)
- More expressive than NICE (learned variance)

**Cons:**
- Half dimensions unchanged per layer
- Needs alternating masks
- Scale can explode if unbounded

**Numerical Stability:**
```python
# Bound scale to prevent explosion
s = tanh(s_raw / 3) * 3  # s ∈ (-3, 3)
# Or: s = torch.clamp(s_raw, min=-5, max=5)
```

### 3. Glow (Generative Flow)

**Paper**: Kingma & Dhariwal, NeurIPS 2018

**Improvements over RealNVP:**
1. **Actnorm**: Learnable activation normalization (data-dependent init)
2. **Invertible 1x1 Conv**: Learnable permutation (instead of fixed shuffle)
3. **Affine coupling**: Same as RealNVP

**Actnorm:**
```
y = s ⊙ (x + b)
log |det| = ∑ log |s_i|
```
Initialized from first batch:
```python
s = 1 / (std(x) + eps)
b = -mean(x)
```

**1x1 Convolution (channel permutation):**
```
y = Wx  where W is learnable and det(W) ≠ 0
log |det| = log |det(W)|
```

Use LU decomposition for efficient det:
```python
W = PLU  # P=permutation, L=lower, U=upper
log |det(W)| = ∑ log |U_ii|
```

### 4. MAF (Masked Autoregressive Flow)

**Paper**: Papamakarios et al., NeurIPS 2017

**Autoregressive Transformation:**
```
y_i = x_i ⊙ exp(α_i(x_{<i})) + μ_i(x_{<i})
```
Each dimension depends on all previous dimensions.

**Jacobian:** Triangular
```
log |det| = ∑_i α_i(x_{<i})
```

**Pros:**
- Very expressive (each dim conditioned on all previous)
- Exact likelihood computation (one forward pass)

**Cons:**
- **Slow sampling**: O(D) sequential steps (D = dimensionality)
- Requires autoregressive networks (MADE, PixelCNN++)

**Implementation:**
Use MADE (Masked Autoencoder for Distribution Estimation):
```python
class MADE(nn.Module):
    def __init__(self, dim, hidden_dim):
        # Masks enforce autoregressive property
        self.masks = create_masks(dim, hidden_dim)

    def forward(self, x):
        # Returns μ and α for each dimension
        h = masked_linear(x, self.masks[0])
        # ... more masked layers
        params = masked_linear(h, self.masks[-1])
        mu, alpha = params.chunk(2, dim=-1)
        return mu, alpha
```

### 5. IAF (Inverse Autoregressive Flow)

**Paper**: Kingma et al., NeurIPS 2016

**Inverse Autoregressive:**
```
y_i = x_i ⊙ exp(α_i(y_{<i})) + μ_i(y_{<i})
```

**Pros:**
- **Fast sampling**: Can compute all y in parallel
- Good for VAE decoders

**Cons:**
- **Slow density**: O(D) sequential steps
- Opposite tradeoff to MAF

**When to use:**
- MAF: When you need fast density (inference, classification)
- IAF: When you need fast sampling (generation, VAE decoder)

### 6. Neural Spline Flows

**Paper**: Durkan et al., NeurIPS 2019

**Monotonic Rational Quadratic Splines:**
Each dimension transformed by smooth monotonic spline:
```
y_i = spline_i(x_i; θ_i(x_{<i}))
```

**Spline Properties:**
- Piecewise rational quadratic functions
- Monotonic (invertible)
- Smooth (differentiable)
- Unbounded domain

**Jacobian:**
```
dy_i/dx_i = spline_derivative_i(x_i)
log |det| = ∑_i log spline_derivative_i(x_i)
```

**Pros:**
- Very expressive transformations
- Smooth, no sharp corners
- Unbounded support

**Cons:**
- Complex implementation
- More parameters than affine coupling

### 7. FFJORD (Free-Form Continuous Dynamics)

**Paper**: Grathwohl et al., ICLR 2019

**Continuous Normalizing Flow:**
```
dz/dt = f(z, t, θ)
z(t1) = z(t0) + ∫_{t0}^{t1} f(z(t), t) dt
```

**Log-Determinant:**
```
log |det ∂z(t1)/∂z(t0)| = -∫_{t0}^{t1} tr(∂f/∂z) dt
```

**Trace Estimation (Hutchinson):**
```python
def trace_estimator(f, z, eps):
    """
    Unbiased trace estimate: tr(J) ≈ eps^T J eps
    where eps ~ N(0,I)
    """
    f_eval = f(z)
    grad = autograd.grad(f_eval, z, eps, create_graph=True)[0]
    return (grad * eps).sum(dim=-1)
```

**Pros:**
- Unrestricted architecture for f
- Flexible dynamics
- No coupling constraints

**Cons:**
- Expensive trace estimation
- Slow (requires ODE solve)
- Memory intensive

### 8. Residual Flows

**Paper**: Chen et al., NeurIPS 2019

**Residual Transformation:**
```
y = x + g(x)
```
where g is Lipschitz continuous with Lip(g) < 1.

**Invertibility:**
Via fixed-point iteration or spectral normalization.

**Log-Determinant:**
Via power series approximation:
```
log |det(I + J_g)| ≈ tr(J_g) - (1/2)tr(J_g²) + (1/3)tr(J_g³) - ...
```

**Pros:**
- Unrestricted architecture
- Free-form transformations

**Cons:**
- Approximate inverse
- Expensive log-det (many trace terms)

## Advanced Implementation Techniques

### 1. Multi-Scale Architecture (RealNVP, Glow)

**Idea**: Process different scales (coarse-to-fine)

```python
class MultiScaleFlow(nn.Module):
    def __init__(self, dims=[32, 16, 8, 4]):
        self.scales = nn.ModuleList()
        for dim in dims:
            # Flow at this scale
            flow = StackedCouplings(dim, num_layers=4)
            self.scales.append(flow)

    def forward(self, x):
        log_det = 0
        z_out = []

        for i, flow in enumerate(self.scales):
            # Flow transformation
            x, ld = flow(x)
            log_det += ld

            if i < len(self.scales) - 1:
                # Split: half to output, half to next scale
                x, z_split = x.chunk(2, dim=-1)
                z_out.append(z_split)

        z_out.append(x)  # Final latent
        z_full = torch.cat(z_out, dim=-1)

        return z_full, log_det
```

**Benefits:**
- Captures multi-scale structure
- Reduces memory (smaller activations at later scales)
- Better for images/hierarchical data

### 2. Data-Dependent Initialization

**Actnorm Initialization:**
```python
class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x, reverse=False):
        if not self.initialized and not reverse:
            # Initialize from first batch
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0) + 1e-6
                self.bias.copy_(-mean)
                self.scale.copy_(torch.log(1.0 / std))
            self.initialized = True

        if not reverse:
            y = (x + self.bias) * torch.exp(self.scale)
            log_det = self.scale.sum()
        else:
            y = x * torch.exp(-self.scale) - self.bias
            log_det = -self.scale.sum()

        return y, log_det.expand(x.shape[0])
```

**Why it helps:**
- Normalizes activations initially
- Prevents saturation in early training
- Improves gradient flow

### 3. Dequantization (for discrete data)

**Problem**: Discrete data (images, text) have zero probability under continuous flows.

**Solution**: Add uniform noise (variational dequantization)
```python
def dequantize(x, num_bits=8):
    """
    Add uniform noise to discrete data.

    Args:
        x: Discrete data in [0, 2^num_bits - 1]
    Returns:
        x_cont: Continuous data in [0, 2^num_bits)
    """
    # Add U(0,1) noise
    x_cont = x + torch.rand_like(x)
    # Normalize to [0, 1]
    x_cont = x_cont / (2 ** num_bits)
    return x_cont

# Adjust log-likelihood:
# log p(x_discrete) = log ∫ p(x_cont) dx_cont
#                   ≈ log p(x_cont + u) - num_dims * log(2^num_bits)
```

### 4. Preventing Checkerboard Artifacts

**Problem**: Alternating masks create checkerboard patterns in images.

**Solutions:**
1. **Random channel permutation** between layers
2. **Learnable 1x1 convolution** (Glow)
3. **Different mask patterns** (channel-wise, spatial)

```python
def shuffle_channels(x):
    """Random channel permutation."""
    batch, channels = x.shape[:2]
    perm = torch.randperm(channels)
    return x[:, perm, ...]
```

## Training Normalizing Flows

### Maximum Likelihood Objective

**Goal**: Maximize log p(x) = log p(f(x)) + log |det J|

```python
def train_step(flow, x, optimizer):
    optimizer.zero_grad()

    # Forward pass
    z, log_det = flow(x, reverse=False)

    # Log-probability under base distribution
    log_p_z = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=-1)

    # Total log-likelihood
    log_p_x = log_p_z + log_det

    # Negative log-likelihood loss
    loss = -log_p_x.mean()

    loss.backward()
    optimizer.step()

    return loss.item()
```

### Flows in VAEs

**Normalizing Flow VAE (Rezende & Mohamed, 2015):**

**Encoder**: q(z|x) with flow q(z_K|x) = q(z_0|x) |det J|
```python
# Encode to z_0
z_0, log_q_0 = encoder(x)

# Apply flow
z_K = z_0
log_det_sum = 0
for flow in flows:
    z_K, log_det = flow(z_K)
    log_det_sum += log_det

# Posterior log-prob
log_q_K = log_q_0 - log_det_sum  # Note: minus sign!
```

**ELBO with flows:**
```
ELBO = E_q[log p(x|z_K)] - KL(q(z_K|x) || p(z_K))
     = E_q[log p(x|z_K) + log p(z_K) - log q(z_K|x)]
```

**Benefits:**
- More expressive posterior (multimodal, heavy tails)
- Tighter ELBO
- Better latent representations

## Debugging Normalizing Flows

### 1. Invertibility Test

**Critical**: Verify f(f⁻¹(x)) ≈ x

```python
def test_invertibility(flow, x, tol=1e-4):
    """Test if flow is invertible."""
    # Forward then inverse
    z, _ = flow(x, reverse=False)
    x_recon, _ = flow(z, reverse=True)

    # Reconstruction error
    error = (x - x_recon).abs().max().item()

    print(f"Max reconstruction error: {error:.2e}")
    assert error < tol, f"Flow not invertible! Error: {error}"

    return error
```

### 2. Log-Det Magnitude Check

**Typical range**: log |det| should be O(10), not O(1000)

```python
def check_log_det(flow, x):
    z, log_det = flow(x)

    print(f"log |det| range: [{log_det.min():.2f}, {log_det.max():.2f}]")
    print(f"log |det| mean: {log_det.mean():.2f} ± {log_det.std():.2f}")

    if log_det.abs().max() > 100:
        print("⚠️  WARNING: Very large log-det magnitudes!")
        print("    Check scale parameter bounds.")
```

### 3. Analytical Test Cases

**Test on known distributions:**

```python
def test_gaussian_to_gaussian():
    """Test flow on simple Gaussian."""
    flow = SimpleFlow(dim=2)

    # Sample from N(0,1)
    x = torch.randn(1000, 2)

    # Transform
    z, log_det = flow(x)

    # Check if z is also Gaussian
    # (Should be if flow is good)
    mean = z.mean(dim=0)
    std = z.std(dim=0)

    print(f"z mean: {mean}")  # Should be ≈ 0
    print(f"z std: {std}")    # Should be ≈ 1
```

## Common Pitfalls

### Pitfall 1: Unbounded Scale Parameters

**Symptom**: log |det| → ±∞, loss explodes

**Fix**: Always bound scale parameters:
```python
# Bad:
s = scale_net(x)  # Can be arbitrary

# Good:
s = torch.tanh(scale_net(x) / 3.0) * 3.0  # s ∈ (-3, 3)
# Or:
s = torch.clamp(scale_net(x), min=-5, max=5)
```

### Pitfall 2: Forgot to Alternate Masks

**Symptom**: Some dimensions never transformed

**Fix**: Ensure masks alternate:
```python
for i in range(num_layers):
    mask = make_mask(dim, parity=i % 2)  # Alternate 0, 1, 0, 1, ...
    layers.append(CouplingLayer(mask))
```

### Pitfall 3: Incorrect Inverse Log-Det Sign

**Symptom**: log p(x) is incorrect

**Fix**: Remember inverse has opposite log-det:
```python
if reverse:
    log_det = -log_det  # Inverse transform
```

### Pitfall 4: Not Testing Invertibility

**Symptom**: Silent errors, poor samples

**Fix**: Always test reconstruction:
```python
x_recon = flow.inverse(flow.forward(x))
assert torch.allclose(x, x_recon, atol=1e-4)
```

## Literature & Resources

1. **NICE**: Dinh et al., "NICE: Non-linear Independent Components Estimation", ICLR 2015
2. **RealNVP**: Dinh et al., "Density estimation using Real NVP", ICLR 2017
3. **Glow**: Kingma & Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions", NeurIPS 2018
4. **MAF**: Papamakarios et al., "Masked Autoregressive Flow for Density Estimation", NeurIPS 2017
5. **IAF**: Kingma et al., "Improving Variational Inference with Inverse Autoregressive Flow", NeurIPS 2016
6. **Neural Spline Flows**: Durkan et al., "Neural Spline Flows", NeurIPS 2019
7. **FFJORD**: Grathwohl et al., "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models", ICLR 2019
8. **Residual Flows**: Chen et al., "Residual Flows for Invertible Generative Modeling", NeurIPS 2019
9. **Flow VAEs**: Rezende & Mohamed, "Variational Inference with Normalizing Flows", ICML 2015

## Quick Reference

| Architecture | Log-Det Cost | Sampling Speed | Density Speed | Expressivity |
|--------------|-------------|----------------|---------------|--------------|
| NICE (additive) | O(1) - zero | Fast (parallel) | Fast (parallel) | Low |
| RealNVP (affine) | O(D) - sum | Fast (parallel) | Fast (parallel) | Medium |
| Glow | O(D) | Fast (parallel) | Fast (parallel) | Medium-High |
| MAF | O(D) | Slow (sequential) | Fast (parallel) | High |
| IAF | O(D) | Fast (parallel) | Slow (sequential) | High |
| Neural Spline | O(D) | Medium | Medium | Very High |
| FFJORD | O(D²) - trace est | Slow (ODE solve) | Slow (ODE solve) | Very High |

**When to use:**
- **RealNVP/Glow**: General purpose, good balance
- **MAF**: Need high-quality density (inference, anomaly detection)
- **IAF**: Need fast sampling (VAE decoder, generation)
- **Spline Flows**: Maximum expressivity per layer
- **FFJORD**: Research, maximum flexibility
