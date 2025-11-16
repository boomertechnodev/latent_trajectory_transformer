---
name: ode-sde-dynamics
description: Deep technical knowledge for ordinary and stochastic differential equations in deep learning, including advanced numerical methods, stability theory, Itô calculus, adjoint methods, and PyTorch implementation patterns for neural ODEs/SDEs.
---

# ODE/SDE Dynamics - Deep Technical Knowledge

## Mathematical Foundations

### Ordinary Differential Equations

**Initial Value Problem (IVP):**
```
dz/dt = f(z, t),  z(t₀) = z₀
```

**Existence and Uniqueness (Picard-Lindelöf Theorem):**
- If f is Lipschitz continuous in z: |f(z₁,t) - f(z₂,t)| ≤ L|z₁ - z₂|
- And continuous in t
- Then unique solution exists locally

**Stability Analysis:**
- **Lyapunov Stability**: If V(z) is a Lyapunov function (V̇ ≤ 0), then z* is stable
- **Exponential Stability**: If dV/dt ≤ -αV for α > 0, then exponential convergence
- **Linearization**: Eigenvalues of Jacobian ∂f/∂z at fixed point determine local stability

### Stochastic Differential Equations

**Itô SDE:**
```
dZ_t = f(Z_t, t)dt + σ(Z_t, t)dW_t
```
where W_t is a Wiener process (Brownian motion).

**Itô's Lemma:**
For Y_t = g(Z_t, t):
```
dY_t = (∂g/∂t + f·∇g + (1/2)σ²·∇²g)dt + (σ·∇g)dW_t
```

**Fokker-Planck Equation:**
Evolution of probability density p(z,t):
```
∂p/∂t = -∇·(fp) + (1/2)∇²(σ²p)
```

**Stratonovich vs Itô:**
- **Itô**: Non-anticipating, easier for theory, requires correction term
- **Stratonovich**: Symmetric, chain rule works normally, converts to Itô with drift correction

## Numerical Methods Deep Dive

### ODE Solvers

**1. Euler Method (1st Order)**
```python
z_{n+1} = z_n + h·f(z_n, t_n)
```
- Local error: O(h²)
- Global error: O(h)
- Stability region: |1 + hλ| < 1 (λ = eigenvalue of Jacobian)
- Use when: Simplicity is key, f is smooth

**2. Runge-Kutta 4 (4th Order)**
```python
k1 = f(z_n, t_n)
k2 = f(z_n + h/2·k1, t_n + h/2)
k3 = f(z_n + h/2·k2, t_n + h/2)
k4 = f(z_n + h·k3, t_n + h)
z_{n+1} = z_n + h/6·(k1 + 2k2 + 2k3 + k4)
```
- Local error: O(h⁵)
- Global error: O(h⁴)
- 4 function evaluations per step
- Use when: High accuracy needed, f is expensive

**3. Adaptive Step Size (RK45)**
- Estimates local error by comparing 4th and 5th order solutions
- Adjusts h to keep error below tolerance
- Implemented in scipy.integrate.solve_ivp, torchdiffeq
- Use when: Unknown solution behavior, efficiency critical

**4. Implicit Methods (for stiff ODEs)**
- Backward Euler: z_{n+1} = z_n + h·f(z_{n+1}, t_{n+1})
- Requires solving nonlinear equation at each step
- A-stable (stable for all h > 0 when Re(λ) < 0)
- Use when: Very stiff systems (large |λ|)

### SDE Solvers

**1. Euler-Maruyama (Order 0.5)**
```python
Z_{n+1} = Z_n + f(Z_n, t_n)·Δt + σ(Z_n, t_n)·√Δt·N(0,1)
```
- Strong order 0.5, weak order 1
- Simplest SDE method
- Requires: Bounded f and σ

**2. Milstein Method (Order 1.0)**
```python
Z_{n+1} = Z_n + f·Δt + σ·√Δt·ΔW + (1/2)σ(∂σ/∂z)·((ΔW)² - Δt)
```
- Includes second-order correction term
- Strong order 1.0
- Requires: ∂σ/∂z computable

**3. Stochastic Runge-Kutta**
- Multiple stages like RK4 but with random variables
- Higher order but more complex
- Use when: High accuracy needed, smooth σ

## PyTorch Implementation Patterns

### Adjoint Method for Memory Efficiency

**Problem**: Backprop through ODE solver requires storing entire trajectory (O(num_steps) memory)

**Solution**: Adjoint sensitivity method

```python
# Instead of backprop through solver:
# z(t1) = ODESolve(f, z0, t0, t1)
# loss = L(z(t1))
# Compute ∂L/∂θ by solving adjoint ODE backwards:

def adjoint_dynamics(aug_state, t):
    z, a, _, _ = aug_state  # z=state, a=adjoint
    with torch.enable_grad():
        z = z.requires_grad_(True)
        t = t.requires_grad_(True)
        f_eval = f(z, t, theta)

    # Adjoint ODE: da/dt = -a^T·∂f/∂z
    a_new = -torch.autograd.grad(f_eval, z, a, allow_unused=True)[0]

    # Parameter gradients: da_θ/dt = -a^T·∂f/∂θ
    a_theta = -torch.autograd.grad(f_eval, theta, a, allow_unused=True)[0]

    return (f_eval, a_new, torch.zeros_like(t), a_theta)

# Solve backwards from t1 to t0
aug_state = (z1, adj_z1, torch.zeros(1), torch.zeros_like(theta))
z0, adj_z0, _, adj_theta = ode_solve(adjoint_dynamics, aug_state, t1, t0)
```

**Benefits**:
- Memory: O(1) instead of O(num_steps)
- Trade computation for memory
- Essential for long trajectories

### Continuous Normalizing Flows

**Instantaneous Change of Variables:**
```python
def cnf_forward(z0, t_span):
    # Solve augmented ODE: [z, log_det_jac]
    def dynamics(state, t):
        z = state[0]
        f_z = drift(z, t)

        # Trace of Jacobian: div(f) = tr(∂f/∂z)
        # Hutchinson trace estimator (unbiased):
        eps = torch.randn_like(z)
        with torch.enable_grad():
            z.requires_grad_(True)
            f_eps = drift(z, t)
            grad = torch.autograd.grad(f_eps, z, eps, create_graph=True)[0]
        trace = (grad * eps).sum(dim=-1, keepdim=True)

        return torch.cat([f_z, -trace], dim=-1)

    z0_aug = torch.cat([z0, torch.zeros(z0.shape[0], 1)], dim=-1)
    z1_aug = ode_solve(dynamics, z0_aug, t_span)
    z1, log_det = z1_aug[:, :-1], z1_aug[:, -1]
    return z1, log_det
```

### Numerical Stability Tricks

**1. Log-Variance Parameterization for Diffusion:**
```python
# Instead of: σ = net(z, t)  # Can go negative!
log_sigma = net(z, t)
log_sigma = torch.clamp(log_sigma, min=math.log(1e-4), max=math.log(1.0))
sigma = torch.exp(log_sigma)  # Always positive, bounded
```

**2. Gradient Clipping for Drift:**
```python
# In training loop:
loss.backward()
torch.nn.utils.clip_grad_norm_(drift_net.parameters(), max_norm=1.0)
optimizer.step()
```

**3. Spectral Normalization:**
```python
# Constrain Lipschitz constant of drift network
from torch.nn.utils import spectral_norm
drift_layer = spectral_norm(nn.Linear(dim, dim))
```

**4. Time Rescaling:**
```python
# Instead of t in [0, T], use τ = t/T in [0, 1]
# Improves numerical conditioning
def drift(z, tau, T=1.0):
    t = tau * T
    # ... compute drift at time t
    return f / T  # Rescale by 1/T
```

## Common Pitfalls and Debugging

### Pitfall 1: Exploding Trajectories

**Symptom**: z_norm grows to infinity, NaN losses

**Diagnosis**:
```python
# Monitor trajectory norms
norms = torch.norm(z_traj, dim=-1)  # (batch, num_steps)
print(f"Max norm: {norms.max():.2f}, Mean: {norms.mean():.2f}")
```

**Fixes**:
- Reduce initialization scale: `gain=0.01` instead of `1.0`
- Add gradient clipping: `clip_grad_norm_(params, 1.0)`
- Reduce step size: `dt=0.01` instead of `0.1`
- Add Lipschitz constraint: spectral normalization
- Check drift network outputs: `print(drift.abs().max())`

### Pitfall 2: Stiff ODEs (Eigenvalues with Large |λ|)

**Symptom**: Requires very small dt, otherwise unstable

**Diagnosis**:
```python
# Compute Jacobian eigenvalues
z = torch.randn(1, latent_dim, requires_grad=True)
f = drift(z, t)
jacobian = torch.autograd.functional.jacobian(lambda z: drift(z, t), z)
eigenvalues = torch.linalg.eigvals(jacobian[0, :, 0, :])
print(f"Max |λ|: {eigenvalues.abs().max():.2e}")
```

**Fixes**:
- Use implicit solver (backward Euler)
- Add diffusion (SDE smooths dynamics)
- Reduce network depth
- Add skip connections in drift network

### Pitfall 3: SDE Diffusion Too Large or Too Small

**Symptom**:
- Too large: Random walk, no learning
- Too small: Deterministic, no exploration

**Diagnosis**:
```python
drift, sigma = sde(z, t)
print(f"σ range: [{sigma.min():.3f}, {sigma.max():.3f}]")
print(f"SNR: {(drift.norm() / sigma.norm()):.2f}")  # Signal-to-noise ratio
```

**Fixes**:
- Enforce bounds: `sigma_min=1e-4, sigma_max=1.0`
- Adjust SNR: increase drift scale or decrease diffusion scale
- Learned time-dependent diffusion schedule

### Pitfall 4: Incorrect Tensor Broadcasting

**Symptom**: Shape errors when time t is scalar

**Fix**:
```python
def drift(z, t):
    # z: (batch, latent_dim)
    # t: scalar or (batch, 1)

    # ALWAYS ensure t has batch dimension:
    if t.ndim == 0:  # scalar
        t = t.reshape(1, 1).expand(z.shape[0], 1)
    elif t.ndim == 1:  # (batch,)
        t = t.unsqueeze(-1)  # (batch, 1)

    zt = torch.cat([z, t], dim=-1)  # (batch, latent_dim+1)
    return net(zt)
```

## Advanced Topics

### 1. Augmented Neural ODEs

**Problem**: Standard ODE dz/dt = f(z,t) may not be expressive enough

**Solution**: Add auxiliary dimensions
```python
# Augment z with learnable dimensions
z_aug = torch.cat([z, torch.zeros(batch, n_aug)], dim=-1)
# Solve ODE in augmented space
z_aug_final = ode_solve(drift_aug, z_aug, t_span)
# Extract original dimensions
z_final = z_aug_final[:, :latent_dim]
```

**Theory**: Proved to be universal approximators (covers all continuous maps)

### 2. Second-Order ODEs (Hamiltonian)

**Hamiltonian Mechanics:**
```
dq/dt = ∂H/∂p
dp/dt = -∂H/∂q
```

**Symplectic Integrator (Leapfrog):**
```python
def leapfrog_step(q, p, H, dt):
    # Half step for p
    p_half = p - (dt/2) * grad_q(H, q)
    # Full step for q
    q_new = q + dt * grad_p(H, p_half)
    # Half step for p
    p_new = p_half - (dt/2) * grad_q(H, q_new)
    return q_new, p_new
```

**Benefits**: Energy conservation, reversibility, long-term stability

### 3. Score-Based Diffusion Models

**Connection to SDEs:**
Reverse-time SDE for denoising diffusion:
```
dz = [f(z,t) - g²(t)∇_z log p_t(z)]dt + g(t)dW̄
```
where ∇_z log p_t(z) is the score function.

**Training**: Learn score network s_θ(z,t) ≈ ∇_z log p_t(z) via denoising score matching

### 4. Adaptive Step Size Implementation

```python
def adaptive_euler(f, z0, t_span, tol=1e-3):
    z = z0
    t = t_span[0]
    dt = (t_span[1] - t_span[0]) / 10  # Initial guess

    trajectory = [z0]
    times = [t]

    while t < t_span[-1]:
        # Two steps with dt
        z1 = z + dt * f(z, t)
        z2 = z1 + dt * f(z1, t + dt)

        # One step with 2*dt
        z_big = z + 2*dt * f(z, t)

        # Error estimate
        error = torch.norm(z2 - z_big) / torch.norm(z2 + 1e-8)

        if error < tol:
            # Accept step
            z = z2
            t = t + 2*dt
            trajectory.append(z)
            times.append(t)

            # Increase dt
            dt = dt * 1.5
        else:
            # Reject step, decrease dt
            dt = dt * 0.5

    return torch.stack(trajectory)
```

## Literature References

1. **Neural ODEs**: Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
2. **Augmented NODEs**: Dupont et al., "Augmented Neural ODEs", NeurIPS 2019
3. **Adjoint Sensitivity**: Pontryagin et al., "Mathematical Theory of Optimal Processes", 1962
4. **Score SDEs**: Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021
5. **Itô Calculus**: Øksendal, "Stochastic Differential Equations", Springer
6. **Numerical Methods**: Hairer et al., "Solving Ordinary Differential Equations I & II", Springer
7. **Stability Theory**: Khalil, "Nonlinear Systems", Prentice Hall
8. **Continuous Normalizing Flows**: Grathwohl et al., "FFJORD", ICLR 2019

## Quick Reference: Method Selection

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| Smooth ODE, high accuracy | RK4 | 4th order, stable |
| Smooth ODE, efficiency | Adaptive RK45 | Auto step size |
| Stiff ODE | Implicit Euler | A-stable |
| Simple ODE, fast | Euler | 1st order, simple |
| Smooth SDE | Milstein | 2nd order correction |
| Simple SDE | Euler-Maruyama | Easy to implement |
| Memory constraint | Adjoint method | O(1) memory |
| Physics-informed | Symplectic | Energy conservation |
| Long trajectories | Adaptive + adjoint | Efficiency + memory |

## Code Checklist for Production

- [ ] Small initialization (gain ≤ 0.1)
- [ ] Gradient clipping enabled
- [ ] Diffusion bounds enforced (σ_min, σ_max)
- [ ] Time tensor broadcasting handled correctly
- [ ] Step size justified (not too large)
- [ ] Trajectories monitored for boundedness
- [ ] Unit tests with analytical solutions
- [ ] Stability analysis performed (eigenvalues)
- [ ] Adjoint method for long sequences
- [ ] Visualizations for phase space debugging
