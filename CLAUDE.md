# CLAUDE.md - AI Assistant Guide for Latent Trajectory Flow Transformer

## Project Overview

This repository implements a **Latent Trajectory Flow Transformer** - a novel architecture combining discrete/text transformers with time-dependent latent spaces suitable for planning in trajectory space. The key innovation is using a latent ODE (or SDE) to evolve latent representations over time, enabling trajectory rollouts without growing the context length of the backbone transformer.

### Research Context
- **Purpose**: Continual learning and inference-time planning
- **Key Innovation**: Latent trajectory evolution that maintains constant context length
- **Status**: Experimental/research code with working proof-of-concept
- **Branch**: **feynman** - Extended with Raccoon-in-a-Bungeecord continuous learning system

## Repository Structure

```
latent_trajectory_transformer/
├── README.md                      # Brief project description with example outputs
├── latent_drift_trajectory.py     # Single-file implementation (1824 lines)
└── CLAUDE.md                      # This file
```

This is a **single-file research implementation** - all code is contained in `latent_drift_trajectory.py`.

## Major Components

### Part 1: Original Latent ODE Model (Lines 1-905)
Classic latent trajectory transformer with deterministic ODE dynamics for character-level sequence modeling.

### Part 2: Raccoon-in-a-Bungeecord (Lines 912-1823)
Advanced continuous learning architecture with:
- **Stochastic Differential Equations (SDE)** with drift + diffusion
- **Normalizing Flows** (invertible transformations)
- **Experience Replay Memory** with priority sampling
- **Log Classification Task** with concept drift
- **Online Adaptation** capability

## Code Architecture - Part 1: Latent ODE Model

### Component Breakdown (Lines 1-905)

1. **Character-Level Dataset** (lines 14-64)
   - Vocabulary: `_`, `A-Z`, `!`, `>`, `?` (29 tokens total)
   - `SyntheticTargetDataset`: Generates 64-char sequences with 8-char letter blocks + noise
   - Format: `?{LETTER}>{64-char sequence with LETTER block}`

2. **Statistical Testing Infrastructure** (lines 66-101)
   - Distributed training support (DDP hooks)
   - `all_reduce` for tensor synchronization
   - `UnivariateTest`: Base class for statistical tests

3. **Normality Tests** (lines 103-288)
   - `FastEppsPulley`: **IMPROVED** - Fixed weight calculation (lines 120-134)
     - Properly separates quadrature weights from weight function
     - Uses trapezoid integration correctly
   - `EppsPulleyCF`: Alternative EP test implementation
   - `SlicingUnivariateTest`: Multivariate testing via random projections
   - **Purpose**: Regularize latent space to be approximately Gaussian

4. **ODE Dynamics** (lines 290-323)
   - `ODE`: Abstract base class for ordinary differential equations
   - `solve_ode`: Simple Euler integration solver
   - **Key**: Enables continuous-time evolution of discrete latent states

5. **Prior/Generative Model** (lines 325-368)
   - `PriorInitDistribution`: Learned initial latent distribution (mean + log-scale)
   - `PriorODE`: **IMPROVED** - Shallower network (lines 343-371)
     - Now 5 layers instead of 11 (configurable with `depth` parameter)
     - Better initialization: `xavier_uniform` with gain=0.1
     - Orthogonal init for output layer
     - Added LayerNorm before final layer
     - **Benefit**: Better gradient flow and numerical stability

6. **Observation/Decoder Model** (lines 370-441)
   - `DiscreteObservation`: Autoregressive transformer decoder
   - Predicts p(x_t | x_{<t}, z_{0:t}) - tokens given latent path
   - Single causal transformer block with 4 heads

7. **Encoder Architecture** (lines 443-543)
   - `PosteriorEncoder`: 4-block bidirectional transformer encoder
   - `DeterministicEncoder`: Wraps encoder with projection to latent space
   - Optional Gaussian smoothing kernel (commented out at line 540)

8. **Transformer Building Blocks** (lines 545-649)
   - `AddPositionalEncoding`: Sinusoidal positional encoding
   - `QKVAttention`: Multi-head attention implementation (supports causal mode)
   - `TransformerBlock`: Standard transformer block (attention + FFN)
   - `Predictor`: Simple MLP for latent state prediction

9. **Main ODE Model** (lines 650-776)
   - `DeterministicLatentODE`: Combines all components
   - **Three loss terms**:
     1. Reconstruction loss (negative log-likelihood)
     2. Latent regularization (Epps-Pulley normality test) - **FIXED** at line 747
     3. ODE matching loss (Euler step consistency)

10. **Sampling** (lines 778-829)
    - `sample_sequences_ode`: Generate sequences by:
      - Sampling z0 ~ N(0,I)
      - Rolling out ODE to get z_path
      - Autoregressively decoding tokens from latent path

11. **Training Loop** (lines 831-905)
    - `train_ode`: AdamW optimizer with learning rate 1e-3
    - Loss weight warmup for EP term (0.0005 → final over 10k steps)
    - Periodic sampling every 100 steps

## Code Architecture - Part 2: Raccoon-in-a-Bungeecord

### Component Breakdown (Lines 912-1823)

12. **Time Embedding** (lines 917-953)
    - `TimeAwareTransform`: Multi-scale time embedding
    - Uses exponentially-spaced frequency bands (1 Hz to 1000 Hz)
    - Provides richer temporal representation than simple sinusoidal encoding
    - Output: sin/cos pairs for rotation-invariant features

13. **SDE Dynamics** (lines 955-1015)
    - `RaccoonDynamics`: **Stochastic** differential equation dynamics
    - Implements: `dz = drift(z,t)*dt + diffusion(z,t)*dW`
    - **FIXED**: Uses log-variance for numerical stability (lines 977-1012)
    - Bounded diffusion: sigma_min=1e-4, sigma_max=1.0
    - Separate networks for drift and diffusion

14. **SDE Solver** (lines 1017-1053)
    - `solve_sde`: Euler-Maruyama integration method
    - **FIXED**: Proper tensor broadcasting for time (line 1041)
    - Returns full trajectory: (batch, num_steps, latent_dim)

15. **Normalizing Flows** (lines 1055-1164)
    - `CouplingLayer`: Affine coupling transformation
      - **IMPROVED**: Parameterized `time_dim` and `scale_range` (lines 1062-1063)
      - Invertible transformation with log-det Jacobian
      - Time-conditioned via TimeAwareTransform
    - `RaccoonFlow`: Stack of coupling layers
      - 4 layers with alternating masks
      - **FIXED**: Passes `time_dim` consistently (line 1131)
      - Supports forward and reverse modes

16. **Experience Replay Memory** (lines 1166-1253)
    - `RaccoonMemory`: Priority-based experience buffer
    - **FIXED**: Multiple improvements
      - Uses numpy for efficient min-finding (line 1192-1196)
      - Robust score normalization with softmax trick (lines 1219-1225)
      - Handles replacement when buffer smaller than sample size (lines 1212-1214)
      - Supports dict format from fixed `continuous_update` (line 1234)
    - **NEW**: Added state_dict methods for checkpointing (lines 1240-1252)

17. **Log Classification Task** (lines 1256-1335)
    - Extended vocabulary: original chars + digits 0-9
    - `LogDataset`: Synthetic log message generation
      - 4 categories: ERROR, WARNING, INFO, DEBUG
      - Supports concept drift at specified sample index
      - 10% character corruption noise
    - Templates for realistic log messages

18. **Raccoon Log Classifier** (lines 1337-1566)
    - `RaccoonLogClassifier`: Main continuous learning model
    - Components:
      - Encoder: tokens → latent distribution (mean + logvar)
      - SDE dynamics: `RaccoonDynamics`
      - Normalizing flows: `RaccoonFlow`
      - Classifier head: latent → class logits
      - EP regularization: `SlicingUnivariateTest`
      - Experience memory: `RaccoonMemory`
    - **FIXED**: Empty batch guard (lines 1444-1451)
    - **FIXED**: Corrected KL divergence formula (lines 1473-1483)
    - **FIXED**: Proper gradient management in `continuous_update` (lines 1553-1565)
      - Creates SGD optimizer for online adaptation
      - Stores data as dict structure for memory efficiency

19. **Training Loops** (lines 1568-1666)
    - `train_raccoon_classifier`: Phase 1 supervised training
      - AdamW optimizer, 1e-3 learning rate
      - Tracks accuracy, classification loss, KL loss
    - `continuous_learning_phase`: Phase 2 online adaptation
      - Single-sample updates with memory replay
      - Tracks rolling 100-sample accuracy
      - Shows memory buffer growth

20. **Main Entry Point** (lines 1668-1823)
    - **Device**: Forced to CPU for compatibility (line 1670)
    - **Optional**: Run original ODE model (lines 1675-1714)
    - **Main**: Raccoon log classifier pipeline (lines 1717-1823)
      - Phase 1: 1000 steps initial training
      - Test evaluation
      - Phase 2: 1000 samples continuous learning
      - Final evaluation
      - Detailed progress reporting

## Key Architectural Decisions

### 1. ODE vs SDE Dynamics
- **ODE Model**: Deterministic drift only, simpler but less expressive
- **Raccoon Model**: SDE with drift + diffusion, captures stochasticity

### 2. Normalizing Flows for Expressiveness
- Invertible transformations increase latent space flexibility
- Time-conditioned coupling layers
- Enables exact log-likelihood computation

### 3. Experience Replay for Continual Learning
- Prevents catastrophic forgetting
- Priority sampling biases toward high-quality experiences
- Small online updates + memory replay

### 4. Multi-Task Design
- Original: Character sequence modeling (generative)
- Raccoon: Log classification (discriminative + continual learning)

## Improvements in Feynman Branch

### Fixed Bugs
1. **FastEppsPulley weight calculation** (lines 120-134)
   - Separated quadrature weights from weight function
   - Correct trapezoid integration

2. **PriorODE architecture** (lines 343-371)
   - Reduced depth from 11 to 5 layers
   - Better initialization (gain=0.1, orthogonal output)
   - Added normalization before final layer

3. **Latent test in loss_components** (line 747)
   - Uses reshaped tensor correctly for z_pred

4. **RaccoonDynamics diffusion** (lines 1006-1012)
   - Log-variance parameterization
   - Clamped bounds for numerical stability

5. **solve_sde tensor broadcasting** (line 1041)
   - Proper expansion from 0-d scalar time

6. **CouplingLayer parameterization** (lines 1062-1063, 1131)
   - Configurable time_dim throughout
   - Configurable scale_range

7. **RaccoonMemory robustness** (lines 1190-1232)
   - Numpy for efficiency
   - Softmax trick for stable score normalization
   - Handles replacement correctly

8. **RaccoonLogClassifier improvements**
   - Empty batch guard (lines 1444-1451)
   - Corrected KL formula (lines 1473-1483)
   - Proper continuous update with dict storage (lines 1526-1533)
   - SGD optimizer for online adaptation (lines 1554-1565)

### New Features
1. **Multi-scale time embedding** (`TimeAwareTransform`)
2. **Stochastic dynamics** (`RaccoonDynamics`, `solve_sde`)
3. **Normalizing flows** (`CouplingLayer`, `RaccoonFlow`)
4. **Experience replay** (`RaccoonMemory` with state_dict support)
5. **Log classification task** (`LogDataset`, `RaccoonLogClassifier`)
6. **Two-phase training** (initial + continuous learning)
7. **Concept drift simulation** (in LogDataset)

## Development Workflows

### Running Original ODE Model
```bash
# Edit line 1675 to set run_original_ode = True
python latent_drift_trajectory.py
```

### Running Raccoon Classifier (Default)
```bash
python latent_drift_trajectory.py
```

Output:
- Phase 1: 1000 steps supervised training with progress bar
- Test evaluation after Phase 1
- Phase 2: 1000 samples continuous learning
- Final test evaluation
- Memory buffer statistics

### Device Selection
Currently **forced to CPU** (line 1670) for compatibility. To change:
```python
# Original auto-detection (commented out):
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Monitoring Training

**Phase 1 (Initial Training)**:
- Progress bar shows: loss, accuracy, class_loss, kl_loss
- Expect accuracy to increase, losses to decrease

**Phase 2 (Continuous Learning)**:
- Progress bar shows: memory buffer size, rolling 100-sample accuracy
- Watch for stable accuracy despite concept drift

## Key Code Patterns & Conventions

### 1. Tensor Shapes (Critical!)

```python
# Original ODE Model:
# B = batch size, L = sequence length, D = latent dimension
# H = hidden dimension, E = embedding dimension, V = vocabulary size

tokens:     (B, L)           # Discrete tokens
z:          (B, L, D)        # Latent trajectories
embeddings: (B, L, E)        # Token embeddings
hidden:     (B, L, H)        # Transformer hidden states
logits:     (B, L, V)        # Decoder output logits

# Raccoon Model:
tokens:     (B, seq_len)     # Log message tokens
z:          (B, latent_dim)  # Latent state (not sequential)
z_traj:     (B, steps, latent_dim)  # SDE trajectory
logits:     (B, num_classes) # Classification logits
labels:     (B,)             # Class labels
```

### 2. Time Conventions

**ODE Model**:
- Time t ∈ [0, 1] for sequence length
- Normalized: `t_grid = torch.linspace(0.0, 1.0, L, ...)`

**Raccoon Model**:
- Short trajectories: t ∈ [0, 0.1] for dynamics (line 1458)
- Flow evaluation at t=0.5 (line 1463)
- Time embedding: frequencies 1-1000 Hz (lines 926-931)

### 3. Memory Management

**Experience Replay**:
```python
# Storage format (lines 1529-1533):
memory_item = {
    'tokens': tokens[i:i+1].detach().cpu(),  # Keep batch dim
    'label': labels[i:i+1].detach().cpu()
}
# Always store on CPU to save GPU memory
```

### 4. Gradient Flow

**ODE Model**:
- `.detach()` on encoder outputs for ODE matching (line 725)

**Raccoon Model**:
- Separate optimizer for online adaptation (lines 1554-1559)
- No gradient accumulation, single-step updates

### 5. Initialization Patterns

**Original ODE** (lines 1694-1701):
```python
nn.init.trunc_normal_(weight, std=0.02)
nn.init.zeros_(bias)
```

**Raccoon SDE** (lines 985-989):
```python
nn.init.xavier_normal_(weight, gain=0.1)  # Small gain for stability
nn.init.zeros_(bias)
```

## Important Variables & Hyperparameters

### Original ODE Model
```python
# Lines 1686-1690
batch_size = 128
seq_len = 64
latent_size = 64
hidden_size = 128
embed_size = 64
num_slices = 1024
train_steps = 1000  # Reduced for testing (was 100k)
```

### Raccoon Classifier
```python
# Lines 1740-1747
vocab_size = log_vocab_size  # 29 + 10 = 39
num_classes = 4  # ERROR, WARNING, INFO, DEBUG
latent_dim = 32
hidden_dim = 64
embed_dim = 32
memory_size = 2000
adaptation_rate = 1e-4

# Training
initial_train_steps = 1000  # Phase 1
continuous_samples = 1000   # Phase 2
batch_size = 32
```

### Loss Weights

**ODE Model** (line 1760):
```python
loss_weights = (1.0, 0.1, 0.01)  # (class, KL, EP)
```

**Raccoon Model** (line 1760):
```python
loss_weights = (1.0, 0.1, 0.01)  # (class, KL, EP)
```

### SDE Parameters
```python
# RaccoonDynamics (lines 963-966)
sigma_min = 1e-4   # Minimum diffusion
sigma_max = 1.0    # Maximum diffusion

# Time span for short trajectory (line 1458)
t_span = torch.linspace(0.0, 0.1, 3)  # 3 steps from 0 to 0.1
```

### Flow Architecture
```python
# RaccoonFlow (line 1372)
num_layers = 4     # Coupling layers
time_dim = 32      # Time embedding dimension
```

## Common Modifications & Extensions

### 1. Switching Between Tasks
```python
# Line 1675: Toggle original ODE model
run_original_ode = True  # or False
```

### 2. Adjusting Memory Size
```python
# Line 1746
memory_size = 5000  # Increase for longer memory
```

### 3. Changing Adaptation Rate
```python
# Line 1747 (constructor parameter)
adaptation_rate = 1e-3  # Faster adaptation

# Or line 1555 (optimizer)
self._adaptation_optimizer = torch.optim.SGD(
    self.parameters(),
    lr=5e-4,  # Different rate
)
```

### 4. Modifying SDE Trajectory Length
```python
# Line 1458
t_span = torch.linspace(0.0, 0.5, 10)  # Longer, more steps
```

### 5. Adding More Coupling Layers
```python
# Line 1372
self.flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=8)  # Was 4
```

### 6. Adjusting Concept Drift
```python
# Line 1725
drift_ds = LogDataset(n_samples=1000, seq_len=50, drift_point=200)
# Drift happens earlier (at sample 200 instead of 500)
```

## Testing & Validation

### Visual Inspection - ODE Model

Good training shows:
1. **Shared Z samples**: Similar structure/patterns
2. **Random Z samples**: Diverse outputs
3. **Target completion**: Coherent letter blocks

Example from README:
```
Samples that share a Z:
?B>!_______________________________________________________________
?B>_!_____________________________BBBBBBBB_________________________
...
```

### Metrics - Raccoon Classifier

**Phase 1 Success Indicators**:
- Test accuracy > 0.7 after 1000 steps
- Classification loss decreasing
- KL loss stable (not exploding)

**Phase 2 Success Indicators**:
- Rolling accuracy stays > 0.6 despite drift
- Memory buffer fills to ~1000+ samples
- No catastrophic forgetting (accuracy doesn't collapse)

### Common Issues

1. **ODE Model**:
   - Loss explosion → Reduce learning rate
   - Poor samples → Check loss weights
   - No diversity → Check prior ODE training

2. **Raccoon Model**:
   - NaN losses → Check diffusion bounds (sigma_min/max)
   - Memory explosion → Reduce memory_size
   - Poor adaptation → Increase adaptation_rate
   - Forgetting → Increase memory_size or reduce adaptation_rate

## Performance Characteristics

### Computational Complexity

**ODE Model**:
- Encoder: O(L² × B × H) - transformer attention
- ODE Matching: O(L × B × D) - drift network forward passes
- EP Test: O(slices × samples × points) ≈ O(1024 × B×L × 17)

**Raccoon Model**:
- Encoder: O(seq_len × B × H) - mean pooling, not full attention
- SDE Solve: O(steps × B × D) - typically 3 steps
- Flow: O(layers × B × D) - 4 coupling layers
- Memory Sampling: O(memory_size) - multinomial sampling

### Memory Usage

**ODE Model**:
- Dominant: Attention matrices O(B × L² × H)
- Batch size 128 with seq_len 64 is feasible

**Raccoon Model**:
- Lighter: No full transformer, mean pooling only
- Memory buffer: O(memory_size × seq_len) on CPU
- Batch size 32 for training, 1 for continuous learning

## Git Workflow

- **Current branch**: `feynman`
- **Commit style**: Descriptive messages with context
- **Push command**: `git push -u origin feynman`

## Dependencies

Required packages:
```python
torch>=2.0.0      # Core deep learning
tqdm              # Progress bars
numpy             # Used in RaccoonMemory for efficiency
```

Recommended: Python 3.9+

## Advanced Topics

### 1. Why SDE Over ODE?

**ODE**: `dz = f(z,t) dt`
- Deterministic evolution
- Single trajectory per initial condition

**SDE**: `dz = f(z,t) dt + σ(z,t) dW`
- Stochastic evolution
- Multiple possible trajectories (epistemic uncertainty)
- Better for continual learning (exploration)

### 2. Normalizing Flows for Density Estimation

Coupling layers provide:
- Exact likelihood: `log p(z) = log p(f(z)) + log|det J|`
- Invertibility: Can map between latent and flow space
- Expressiveness: Increase capacity without changing architecture

### 3. Experience Replay Strategy

Priority sampling helps:
- Retain high-quality examples
- Remove low-confidence samples when buffer full
- Balance exploration (new data) vs exploitation (memory)

### 4. Online Learning Architecture

Two-phase design:
1. **Offline**: Full gradient descent with batches
2. **Online**: Small SGD steps with memory replay

Prevents:
- Catastrophic forgetting
- Overfitting to recent samples
- Loss of generalization

## Useful Line Number References

### Original ODE Model

| Component | Lines | Description |
|-----------|-------|-------------|
| Dataset | 32-64 | Synthetic target generation |
| EP Test | 103-152 | Fast normality test (FIXED) |
| ODE Solver | 306-322 | Euler integration |
| Prior ODE | 343-371 | Drift function (IMPROVED) |
| Encoder | 447-488, 496-542 | Bidirectional transformer |
| Decoder | 373-440 | Causal AR transformer |
| Main Model | 650-775 | Loss computation (FIXED) |
| Training | 837-905 | Optimization loop |
| Sampling | 781-828 | Generation procedure |

### Raccoon Components

| Component | Lines | Description |
|-----------|-------|-------------|
| Time Embed | 917-952 | Multi-scale time features |
| SDE Dynamics | 955-1014 | Drift + diffusion (FIXED) |
| SDE Solver | 1017-1052 | Euler-Maruyama (FIXED) |
| Coupling Layer | 1055-1110 | Affine coupling (IMPROVED) |
| Normalizing Flow | 1113-1163 | Stack of couplings (FIXED) |
| Memory Buffer | 1166-1252 | Experience replay (FIXED + NEW) |
| Log Dataset | 1280-1334 | Synthetic log generation |
| Raccoon Classifier | 1337-1565 | Main model (FIXED) |
| Training Phase 1 | 1572-1614 | Supervised training |
| Training Phase 2 | 1617-1665 | Continuous learning |
| Main Entry | 1668-1823 | Execution logic |

## Summary for AI Assistants

When working with the **feynman branch**:

### Code Organization
1. **Two distinct models** in one file:
   - Lines 1-905: Original latent ODE model
   - Lines 912-1823: Raccoon continuous learning system
2. **Single file** - all modifications go in `latent_drift_trajectory.py`
3. **Many fixes and improvements** - see "Improvements in Feynman Branch" section

### Key Insights
1. **This is research code** - expect experimentation, not production quality
2. **Many components are interdependent** - understand before modifying
3. **Numerical stability matters** - note all the clamping and initialization fixes
4. **Memory management is crucial** - note CPU storage for replay buffer
5. **Two different tasks** - character generation vs log classification

### What's Working
- ✅ Original ODE model with improvements
- ✅ SDE dynamics with proper numerical stability
- ✅ Normalizing flows with invertibility
- ✅ Experience replay with robust sampling
- ✅ Continuous learning with concept drift handling
- ✅ All major bugs fixed

### Most Likely Requests
- Tuning hyperparameters (learning rates, memory size, loss weights)
- Adding new datasets or tasks
- Experimenting with architecture variations
- Adding checkpointing (RaccoonMemory.state_dict already added)
- Adding evaluation metrics beyond accuracy
- Visualizing latent spaces or trajectories

### Avoid Without Discussion
- Removing numerical stability fixes (clamping, initialization)
- Changing memory storage format (dict structure is fixed)
- Breaking time normalization conventions
- Modifying shape conventions inconsistently
- Removing EP regularization
- Disabling experience replay in continuous learning

### Common Pitfalls
1. **Shape mismatches**: ODE uses (B,L,D), Raccoon uses (B,D)
2. **Device errors**: Memory is on CPU, model on GPU/CPU
3. **Gradient issues**: Online learning has separate optimizer
4. **NaN losses**: Check sigma_min/max bounds in SDE
5. **Memory growth**: Buffer grows during Phase 2, this is expected

---

**Last Updated**: 2025-11-16
**Branch**: feynman
**File**: latent_drift_trajectory.py (1824 lines)
**Status**: All known bugs fixed, ready for experimentation
