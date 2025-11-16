# CLAUDE.md - AI Assistant Guide for Latent Trajectory Flow Transformer

## Project Overview

This repository implements a **Latent Trajectory Flow Transformer** - a novel architecture combining discrete/text transformers with time-dependent latent spaces suitable for planning in trajectory space. The key innovation is using a latent ODE (or SDE) to evolve latent representations over time, enabling trajectory rollouts without growing the context length of the backbone transformer.

### Research Context
- **Purpose**: Continual learning and inference-time planning
- **Key Innovation**: Latent trajectory evolution that maintains constant context length
- **Status**: Experimental/research code with working proof-of-concept

## Repository Structure

```
latent_trajectory_transformer/
├── README.md                      # Brief project description with example outputs
├── latent_drift_trajectory.py     # Single-file implementation (all code)
└── CLAUDE.md                      # This file
```

This is a **single-file research implementation** - all code is contained in `latent_drift_trajectory.py`.

## Code Architecture

### High-Level Components (in order of appearance in code)

1. **Character-Level Dataset** (lines 14-64)
   - Vocabulary: `_`, `A-Z`, `!`, `>`, `?` (29 tokens total)
   - `SyntheticTargetDataset`: Generates 64-char sequences with 8-char letter blocks + noise
   - Format: `?{LETTER}>{64-char sequence with LETTER block}`

2. **Statistical Testing Infrastructure** (lines 66-288)
   - `UnivariateTest`: Base class for normality tests
   - `FastEppsPulley`: Fast Epps-Pulley normality test via characteristic functions
   - `EppsPulleyCF`: Alternative EP test implementation
   - `SlicingUnivariateTest`: Multivariate testing via random projections
   - **Purpose**: Regularize latent space to be approximately Gaussian

3. **ODE Dynamics** (lines 290-328)
   - `ODE`: Abstract base class for ordinary differential equations
   - `solve_ode`: Simple Euler integration solver
   - **Key**: Enables continuous-time evolution of discrete latent states

4. **Prior/Generative Model** (lines 330-368)
   - `PriorInitDistribution`: Learned initial latent distribution (mean + log-scale)
   - `PriorODE`: 11-layer MLP with LayerNorm and SiLU that learns drift function f(z,t)

5. **Observation/Decoder Model** (lines 370-440)
   - `DiscreteObservation`: Autoregressive transformer decoder
   - Predicts p(x_t | x_{<t}, z_{0:t}) - tokens given latent path
   - Single causal transformer block with 4 heads

6. **Encoder Architecture** (lines 442-542)
   - `PosteriorEncoder`: 4-block bidirectional transformer encoder
   - `DeterministicEncoder`: Wraps encoder with projection to latent space
   - Optional Gaussian smoothing kernel (commented out at line 540)

7. **Transformer Building Blocks** (lines 544-635)
   - `AddPositionalEncoding`: Sinusoidal positional encoding
   - `QKVAttention`: Multi-head attention implementation (supports causal mode)
   - `TransformerBlock`: Standard transformer block (attention + FFN)

8. **Prediction Head** (lines 637-648)
   - `Predictor`: Simple MLP for latent state prediction

9. **Main Model** (lines 650-775)
   - `DeterministicLatentODE`: Combines all components
   - **Three loss terms**:
     1. Reconstruction loss (negative log-likelihood)
     2. Latent regularization (Epps-Pulley normality test)
     3. ODE matching loss (Euler step consistency)

10. **Sampling** (lines 777-828)
    - `sample_sequences_ode`: Generate sequences by:
      - Sampling z0 ~ N(0,I)
      - Rolling out ODE to get z_path
      - Autoregressively decoding tokens from latent path

11. **Training Loop** (lines 830-905)
    - `train_ode`: AdamW optimizer with learning rate 1e-3
    - Loss weight warmup for EP term (0.0005 → final over 10k steps)
    - Periodic sampling every 100 steps

12. **Main Execution** (lines 907-954)
    - Dataset: 100k synthetic samples
    - Hyperparameters:
      - Batch size: 128
      - Sequence length: 64
      - Latent size: 64
      - Hidden size: 128
      - Embedding size: 64
      - Slices for EP test: 1024
    - Training: 100k steps

## Key Architectural Decisions

### 1. Deterministic Encoder + Stochastic Prior
- Encoder produces deterministic latent paths z(tokens)
- Prior learns stochastic ODE dynamics in latent space
- Enables both reconstruction and generation

### 2. ODE Matching Loss
The model learns f(z,t) such that:
```
z_{t+1} ≈ z_t + f(z_t, t) * Δt
```
This regularizes the discrete encoder outputs to follow smooth ODE dynamics.

### 3. Latent Normality Regularization
- Uses sliced Epps-Pulley test to encourage z ~ N(0,I)
- Applied to both encoder outputs and ODE predictions
- Enables sampling from standard normal prior

### 4. Autoregressive Decoding with Latent Conditioning
- Decoder sees full latent trajectory z_{0:t} when predicting x_t
- Maintains causal structure for token generation
- Shift-right pattern for teacher forcing (line 425)

## Development Workflows

### Running Training
```bash
python latent_drift_trajectory.py
```

The script will:
1. Print an example synthetic sequence
2. Train for 100k steps with progress bar
3. Print sample generations every 100 steps
4. Show two types of samples:
   - "Samples that share a Z": Same z0, different noise
   - "Samples with a random Z": Different z0 per sample

### Device Selection
Automatic device selection (line 913-916):
- macOS: MPS if available
- Otherwise: CUDA if available
- Fallback: CPU

### Monitoring Training
Progress bar shows (line 877-884):
- Total loss
- Reconstruction loss (`rec`)
- Epps-Pulley statistic (`ep`)
- ODE regularization loss (`ode`)
- Current EP weight during warmup

## Key Code Patterns & Conventions

### 1. Tensor Shapes (Critical!)
The codebase uses consistent shape conventions:

```python
# Common shapes:
# B = batch size
# L = sequence length
# D = latent dimension
# H = hidden dimension
# E = embedding dimension
# V = vocabulary size
# N = number of samples (for statistical tests)

tokens:     (B, L)           # Discrete tokens
z:          (B, L, D)        # Latent trajectories
embeddings: (B, L, E)        # Token embeddings
hidden:     (B, L, H)        # Transformer hidden states
logits:     (B, L, V)        # Decoder output logits
```

### 2. Time Convention
- Time t ∈ [0, 1] for ODE dynamics
- Normalized over sequence length
- See line 711: `t_grid = torch.linspace(0.0, 1.0, L, ...)`

### 3. Distributed Training Support
- Code includes DDP hooks (lines 66-76)
- `all_reduce` for synchronizing statistics
- `is_dist_avail_and_initialized()` checks

### 4. Teacher Forcing Pattern
```python
# Line 425-427: Shift tokens right for AR modeling
tokens_in = tokens.roll(1, dims=1)
start_token_id = char2idx["_"]
tokens_in[:, 0] = start_token_id
```

### 5. Weight Initialization
Custom init pattern (lines 932-939):
- Linear: truncated normal (std=0.02)
- Biases: zeros
- LayerNorm: ones for weight, zeros for bias

### 6. Loss Weight Warmup
Lines 849-868: EP loss weight increases from 0.0005 → final over 10k steps
- Prevents early training collapse
- Linear interpolation

## Important Variables & Hyperparameters

### Fixed Constants
```python
vocab_size = 29              # _, A-Z, !, >, ?
seq_len = 64                 # Base sequence length
target_block_len = 8         # Length of target letter block
noise_prob = 1/16            # Character corruption probability
```

### Tunable Hyperparameters
At lines 841-842 and 758-761:
```python
loss_weights = (1.0, 0.05, 1.0)  # (recon, EP, ODE)
lr = 1e-3
weight_decay = 1e-5
batch_size = 128
latent_size = 64
hidden_size = 128
embed_size = 64
num_slices = 1024            # For sliced EP test
```

### ODE Solver
```python
n_steps = seq_len - 1        # Number of Euler steps
t_span = [0, 1]              # Time interval
```

### Epps-Pulley Test
```python
t_max = 5.0                  # Max frequency for CF evaluation
n_points = 17                # Integration points (must be odd)
```

## Common Modifications & Extensions

### 1. Changing Sequence Length
Modify line 924 and dataset property (line 36):
```python
seq_len = 128  # or desired length
# Also update SyntheticTargetDataset.T
```

### 2. Adjusting Loss Weights
Line 842:
```python
loss_weights = (w_recon, w_latent, w_ode)
```
- Higher `w_recon`: Better reconstruction, less smooth dynamics
- Higher `w_latent`: More Gaussian latents, less expressiveness
- Higher `w_ode`: Smoother ODE dynamics, potentially less flexible

### 3. Using SDE Instead of ODE
The architecture supports SDEs. To implement:
1. Modify `PriorODE` to return both drift and diffusion
2. Update `solve_ode` to use stochastic integration
3. Add noise term: `dz = f(z,t)*dt + σ(z,t)*dW`

### 4. Enabling Latent Smoothing
Uncomment line 540 in `DeterministicEncoder.forward`:
```python
z = self.local_smooth(z)  # Gaussian smoothing over sequence
```

### 5. Adding New Datasets
Create a new Dataset class following `SyntheticTargetDataset` pattern:
```python
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        # Return tensor of token indices, shape (seq_len,)
        pass
```

## Testing & Validation

### Visual Inspection
The model outputs samples every 100 steps. Good training shows:
1. **Shared Z samples**: Should show similar structure/patterns
2. **Random Z samples**: Should show diverse outputs
3. **Target completion**: Sequences should contain coherent letter blocks

Example from README (lines 16-35):
```
Samples that share a Z:
?B>!_______________________________________________________________
?B>_!_____________________________BBBBBBBB_________________________
?B>!____________________________BBBBBBBBB__________________________
...

Samples with a random Z:
?I>_!_____________________________!IIIIIIIII__!!_!_________________
?R>!___________________!_________!________!____!___________________
?E>_!EEEEEEEE___________________!__________________________________
...
```

### Loss Monitoring
Healthy training shows:
- `rec` decreasing steadily (reconstruction improving)
- `ep` staying reasonably low (<1.0 typically)
- `ode` decreasing (dynamics fitting encoder path)

### Common Issues

1. **Loss explosion**: Reduce learning rate or increase warmup
2. **Poor samples**: Check loss weights, may need more EP regularization
3. **Degenerate latents**: Increase `w_latent` or reduce `w_ode`
4. **No diversity**: Check prior ODE training, may be mode-collapsed

## Git Workflow

This repository uses feature branches with Claude-specific naming:
- Branch pattern: `claude/claude-md-{session-id}`
- Always develop on designated branch
- Commit with descriptive messages
- Push with: `git push -u origin <branch-name>`

## Dependencies

Required packages (inferred from imports):
```python
torch                 # Core deep learning
tqdm                  # Progress bars
```

No `requirements.txt` currently exists. Recommended Python 3.9+.

## Code Quality Notes

### Strengths
- Clear separation of concerns (encoder, decoder, prior)
- Well-commented key sections
- Consistent naming conventions
- Type hints on function signatures

### Areas for Improvement
- No type hints on class methods
- Some magic numbers hardcoded (e.g., 11 layers at line 349)
- No logging beyond print statements
- No checkpointing/model saving
- No evaluation metrics beyond visual inspection

## Research Notes

From README.md (lines 4-13):
> "This is implimented as A latent ode (sde works as well). Which, is honestly pretty fasinating that this works. One of the big wins here would be the fact you can potentially roll out the latent trajectory without growing the context length of the backbone."

> "I think this is a good candidate for continual learning and inference time planning."

> "There is still some work todo on tweaking params, and I am not sure what can be ablated. But it's working as intended atleast some of the time."

**Interpretation**:
- Active research project, parameters not fully tuned
- Some instability expected ("works at least some of the time")
- Ablation studies pending

## Advanced Topics

### 1. Latent Trajectory Rollout Without Context Growth
Key benefit at inference: Instead of maintaining O(T) context, can:
1. Encode prefix to get z0
2. Roll out ODE: z_t = integrate(f, z0, t)
3. Decode from z_t without full history

### 2. Inference-Time Planning
Potential use: Search over z0 space to find trajectories satisfying constraints
- Sample multiple z0
- Roll out ODE dynamics
- Score/filter by decode quality or constraints
- Resample/optimize

### 3. Continual Learning
Architecture enables:
- Update encoder on new data
- Keep prior ODE frozen or adapt slowly
- Latent space remains approximately Gaussian (via EP regularization)

## Performance Characteristics

### Computational Bottlenecks
1. **Encoder**: 4-layer bidirectional transformer - O(L²) attention
2. **ODE Matching**: O(L) forward passes through drift network
3. **EP Test**: 1024 random projections + CF evaluation
4. **Decoder**: Autoregressive generation - O(L) sequential steps

### Memory Usage
- Dominant: Transformer attention matrices O(B × L² × H)
- ODE intermediate states: O(B × L × D)
- Gradients: ~2× parameter count

### Scaling Considerations
- Batch size limited by memory (current: 128)
- Sequence length quadratic in attention (current: 64)
- Latent size affects ODE network (current: 64)

## When to Ask User for Clarification

1. **Modifying loss weights**: These are sensitive, ask for target behavior
2. **Architecture changes**: Confirm before major structural changes
3. **Dataset changes**: Different data may need different hyperparameters
4. **Performance issues**: User may have domain knowledge about expected behavior
5. **Ablation studies**: Clarify which components to remove/modify

## References in Code

Key implementation details:
- Positional encoding: Lines 554-565 (sinusoidal, not learned)
- Attention: Lines 568-602 (custom QKV implementation)
- ODE solver: Lines 306-322 (simple Euler method)
- EP test: Lines 103-152 (fast version with trapezoid integration)

## Useful Line Number References

| Component | Lines | Description |
|-----------|-------|-------------|
| Dataset | 32-64 | Synthetic target generation |
| EP Test | 103-152 | Fast normality test |
| ODE Solver | 306-322 | Euler integration |
| Prior ODE | 343-367 | Drift function network |
| Encoder | 447-488, 496-542 | Bidirectional transformer |
| Decoder | 373-440 | Causal AR transformer |
| Main Model | 650-775 | Loss computation |
| Training | 837-905 | Optimization loop |
| Sampling | 781-828 | Generation procedure |

## Summary for AI Assistants

When working with this code:

1. **This is research code** - expect experimentation, not production quality
2. **Single file** - all modifications go in `latent_drift_trajectory.py`
3. **Shape conventions matter** - always verify (B, L, D) vs (B, L, H) etc.
4. **Loss weights are sensitive** - small changes can break training
5. **Visual inspection is primary validation** - watch the printed samples
6. **ODE matching is the key insight** - don't break this without understanding impact
7. **EP regularization prevents collapse** - needed for generation from prior

**Most likely requests:**
- Tuning hyperparameters (loss weights, layer counts, dimensions)
- Adding checkpointing/logging
- Implementing new datasets
- Experimenting with architecture variations
- Adding evaluation metrics

**Avoid without discussion:**
- Removing EP regularization
- Changing time normalization
- Breaking ODE matching loss
- Modifying shape conventions inconsistently
