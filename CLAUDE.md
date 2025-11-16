# CLAUDE.md - AI Assistant Guide for Latent Trajectory Flow Transformer

## Project Overview

This repository implements a **Latent Trajectory Flow Transformer** - a novel architecture that combines discrete/text transformers with a time-dependent latent space suitable for planning in trajectory space. The implementation uses a latent ODE (Ordinary Differential Equation) to model continuous dynamics in latent space while processing discrete sequences.

**Key Innovation**: The model can potentially roll out latent trajectories without growing the context length of the backbone transformer, making it suitable for continual learning and inference-time planning.

## Repository Structure

```
latent_trajectory_transformer/
├── latent_drift_trajectory.py  # Main implementation (954 lines)
├── README.md                    # Project overview and examples
└── CLAUDE.md                    # This file
```

### Single-File Architecture

This is intentionally a **monolithic implementation** contained in one Python file. When making changes:
- Keep all code in `latent_drift_trajectory.py`
- Do NOT split into multiple files unless explicitly requested
- Maintain the current section organization (marked by comment dividers)

## Code Structure and Components

The code is organized into logical sections separated by comment dividers (`# ────────...`):

### 1. Character-Level Dataset (Lines 14-64)
- **Vocabulary**: `{_, A-Z, !, >, ?}` (29 tokens total)
- **`SyntheticTargetDataset`**: Generates sequences with target letters embedded in noise
- **Encoding/Decoding**: `encode()` and `decode()` functions for string ↔ tensor conversion

### 2. Distributed Training Support (Lines 66-101)
- **`is_dist_avail_and_initialized()`**: Check for distributed setup
- **`all_reduce()`**: Tensor reduction across ranks
- **`UnivariateTest`**: Base class for statistical tests with DDP support

### 3. Statistical Testing (Lines 103-288)
- **`FastEppsPulley`**: Univariate normality test using characteristic functions
- **`EppsPulleyCF`**: Alternative Epps-Pulley implementation
- **`SlicingUnivariateTest`**: Multivariate test via random projections
- **Purpose**: Regularize latent distributions to be approximately Gaussian

### 4. ODE Framework (Lines 290-323)
- **`ODE`**: Abstract base class for ODE systems
- **`solve_ode()`**: Simple Euler method ODE solver
- **Time range**: [0, 1] normalized time

### 5. Prior Models (Lines 325-368)
- **`PriorInitDistribution`**: Learned initial distribution for latent trajectories
- **`PriorODE`**: Deep neural network (11 layers) modeling drift dynamics
  - Input: `[z, t]` (latent state + time)
  - Output: `dz/dt` (velocity in latent space)

### 6. Observation Model (Lines 370-441)
- **`DiscreteObservation`**: Autoregressive transformer decoder
- **Architecture**: Single causal TransformerBlock
- **Functionality**: `p(x_t | x_{<t}, z_{0:t})` - predicts tokens from latent path

### 7. Encoder (Lines 443-543)
- **`PosteriorEncoder`**: Non-causal transformer (4 blocks) for sequence encoding
- **`DeterministicEncoder`**: Projects encoded sequences to latent space
- **`local_smooth()`**: Optional Gaussian smoothing along sequence (currently disabled)

### 8. Transformer Building Blocks (Lines 545-649)
- **`AddPositionalEncoding`**: Sinusoidal position encoding
- **`QKVAttention`**: Multi-head attention with optional causal masking
- **`TransformerBlock`**: Standard transformer block (attention + FFN)
- **`Predictor`**: Simple MLP for latent prediction (currently unused in main path)

### 9. Main Model (Lines 650-776)
- **`DeterministicLatentODE`**: Complete model combining all components
- **Key Methods**:
  - `encode()`: Tokens → latent sequence
  - `decode_logits()`: Latent → token logits
  - `ode_matching_loss()`: Train ODE to match encoder dynamics
  - `loss_components()`: Returns (recon_loss, latent_reg, ode_reg_loss)
  - `forward()`: Combines losses with weights

### 10. Sampling (Lines 778-829)
- **`sample_sequences_ode()`**: Generate sequences by:
  1. Sampling initial z0
  2. Solving ODE forward in time
  3. Autoregressively sampling tokens from decoder

### 11. Training (Lines 831-905)
- **`train_ode()`**: Main training loop
- **Warmup schedule**: EP regularization weight increases over 10k steps
- **Logging**: Every 100 steps, generates samples with fixed/random z0

### 12. Main Entry Point (Lines 907-954)
- Device selection (MPS/CUDA/CPU)
- Hyperparameters
- Model initialization
- Training invocation

## Development Workflows

### Making Code Changes

1. **Always read the file first**:
   ```python
   # Read to understand context before editing
   Read("latent_drift_trajectory.py")
   ```

2. **Preserve exact formatting**:
   - Use 4-space indentation (not tabs)
   - Keep comment divider lines intact
   - Match existing code style

3. **Test changes locally**:
   ```bash
   python latent_drift_trajectory.py
   ```

4. **Commit with descriptive messages**:
   - Focus on the "why" not just "what"
   - Reference line numbers for specific changes

### Hyperparameter Tuning

Key hyperparameters are defined in `__main__` (lines 912-952):

```python
batch_size = 128          # Batch size
seq_len = 64              # Sequence length (fixed)
latent_size = 64          # Latent dimension
hidden_size = 128         # Transformer/MLP hidden size
embed_size = 64           # Token embedding size
num_slices = 1024         # Slices for EP test
train_steps = 100_000     # Training iterations
```

Loss weights (line 842):
```python
loss_weights = (1.0, 0.05, 1.0)  # (recon, latent_reg, ode_reg)
# Note: latent_reg has warmup from 0.0005 to 0.05 over first 10k steps
```

**Important**: There's a bug on line 865 - `current_ep` calculation uses wrong index:
```python
# CURRENT (INCORRECT):
current_ep = initial_ep + interp * (final_ep - initial_ep)

# Should reference loss_weights[1] for final_ep
```

### Adding New Features

When extending the model:

1. **Add new classes in appropriate section**: Use comment dividers to mark new sections
2. **Update imports at top**: Add any new dependencies after line 12
3. **Modify loss_components()**: If adding new loss terms
4. **Update train_ode()**: If changing training procedure
5. **Document in README.md**: Explain the new feature

### Testing and Validation

The model has no formal test suite. Validation is done by:

1. **Visual inspection**: Sample outputs printed every 100 steps
2. **Loss monitoring**: Check that recon/ep/ode losses decrease
3. **Pattern learning**: Verify model learns to generate target patterns

Expected behavior:
- **Shared Z samples**: Should show similar patterns (same target letter)
- **Random Z samples**: Should show diverse patterns (different letters)

## Key Conventions for AI Assistants

### Code Style

1. **Type hints**: Use throughout (`Tensor`, `int`, `float`, etc.)
2. **Docstrings**: Triple-quoted strings for classes and complex functions
3. **Comments**:
   - Inline for shape annotations: `# (B, L, D)`
   - Block comments for explaining algorithms
4. **Naming**:
   - `snake_case` for functions/variables
   - `PascalCase` for classes
   - Single letters for common tensors: `z` (latent), `x` (input), `t` (time)

### Mathematical Conventions

- **Time**: Always normalized to [0, 1] range
- **Batch dimension**: Always first: `(B, ...)`
- **Sequence length**: `L` or `T`
- **Latent dimension**: `D` or `latent_size`
- **ODE dynamics**: `dz/dt = f(z, t)` where `f` is the drift

### Loss Components

The model optimizes three objectives:

1. **Reconstruction loss** (`recon_loss`): Negative log-likelihood of observed tokens
   - Standard cross-entropy for autoregressive decoding

2. **Latent regularization** (`latent_reg`): Epps-Pulley statistic
   - Encourages latent samples to be approximately Gaussian
   - Uses sliced testing for multivariate normality

3. **ODE regression** (`ode_reg_loss`): L1 distance between:
   - Actual increments: `z_{t+1} - z_t`
   - Predicted increments: `f(z_t, t) * dt`

### Common Pitfalls

1. **Shape mismatches**: Always check tensor shapes, especially when mixing batch/sequence dimensions
2. **Device placement**: Ensure all tensors are on same device (MPS/CUDA/CPU)
3. **Gradient flow**: ODE matching uses `.detach()` on line 725 to prevent backprop through encoder
4. **Causal masking**: `DiscreteObservation` must use `causal=True`, encoder uses `causal=False`

### Performance Considerations

1. **Memory usage**: Scales with `batch_size × seq_len × latent_size`
2. **Computation**: ODE solving is O(n_steps) per sample
3. **Statistical tests**: EP test is O(num_slices × n_points), can be expensive
4. **Distributed training**: Supported via PyTorch DDP, but not enabled by default

### Debugging Tips

1. **Check loss values**:
   - `recon_loss` should decrease below 1.0
   - `latent_ep` should stay near 0-5 range
   - `ode_reg` should decrease toward 0

2. **Inspect samples**:
   - Fixed Z should show consistent patterns
   - Random Z should show diversity
   - Look for target letters appearing in samples

3. **Visualize latents**:
   ```python
   z = model.encode(tokens)
   print(z.shape, z.mean(), z.std())
   ```

4. **ODE diagnostics**:
   ```python
   ode_loss, z_pred = model.ode_matching_loss(z)
   print((z[:, 1:] - z_pred).abs().max())  # Check prediction error
   ```

## Recent Changes and Known Issues

### Recent Commits

1. **00bfaec**: "Update loss_weights in latent_drift_trajectory.py"
2. **a54f092**: "Fix loss_weights index for current_ep calculation"
3. **f6ca1b7**: "Remove redundant text from comment in code"
4. **20ce148**: "Rename README.py to README.md"
5. **1dbe01f**: "Create README.py for Latent Trajectory Flow Transformer"

### Known Issues

1. **Hyperparameter tuning needed**: README notes "still some work todo on tweaking params"
2. **Ablation unclear**: Not clear which components can be removed
3. **Inconsistent results**: Works "atleast some of the time" - may need stabilization
4. **Commented code**: Line 540 has commented `local_smooth()` call
5. **Unused components**: `Predictor` class defined but not used in main training path

## Working with This Codebase

### Before Making Changes

1. **Read README.md**: Understand the motivation and examples
2. **Read entire Python file**: Understand component relationships
3. **Run training**: See baseline behavior before modifications
4. **Check samples**: Verify current generation quality

### When Modifying

1. **Preserve working features**: Don't break existing functionality
2. **Test incrementally**: Make small changes and test frequently
3. **Monitor all losses**: Ensure no single loss dominates
4. **Check both sample types**: Fixed Z and random Z should both work

### When Stuck

1. **Check shapes**: Print tensor shapes liberally
2. **Simplify**: Try smaller models (reduce hidden_size, num_slices)
3. **Reduce complexity**: Temporarily disable components (set loss weights to 0)
4. **Compare to baseline**: Diff against last working commit

## Git Workflow

- **Current branch**: `claude/claude-md-mi19hues2l74efxg-013ka3arkXmoZDCJFBwdM7wZ`
- **Commit style**: Imperative mood, concise (e.g., "Fix loss calculation")
- **Push command**: `git push -u origin <branch-name>`

## Dependencies

Core dependencies (install with pip/conda):
```
torch>=2.0.0
tqdm
```

Optional for distributed training:
```
torch.distributed
```

The code auto-detects device:
- macOS: MPS (Metal Performance Shaders)
- NVIDIA: CUDA
- Fallback: CPU

## Questions and Contact

For questions about this codebase:
1. Check README.md for high-level overview
2. Read relevant code sections (use comment dividers to navigate)
3. Run experiments to understand behavior
4. Refer to recent commits for context on changes

---

**Last Updated**: 2025-11-16
**Repository**: latent_trajectory_transformer
**Main File**: latent_drift_trajectory.py (954 lines)
