# ULTRATHINK PLAN: Latent Trajectory Transformer → Production Codebase Search

**Author**: Claude (11 specialized agent analysis)
**Date**: 2025-11-16
**Branch**: feynman (merged)
**Objective**: Transform research code into production CPU-optimized codebase search tool with flow matching

---

## Executive Summary

Transform the latent trajectory transformer from toy character generation to **production codebase search** that processes queries on a single CPU core. Key innovations:

1. **Flow Matching Integration**: Scale latent trajectories with rectified flows (2-3x faster, more expressive)
2. **CPU Optimization**: BFloat16, Flash Attention CPU, gradient checkpointing (<100ms inference, <2GB memory)
3. **Practical Application**: Continual learning codebase search (handles 10k+ files, >80% accuracy)
4. **Syslog Processing**: Demonstrate on firewall packets (real-world validation)

**Expected Impact**: Research prototype → deployable tool in ~6 phases

---

## Kepner-Tregoe IS/IS-NOT Analysis

### Current State Analysis

| Dimension | **IS** (What's Working) | **IS-NOT** (What's Missing) |
|-----------|------------------------|----------------------------|
| **WHAT** | Latent ODE with Euler integration (lines 290-323), SDE with drift+diffusion (lines 971-1015), Raccoon continual learning (lines 1337-1566), experience replay with priority sampling (lines 1166-1253), character-level generation, log classification | Flow matching/rectified flows, CPU-optimized inference kernels, production codebase search application, quantization (INT8/INT4), adaptive ODE solvers (RK4/RK45), practical tutorial, real syslog validation |
| **WHERE** | Single-file implementation (1824 lines), forced CPU device (line 1670), character sequences (64 chars), log messages (50 chars), latent space (ODE: 64-dim, Raccoon: 32-dim) | Distributed across modules, GPU optimization, long documents (1000+ tokens), codebase files (variable length), optimal latent dimensionality (agent suggested 16-dim ODE, 8-dim Raccoon) |
| **WHEN** | Research prototyping phase, toy tasks (synthetic characters, synthetic logs), small batches (ODE: 128, Raccoon: 32), short sequences (<100 tokens), 1000-step training runs | Production deployment, real-world data (actual syslogs, actual codebases), large batches (256+), long sequences (1000+ tokens), 100k+ step training, online inference with <100ms latency |
| **EXTENT** | Proof-of-concept working "some of the time" (per user), numerical stability fixes applied (epsilon, gradient bounds, KL formula), backward compatibility maintained, 11 specialized agent improvements identified (15k+ lines analysis) | Production-grade reliability (99%+ uptime), comprehensive error handling, deployment infrastructure, monitoring/logging, automated testing suite, performance benchmarks on real hardware, ablation studies identifying minimal architecture |

### Problem Definition

**PRIMARY PROBLEM**: Research code demonstrates latent trajectory planning concept but lacks:
1. Scalability (flow matching for richer trajectories)
2. Efficiency (CPU optimization for single-core deployment)
3. Practicality (real application solving actual problem)

**SECONDARY PROBLEM**: User needs to process syslogs/firewall packets on CPU-constrained hardware + wants codebase search demonstrating the architecture's power.

**ROOT CAUSE**: Focused on research validation, not production deployment. Missing:
- Modern flow-based dynamics (rectified flows > vanilla ODE)
- CPU-specific optimizations (BFloat16, efficient attention)
- Application layer (search interface, continual learning on codebases)

---

## Phase 1: Architecture Deep Dive

### 1.1 Current Latent ODE System

**What It Does** (lines 290-776):
```
Input: x (tokens) → Encoder → z0 (initial latent) → ODE → z_path (trajectory) → Decoder → x_reconstructed
                                                       ↓
                                              Regularization (EP test: z ~ N(0,I))
```

**Key Innovation**:
- Fixed context length regardless of trajectory length
- Can plan in latent space before decoding
- ODE matching loss ensures trajectory follows learned dynamics

**Current Limitations**:
1. **Euler integration** (line 312-322): O(dt) error, unstable for stiff systems
2. **Deterministic ODE**: Single trajectory, no stochasticity modeling
3. **No flow matching**: Latent dynamics learned via reconstruction + matching, not as flow

### 1.2 Current Raccoon SDE System

**What It Does** (lines 912-1823):
```
Input: x (log) → Encoder → z ~ N(μ, σ²) → SDE → z_traj → Flow → z_transformed → Classifier → class
                                           ↓              ↓
                                   Drift + Diffusion   Coupling Layers (4x)
                                           ↓
                                   Experience Replay Memory (priority sampling)
```

**Key Innovation**:
- Stochastic dynamics capture uncertainty
- Normalizing flows increase expressiveness
- Continual learning via experience replay (prevents catastrophic forgetting)

**Current Limitations**:
1. **Euler-Maruyama** (line 1041-1048): Simple SDE solver, could use Milstein
2. **Affine coupling** (lines 1055-1110): Less expressive than neural splines
3. **No flow matching**: Flows trained with KL + reconstruction, not FM objective

---

## Phase 2: Flow Matching Integration Strategy

### 2.1 What is Flow Matching?

**Flow matching** (aka rectified flows) learns a **velocity field** v(z, t) such that:
```
dz/dt = v(z, t)
z(0) ~ p_data (actual latent encodings)
z(1) ~ p_prior (simple Gaussian)
```

**Key Advantages over VAE/ODE**:
1. **Simpler training**: No adversarial training, no ELBO bound
2. **Straighter paths**: Rectified flows minimize trajectory curvature (faster inference)
3. **Exact likelihood**: Can compute log p(z) via ODE integration
4. **Scalability**: State-of-the-art for large generative models

### 2.2 Integration Option 1: Full Trajectory Flow Matching

**Concept**: Treat entire latent trajectory as flow state.

**Architecture**:
```python
# Instead of:
z_path = solve_ode(prior_ode, z0, t_start=0.0, t_end=1.0, n_steps=seq_len)

# Do:
z_path = solve_rectified_flow(velocity_net, z0, t_start=0.0, t_end=1.0, n_steps=seq_len)
```

**Training**:
```python
# Flow matching loss (simple!)
t = torch.rand(batch_size, 1)  # Random time in [0, 1]
z0 = sample_data_latents(x)     # Encode actual data
z1 = torch.randn_like(z0)       # Sample from prior

# Interpolate (rectified flow)
zt = t * z1 + (1 - t) * z0      # Linear interpolation
target_velocity = z1 - z0        # Optimal straight-line velocity

# Predict velocity
v_pred = velocity_net(zt, t)

# Loss: make predicted velocity match optimal velocity
loss_fm = F.mse_loss(v_pred, target_velocity)
```

**Pros**:
- Entire trajectory learned as coherent flow
- Can generate diverse trajectories by sampling different z1 ~ N(0, I)
- Straighter paths → faster inference

**Cons**:
- Lose explicit ODE matching loss (but can keep it as auxiliary)
- Harder to interpret (no separate "drift" network)

**Expected Performance**:
- 2-3x faster inference (straighter paths, fewer steps)
- Better sample quality (rectified flows > VAE in generative modeling)
- More stable training (no KL annealing, simpler objective)

### 2.3 Integration Option 2: Summary Latent Flow Matching

**Concept**: Keep existing ODE for full trajectory, add flow matching on **summary latent**.

**Architecture**:
```python
# Existing ODE trajectory
z_path = solve_ode(prior_ode, z0, t_start=0.0, t_end=1.0, n_steps=seq_len)  # (B, L, D)

# Aggregate to summary latent
z_summary = z_path.mean(dim=1)  # (B, D) - could also use attention pooling

# Flow matching on summary
z_flow = solve_rectified_flow(velocity_net, z_summary, t_start=0.0, t_end=1.0, n_steps=10)

# Use flowed latent for reconstruction
x_recon = decoder(z_flow.expand(batch, seq_len, latent_dim))
```

**Training** (phased curriculum):
```python
# Phase 1 (steps 0-10k): Only ODE loss
loss = recon_loss + ode_matching_loss + ep_regularization

# Phase 2 (steps 10k-20k): Add flow matching gradually
alpha = min(1.0, (step - 10000) / 10000)  # 0 → 1 over 10k steps
loss = recon_loss + (1 - alpha) * ode_matching_loss + alpha * flow_matching_loss + ep_regularization

# Phase 3 (steps 20k+): Primarily flow matching
loss = recon_loss + 0.1 * ode_matching_loss + flow_matching_loss + ep_regularization
```

**Pros**:
- Keep existing ODE trajectory (backward compatible)
- Add flow matching as enhancement
- Gradual transition (phased curriculum)

**Cons**:
- Two separate dynamics systems (ODE + flow)
- More complex architecture
- Summary latent loses sequential structure

**Expected Performance**:
- 1.5x faster inference (flow on summary is quick)
- Better reconstruction (richer summary latent)
- Smoother training (gradual phase-in)

### 2.4 Recommended Approach: Option 1 (Full Trajectory)

**Rationale**:
1. **Simpler**: Single dynamics system (flow replaces ODE)
2. **Faster**: Rectified flows are designed for fast inference
3. **State-of-the-art**: Flow matching is current best practice for latent dynamics
4. **User's suggestion**: "you could scale this by turning the latent ODE into an actual flow model"

**Implementation Plan**:
1. Create `VelocityNet` (similar to `PriorODE` but predicts velocity, not drift)
2. Implement `solve_rectified_flow` (reuse ODE solver, just different objective)
3. Add flow matching loss to training loop
4. **Optionally keep ODE matching as auxiliary** (0.1 weight) for stability

**Code Changes Required**:
- Lines 290-323: Add `solve_rectified_flow` function
- Lines 343-377: Create `VelocityNet` class (similar to `PriorODE`)
- Lines 725-747: Replace ODE matching loss with flow matching loss
- Lines 831-905: Update training loop with FM objective

---

## Phase 3: CPU Optimization Strategy

### 3.1 Current Performance Bottlenecks

**Profiling Analysis** (estimated from architecture):

| Component | CPU Time (%) | Memory (MB) | Optimization Potential |
|-----------|--------------|-------------|------------------------|
| **Attention (QKV)** | 40% | 800 MB | HIGH - Flash Attention CPU, quantization |
| **ODE/SDE Solver** | 25% | 200 MB | MEDIUM - Adaptive step size, fewer steps with rectified flows |
| **Normalizing Flows** | 15% | 300 MB | MEDIUM - Coupling layer fusion, quantization |
| **EP Regularization** | 10% | 100 MB | LOW - Already optimized, disable during inference |
| **Misc (embedding, etc.)** | 10% | 100 MB | LOW - Minor gains available |

**Target**: <100ms inference on single CPU core

### 3.2 Optimization 1: BFloat16 Mixed Precision

**Why BFloat16 > FP16 on CPU**:
- BFloat16 has same exponent range as FP32 (less overflow/underflow)
- Modern CPUs have BFloat16 instructions (AVX-512 BF16)
- 2x memory reduction, 1.5-2x speedup

**Implementation**:
```python
# Add to training loop (lines 831-905)
from torch.cuda.amp import autocast  # Works on CPU too with torch 2.0+

# Enable BF16 autocasting
with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    logits = model(tokens)
    loss = criterion(logits, labels)

# Scale loss for numerical stability
scaler = torch.cuda.amp.GradScaler()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected Gains**:
- 1.8x faster inference
- 2x memory reduction (4GB → 2GB for large models)
- Minimal accuracy loss (<1%)

### 3.3 Optimization 2: Flash Attention (CPU Variant)

**Problem**: Standard attention is O(N²) in memory and compute.

**Solution**: Flash Attention uses tiling to reduce memory from O(N²) to O(N).

**CPU Implementation** (simplified):
```python
# Replace QKVAttention (lines 558-602) with FlashAttention
import torch.nn.functional as F

def flash_attention_cpu(q, k, v, causal=False):
    """
    Memory-efficient attention using tiling.
    q, k, v: (B, H, L, D)
    """
    B, H, L, D = q.shape

    # Tile size (tuned for CPU cache)
    BLOCK_SIZE = 32

    output = torch.zeros_like(q)
    row_max = torch.full((B, H, L), float('-inf'), device=q.device)
    row_sum = torch.zeros(B, H, L, device=q.device)

    # Tile over sequence length
    for i in range(0, L, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, L)
        q_block = q[:, :, i:i_end, :]  # (B, H, block, D)

        for j in range(0, L, BLOCK_SIZE):
            j_end = min(j + BLOCK_SIZE, L)
            k_block = k[:, :, j:j_end, :]  # (B, H, block, D)
            v_block = v[:, :, j:j_end, :]

            # Compute attention scores for this tile
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / (D ** 0.5)

            # Apply causal mask if needed
            if causal and j >= i:
                mask = torch.triu(torch.ones(i_end - i, j_end - j), diagonal=j - i + 1).bool()
                scores = scores.masked_fill(mask, float('-inf'))

            # Online softmax (numerically stable)
            block_max = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - block_max)

            # Update running statistics
            new_max = torch.maximum(row_max[:, :, i:i_end].unsqueeze(-1), block_max)
            exp_sum = exp_scores.sum(dim=-1, keepdim=True)

            # Rescale previous output
            scale = torch.exp(row_max[:, :, i:i_end].unsqueeze(-1) - new_max)
            output[:, :, i:i_end, :] = output[:, :, i:i_end, :] * scale

            # Add new contribution
            output[:, :, i:i_end, :] += torch.matmul(exp_scores, v_block) * torch.exp(block_max - new_max)

            # Update statistics
            row_max[:, :, i:i_end] = new_max.squeeze(-1)
            row_sum[:, :, i:i_end] += exp_sum.squeeze(-1) * torch.exp(block_max.squeeze(-1) - new_max.squeeze(-1))

    # Final normalization
    output = output / row_sum.unsqueeze(-1)
    return output
```

**Expected Gains**:
- 1.5x faster for seq_len=64
- 2-3x faster for seq_len=256+
- Memory: O(N²) → O(N) (enables longer sequences)

**Note**: For production, use optimized library like `xformers` or `flash-attn` CPU backend.

### 3.4 Optimization 3: Gradient Checkpointing

**Problem**: Training stores all activations (high memory).

**Solution**: Recompute activations during backward pass (trade compute for memory).

**Implementation**:
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        # Wrap forward pass in checkpoint
        return checkpoint(self._forward, x, mask, use_reentrant=False)

    def _forward(self, x, mask):
        # Original forward logic
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
```

**Expected Gains**:
- 2-3x memory reduction during training
- 20-30% slower training (recomputation overhead)
- Enables 2-4x larger batch sizes (better throughput)

### 3.5 Optimization 4: INT8 Quantization (Inference Only)

**Problem**: FP32 weights are large and slow.

**Solution**: Quantize weights to INT8 for inference.

**Implementation**:
```python
import torch.quantization as quant

# Quantize model for inference
model_int8 = quant.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM, nn.GRU},  # Quantize these layers
    dtype=torch.qint8
)

# Inference is now 2-4x faster with INT8
with torch.no_grad():
    output = model_int8(input_tokens)
```

**Expected Gains**:
- 2-4x faster inference (CPU)
- 4x smaller model size (400MB → 100MB)
- ~1-2% accuracy loss (acceptable for search)

### 3.6 Combined Optimization Pipeline

**Training** (lines 831-905):
1. BFloat16 autocasting → 1.8x faster, 2x less memory
2. Gradient checkpointing → 2x less memory, enables bigger batches
3. Flash Attention → 1.5x faster attention
4. **Total**: ~3x faster training, 4x less memory

**Inference** (new function):
1. BFloat16 precision → 1.8x faster
2. Flash Attention → 1.5x faster
3. INT8 quantization → 2x faster
4. Rectified flows (fewer steps) → 2x faster
5. **Total**: ~10x faster inference (1000ms → <100ms)

---

## Phase 4: Codebase Search Application Design

### 4.1 Problem Statement

**User Need**: Search entire codebase using natural language queries.

**Challenge**:
- Traditional search (grep/ripgrep) requires exact keywords
- Semantic search requires large embeddings (BERT, etc.) - too slow on CPU
- Need continual learning (codebase evolves over time)

**Solution**: Use latent trajectory transformer with Raccoon continual learning.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CODEBASE SEARCH SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INDEXING PHASE (one-time setup)                         │
│     ┌──────────────────────────────────────────┐            │
│     │ Scan codebase → Tokenize files          │            │
│     │ → Encode to latent → Store in index     │            │
│     └──────────────────────────────────────────┘            │
│                                                              │
│  2. QUERY PHASE (real-time search)                          │
│     ┌──────────────────────────────────────────┐            │
│     │ User query → Tokenize → Encode to latent│            │
│     │ → Compare with index (cosine similarity)│            │
│     │ → Return top-k matches with snippets    │            │
│     └──────────────────────────────────────────┘            │
│                                                              │
│  3. CONTINUAL LEARNING (background)                         │
│     ┌──────────────────────────────────────────┐            │
│     │ Watch for file changes (git hooks)      │            │
│     │ → Incremental learning on new/modified  │            │
│     │ → Update index + experience replay      │            │
│     └──────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Data Preprocessing

**Tokenization** (extend vocabulary for code):
```python
# Current vocab: _, A-Z, !, >, ?, 0-9 (39 chars)
# Code vocab: Add common programming symbols

code_vocab = {
    # Existing
    **{chr(i): i for i in range(ord('A'), ord('Z') + 1)},  # A-Z
    **{chr(i): i for i in range(ord('a'), ord('z') + 1)},  # a-z
    **{chr(i): i for i in range(ord('0'), ord('9') + 1)},  # 0-9

    # Code symbols
    '_': 62, '.': 63, '/': 64, '-': 65, '(': 66, ')': 67,
    '{': 68, '}': 69, '[': 70, ']': 71, ':': 72, ';': 73,
    ',': 74, '=': 75, '+': 76, '*': 77, '#': 78, '@': 79,
    '!': 80, '?': 81, '<': 82, '>': 83, '&': 84, '|': 85,
    ' ': 86, '\n': 87, '\t': 88, '"': 89, "'": 90,
    '<PAD>': 0, '<UNK>': 91, '<CLS>': 92, '<SEP>': 93
}
# Total: 94 tokens
```

**File Chunking**:
```python
def chunk_file(file_path, chunk_size=256, overlap=64):
    """
    Split file into overlapping chunks for indexing.

    Args:
        file_path: Path to source file
        chunk_size: Max tokens per chunk
        overlap: Overlap between chunks (context)

    Returns:
        List of (chunk_text, start_line, end_line)
    """
    with open(file_path) as f:
        lines = f.readlines()

    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i + chunk_size]
        chunk_text = ''.join(chunk_lines)

        chunks.append({
            'text': chunk_text,
            'file': file_path,
            'start_line': i + 1,
            'end_line': i + len(chunk_lines)
        })

        i += chunk_size - overlap  # Sliding window

    return chunks
```

### 4.4 Index Structure

**In-Memory Index** (for fast search):
```python
class CodebaseIndex:
    def __init__(self):
        self.embeddings = []  # List of (latent_vector, metadata)
        self.index = None      # Faiss index for fast similarity search

    def add_file(self, file_path, model):
        """Add file to index by encoding chunks."""
        chunks = chunk_file(file_path)

        for chunk in chunks:
            # Tokenize chunk
            tokens = tokenize_code(chunk['text'])

            # Encode to latent vector
            with torch.no_grad():
                z = model.encode(tokens)  # (1, latent_dim)

            # Store in index
            self.embeddings.append({
                'vector': z.cpu().numpy(),
                'file': chunk['file'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line'],
                'text': chunk['text']
            })

        # Rebuild Faiss index
        self._rebuild_index()

    def _rebuild_index(self):
        """Build Faiss index for fast k-NN search."""
        import faiss

        vectors = np.array([e['vector'] for e in self.embeddings])
        dim = vectors.shape[1]

        # L2 index (could use cosine by normalizing)
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query_vector, top_k=10):
        """Find top-k most similar chunks."""
        D, I = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append({
                **self.embeddings[idx],
                'similarity': 1 / (1 + dist)  # Convert distance to similarity
            })

        return results
```

### 4.5 Search Interface

**CLI Tool**:
```python
# codebase_search.py
import argparse
from raccoon_search import CodebaseSearchEngine

def main():
    parser = argparse.ArgumentParser(description='Search codebase using latent trajectories')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--update', action='store_true', help='Update index before searching')
    args = parser.parse_args()

    # Initialize search engine
    engine = CodebaseSearchEngine(
        model_path='raccoon_codebase.pt',
        index_path='codebase.index'
    )

    # Update index if requested
    if args.update:
        print("Updating index...")
        engine.update_index()

    # Search
    results = engine.search(args.query, top_k=args.top_k)

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['file']}:{result['start_line']}-{result['end_line']} "
              f"(similarity: {result['similarity']:.3f})")
        print("-" * 80)
        print(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])

if __name__ == '__main__':
    main()
```

**Example Usage**:
```bash
# Initial indexing
python codebase_search.py --update "initialize index"

# Search for attention implementation
python codebase_search.py "multi-head attention with causal masking" --top-k=5

# Search for ODE solver
python codebase_search.py "solve ordinary differential equation euler integration"

# Search for experience replay
python codebase_search.py "memory buffer priority sampling continual learning"
```

### 4.6 Continual Learning Integration

**Git Hook** (watch for changes):
```bash
# .git/hooks/post-commit
#!/bin/bash

# Get list of changed files
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD | grep "\.py$")

if [ -n "$CHANGED_FILES" ]; then
    echo "Updating codebase index with changed files..."
    python codebase_search.py --incremental-update $CHANGED_FILES
fi
```

**Incremental Update**:
```python
def incremental_update(self, file_paths, model):
    """
    Continual learning on changed files.

    Uses Raccoon experience replay to:
    1. Learn new file representations
    2. Prevent forgetting existing index
    """
    # Encode new files
    new_chunks = []
    for file_path in file_paths:
        new_chunks.extend(chunk_file(file_path))

    # Sample from memory (experience replay)
    memory_samples = random.sample(self.embeddings, k=min(100, len(self.embeddings)))

    # Fine-tune model on new + memory
    for epoch in range(5):  # Quick fine-tuning
        # New data
        for chunk in new_chunks:
            tokens = tokenize_code(chunk['text'])
            model.continuous_update(tokens, auto_label=True)  # Unsupervised

        # Memory replay
        for sample in memory_samples:
            model.continuous_update(sample['tokens'], auto_label=True)

    # Re-encode and update index
    for chunk in new_chunks:
        tokens = tokenize_code(chunk['text'])
        with torch.no_grad():
            z = model.encode(tokens)

        self.embeddings.append({
            'vector': z.cpu().numpy(),
            'file': chunk['file'],
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'text': chunk['text']
        })

    self._rebuild_index()
```

### 4.7 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Indexing Speed** | >100 files/sec | Time to encode and add to index |
| **Search Latency** | <100ms | Query → results (including encoding) |
| **Memory Usage** | <2GB | Total RAM for 10k file codebase |
| **Search Accuracy** | >80% | Relevant results in top-5 (human eval) |
| **Incremental Update** | <10 sec | Time to update index with 10 changed files |

---

## Phase 5: Syslog Processing Application

### 5.1 Problem Statement

**User Quote**: "I just needed something that could eat syslogs firewall packets on a single cpu core"

**Requirements**:
1. Parse syslog/firewall packet logs
2. Classify log severity (ERROR, WARNING, INFO, DEBUG)
3. Detect anomalies (concept drift)
4. Run on single CPU core (<100ms per log)

### 5.2 Syslog Dataset

**Format** (RFC 5424):
```
<priority>timestamp hostname app-name process-id message-id message
```

**Example**:
```
<34>2024-11-16T10:23:45.123Z firewall kernel: [UFW BLOCK] IN=eth0 OUT= MAC=00:11:22:33:44:55 SRC=192.168.1.100 DST=10.0.0.1 LEN=60 PROTO=TCP SPT=54321 DPT=22 WINDOW=5840
```

**Preprocessing**:
```python
import re
from datetime import datetime

def parse_syslog(log_line):
    """
    Parse syslog line into structured format.

    Returns:
        {
            'priority': int,
            'timestamp': datetime,
            'hostname': str,
            'app': str,
            'message': str,
            'severity': str  # Derived from priority
        }
    """
    # Priority extraction
    match = re.match(r'<(\d+)>(.+)', log_line)
    if not match:
        return None

    priority = int(match.group(1))
    rest = match.group(2)

    # Severity (priority % 8)
    severity_map = {
        0: 'EMERGENCY',
        1: 'ALERT',
        2: 'CRITICAL',
        3: 'ERROR',
        4: 'WARNING',
        5: 'NOTICE',
        6: 'INFO',
        7: 'DEBUG'
    }
    severity = severity_map[priority % 8]

    # Parse timestamp, hostname, app, message
    parts = rest.split(None, 3)
    if len(parts) < 4:
        return None

    return {
        'priority': priority,
        'timestamp': datetime.fromisoformat(parts[0].replace('Z', '+00:00')),
        'hostname': parts[1],
        'app': parts[2],
        'message': parts[3],
        'severity': severity
    }
```

### 5.3 Classification Task

**Input**: Syslog message (tokenized)
**Output**: Severity class (7 classes) + anomaly score

**Model**: Use existing `RaccoonLogClassifier` with modifications:
```python
class SyslogClassifier(RaccoonLogClassifier):
    def __init__(self, vocab_size=94, num_classes=7, latent_dim=16, hidden_dim=32):
        # Smaller model for faster inference (16-dim latent, 32-dim hidden)
        super().__init__(vocab_size, num_classes, latent_dim, hidden_dim, ...)

    def forward(self, tokens):
        # Standard forward pass
        logits = super().forward(tokens)

        # Compute anomaly score (latent space distance from prior)
        z_mean = self.encoder_mean(self.embedding(tokens).mean(dim=1))
        anomaly_score = torch.norm(z_mean, dim=-1)  # Distance from N(0, I)

        return logits, anomaly_score
```

### 5.4 Real-Time Processing Pipeline

```python
import sys
from raccoon_syslog import SyslogClassifier

def process_syslog_stream(model_path='raccoon_syslog.pt'):
    """
    Process syslog stream from stdin in real-time.
    """
    model = SyslogClassifier.load(model_path)
    model.eval()

    # Continual learning buffer
    memory_buffer = []
    update_interval = 100  # Update every 100 logs

    for i, line in enumerate(sys.stdin, 1):
        # Parse syslog
        log = parse_syslog(line.strip())
        if not log:
            continue

        # Tokenize message
        tokens = tokenize_code(log['message'])

        # Classify
        with torch.no_grad():
            logits, anomaly_score = model(tokens.unsqueeze(0))
            pred_severity = logits.argmax(dim=-1).item()

        # Detect anomalies
        if anomaly_score > 3.0:  # Threshold (3 std devs)
            print(f"[ANOMALY] {log['message'][:80]} (score: {anomaly_score:.2f})")

        # Continual learning (incremental update)
        memory_buffer.append((tokens, pred_severity))

        if i % update_interval == 0:
            # Quick update with memory replay
            model.train()
            for tokens, label in memory_buffer[-10:]:  # Last 10 logs
                model.continuous_update(tokens.unsqueeze(0), torch.tensor([label]))
            model.eval()

            # Keep buffer bounded
            if len(memory_buffer) > 1000:
                memory_buffer = memory_buffer[-500:]  # Keep recent 500

        # Print classification
        severity_name = ['EMERGENCY', 'ALERT', 'CRITICAL', 'ERROR', 'WARNING', 'NOTICE', 'INFO', 'DEBUG'][pred_severity]
        print(f"[{severity_name}] {log['message'][:80]}")
```

**Usage**:
```bash
# Process live syslog
tail -f /var/log/syslog | python process_syslog.py

# Process firewall logs
tail -f /var/log/ufw.log | python process_syslog.py

# Benchmark on historical logs
cat /var/log/syslog | python process_syslog.py | head -1000
```

### 5.5 Performance Benchmarking

**Test Setup**:
```python
import time
import numpy as np

def benchmark_syslog_processing(model, test_logs, n_runs=100):
    """
    Benchmark latency and throughput.
    """
    latencies = []

    for _ in range(n_runs):
        log = random.choice(test_logs)
        tokens = tokenize_code(log['message'])

        start = time.perf_counter()
        with torch.no_grad():
            logits, anomaly_score = model(tokens.unsqueeze(0))
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    print(f"Mean latency: {np.mean(latencies):.2f} ms")
    print(f"P50 latency: {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"Throughput: {1000 / np.mean(latencies):.0f} logs/sec")
```

**Target Performance** (single CPU core):
- Mean latency: <50ms
- P99 latency: <100ms
- Throughput: >20 logs/sec

---

## Phase 6: Implementation Roadmap

### 6.1 Timeline (6 Phases, ~2-3 days)

| Phase | Duration | Deliverables | Success Criteria |
|-------|----------|--------------|------------------|
| **Phase 1: Baseline** | 2 hours | Run current code, profile performance | Reproduce character generation, identify bottlenecks |
| **Phase 2: Flow Matching** | 6 hours | Implement `VelocityNet`, `solve_rectified_flow`, flow matching loss | 2x faster inference, better sample quality |
| **Phase 3: CPU Optimization** | 8 hours | BFloat16, Flash Attention, gradient checkpointing, INT8 quantization | <100ms inference, <2GB memory |
| **Phase 4: Codebase Search** | 10 hours | Tokenizer, indexer, search engine, CLI tool | Search 10k files in <100ms |
| **Phase 5: Syslog Processing** | 6 hours | Syslog parser, classifier, real-time pipeline | Process logs at >20/sec with anomaly detection |
| **Phase 6: Tutorial** | 4 hours | Comprehensive markdown guide with examples | Full reproduction instructions |

**Total**: ~36 hours (~1.5-2 days of focused work)

### 6.2 File Structure (After Implementation)

```
latent_trajectory_transformer/
├── latent_drift_trajectory.py          # Original implementation (baseline)
├── latent_flow_transformer.py          # NEW: Flow matching version
├── utils/
│   ├── tokenizers.py                   # NEW: Code tokenizer
│   ├── flash_attention.py              # NEW: CPU-optimized attention
│   └── quantization.py                 # NEW: INT8 quantization utilities
├── applications/
│   ├── codebase_search.py              # NEW: Search engine
│   ├── syslog_classifier.py            # NEW: Syslog processing
│   └── raccoon_search_cli.py           # NEW: CLI tool
├── tests/
│   ├── test_flow_matching.py           # NEW: Flow matching tests
│   ├── test_cpu_optimization.py        # NEW: Performance tests
│   └── test_applications.py            # NEW: Application tests
├── benchmarks/
│   ├── profile_cpu.py                  # NEW: CPU profiling
│   └── benchmark_results.md            # NEW: Performance numbers
├── TUTORIAL.md                          # NEW: Comprehensive guide
├── ULTRATHINK_PLAN.md                   # This file
├── CLAUDE.md                            # Updated with new components
└── README.md                            # Updated with applications
```

---

## Phase 7: Success Metrics

### 7.1 Quantitative Targets

| Metric | Current (Baseline) | Target (After Optimization) | Measurement Method |
|--------|-------------------|----------------------------|-------------------|
| **Inference Latency** | ~1000ms (estimated) | <100ms | Time from tokens → output on CPU |
| **Training Speed** | ~5 steps/sec | ~15 steps/sec | Steps per second during training |
| **Memory Usage** | ~4GB | <2GB | Peak memory during inference |
| **Model Size** | ~400MB (FP32) | ~100MB (INT8) | Disk size of saved model |
| **Search Accuracy** | N/A (not implemented) | >80% top-5 | Human evaluation of search results |
| **Syslog Throughput** | N/A | >20 logs/sec | Logs processed per second on single core |
| **Sample Quality** | Baseline (current) | 20% improvement | Perplexity, diversity metrics |

### 7.2 Qualitative Goals

1. **Usability**: Command-line tools that "just work" (install → run → results)
2. **Documentation**: Tutorial that enables reproduction without expert knowledge
3. **Practicality**: Real-world applications (not just toy demos)
4. **Scalability**: Architecture can handle 10k+ file codebases, streaming syslogs
5. **Maintainability**: Clean code with comprehensive comments and type hints

---

## Phase 8: Risk Analysis & Mitigation

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Flow matching underperforms** | Medium | High | Keep ODE as fallback, use phased curriculum |
| **CPU optimization insufficient** | Low | High | Profile first, focus on bottlenecks, use existing libraries (xformers, etc.) |
| **INT8 quantization degrades accuracy** | Medium | Medium | Test thoroughly, use mixed precision (INT8 + FP32 for critical layers) |
| **Codebase search poor quality** | Medium | High | Use human evaluation, iterative prompt engineering, ensemble with keyword search |
| **Syslog format incompatibility** | Low | Medium | Support multiple formats (RFC 3164, 5424, custom), robust parsing with fallbacks |

### 8.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Scope creep** | High | Medium | Strict phasing, deliver minimal viable product (MVP) first |
| **Integration complexity** | Medium | High | Thorough testing at each phase, backward compatibility checks |
| **Performance gap** | Medium | High | Early profiling, iterative optimization, realistic targets |
| **Documentation burden** | Medium | Medium | Write docs as you go, code comments become tutorial sections |

---

## Conclusion

This plan transforms the latent trajectory transformer from research prototype to production-ready tool in 6 systematic phases:

1. ✅ **Baseline**: Understand current performance
2. 🚀 **Flow Matching**: Scale with state-of-the-art dynamics (2x faster)
3. ⚡ **CPU Optimization**: Deploy on single core (10x faster inference)
4. 🔍 **Codebase Search**: Practical application demonstrating architecture
5. 📋 **Syslog Processing**: Real-world validation on user's exact use case
6. 📚 **Tutorial**: Enable others to reproduce and extend

**Expected Outcome**: Research code → Deployable tool solving real problems (codebase search, syslog classification) on CPU-constrained hardware.

**Next Steps**:
1. Run baseline (Phase 1)
2. Begin flow matching implementation (Phase 2)
3. Iterate based on profiling results

---

**END OF ULTRATHINK PLAN**
