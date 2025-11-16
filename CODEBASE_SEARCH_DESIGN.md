# Latent Code Search: Architecture Design

## Vision
Transform the latent trajectory transformer into a practical, CPU-efficient codebase search engine that uses **trajectory planning in latent space** to find relevant code.

## Why Latent Trajectories for Code Search?

### Traditional Search Problems:
- **Grep/Ripgrep**: Fast but keyword-only, no semantic understanding
- **Embedding search**: Static vectors, no planning or reasoning
- **LLM-based**: Expensive, requires API calls or large models

### Latent Trajectory Solution:
1. **Incremental learning**: Learn codebase representation file-by-file (continual learning)
2. **Trajectory planning**: Model can "plan" search paths through code structure
3. **CPU-efficient**: Single-core operation, no GPU needed
4. **Adaptive**: Learns from queries and usage patterns

## Architecture

### Component 1: Code Tokenizer & Preprocessor

```
Input: Python/JS/etc source files
↓
Tokenize (char-level or simple BPE)
↓
Chunk into fixed-length windows (e.g., 512 chars)
↓
Extract metadata:
  - File path
  - Function/class names
  - Imports
  - Docstrings
↓
Output: {tokens, metadata} pairs
```

**Design Choice**: Start with char-level for simplicity, upgrade to BPE later

### Component 2: Latent Trajectory Encoder

```
Code Chunk → Encoder → (mean, logvar) → Sample z0
                    ↓
           SDE Dynamics: dz = f(z,t)dt + σ(z,t)dW
                    ↓
           Flow Model: z_flow = Flow(z_trajectory)
                    ↓
           Latent Trajectory: [z_0, z_1, ..., z_T]
```

**Key Innovation**: The trajectory represents the "reasoning path" through code concepts

### Component 3: Query-Code Similarity

```
Query (natural language) → Encode → q_trajectory
Code Chunk → Encode → c_trajectory

Similarity = trajectory_distance(q_trajectory, c_trajectory)

Where trajectory_distance could be:
- Euclidean distance at final timestep
- Wasserstein distance between full trajectories
- Maximum Mean Discrepancy (MMD)
- Learned distance function
```

### Component 4: Continual Learning Index

```
Index Building (one-time per repo):
for file in codebase:
    chunks = preprocess(file)
    for chunk in chunks:
        # Learn representation
        loss = model.train_step(chunk)
        # Store in memory buffer (Raccoon)
        model.memory.add(chunk, score=uncertainty)

    # Periodic consolidation
    if step % 100 == 0:
        model.consolidate(replay_from_memory)

```

**Benefit**: No need to retrain on entire codebase, just incremental updates

### Component 5: Search Interface

```
CLI Command:
$ lcs query "where is the SDE dynamics implemented?"

Process:
1. Encode query → q_trajectory
2. Scan indexed chunks (or use ANN for speed)
3. Rank by trajectory_distance(q_trajectory, chunk_trajectory)
4. Return top-k with context (surrounding lines, file path)
5. Learn from click/selection (online adaptation)
```

## Implementation Plan (Mapped to Phases)

### Phase 2: Flow Matching Enhancement
**Why**: Richer trajectory planning for better code understanding

```python
# Add flow matching objective
def flow_matching_loss(z0, z1, t):
    # Interpolate: z_t = (1-t)*z0 + t*z1
    z_t = (1 - t) * z0 + t * z1

    # Flow field should match velocity
    v_pred = flow_model(z_t, t)
    v_target = z1 - z0

    return F.mse_loss(v_pred, v_target)

# Combined objective
total_loss = (
    ode_loss +          # Original trajectory smoothness
    flow_matching_loss  # Flow-based planning
)
```

### Phase 3: Codebase Preprocessing
**Deliverable**: `CodeTokenizer` class

```python
class CodeTokenizer:
    def __init__(self, vocab_size=256):  # char-level
        self.vocab = list(range(256))

    def tokenize(self, code: str) -> List[int]:
        return [ord(c) % 256 for c in code]

    def chunk(self, tokens: List[int], window=512, stride=256):
        for i in range(0, len(tokens), stride):
            yield tokens[i:i+window]

class CodebaseC crawler:
    def crawl(self, repo_path: Path) -> Iterator[CodeChunk]:
        for file in repo_path.rglob("*.py"):
            code = file.read_text()
            metadata = extract_metadata(code)  # AST parse

            for chunk in tokenizer.chunk(code):
                yield CodeChunk(
                    tokens=chunk,
                    file_path=file,
                    metadata=metadata
                )
```

### Phase 4: Model Adaptation
**Key Changes to Raccoon Architecture**:

```python
class CodeSearchModel(RaccoonLogClassifier):
    def __init__(self, vocab_size=256, latent_dim=32, ...):
        # Similar architecture but:
        # - Input: code tokens (vocab_size=256 for char-level)
        # - Output: embedding (not classification)
        # - Loss: contrastive + trajectory smoothness

    def encode_query(self, query_text: str) -> Trajectory:
        """Natural language → latent trajectory"""
        tokens = self.query_tokenizer(query_text)
        return self.encode(tokens)  # Returns trajectory

    def encode_code(self, code_chunk: str) -> Trajectory:
        """Code → latent trajectory"""
        tokens = self.code_tokenizer(code_chunk)
        return self.encode(tokens)

    def similarity(self, q_traj: Trajectory, c_traj: Trajectory) -> float:
        """Trajectory distance in latent space"""
        # Option 1: Final state distance
        return F.cosine_similarity(q_traj[-1], c_traj[-1])

        # Option 2: Wasserstein distance (richer)
        return wasserstein_distance(q_traj, c_traj)
```

### Phase 5: Search Interface
**CLI Tool Structure**:

```bash
# Index a repository
$ lcs index /path/to/repo --output repo.index

# Search
$ lcs query "where is the SDE dynamics?" --index repo.index --top-k 5

# Interactive mode
$ lcs interactive --index repo.index
>>> where is the normalizing flow?
>>> show me the continual learning code
```

**Output Format**:
```
Results for: "where is the SDE dynamics?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] latent_drift_trajectory.py:1007-1030 (score: 0.92)
    class RaccoonDynamics(nn.Module):
        """SDE dynamics with drift and diffusion networks."""
        def __init__(self, latent_dim, hidden_dim, ...):
            self.drift_net = nn.Sequential(...)
            self.log_diffusion_net = nn.Sequential(...)

        def forward(self, z, t) -> tuple[Tensor, Tensor]:
            drift = self.drift_net(torch.cat([z, t], dim=1))
            ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2] latent_drift_trajectory.py:1033-1068 (score: 0.87)
    def solve_sde(dynamics, z0, t_span):
        """Euler-Maruyama SDE solver."""
        ...
```

### Phase 6: Demo on This Repository
**Test Queries**:
1. "where is the SDE dynamics implemented?"
2. "find the normalizing flow code"
3. "show me the continual learning memory buffer"
4. "how does the encoder work?"
5. "where is gradient clipping added?"

**Expected Performance** (target):
- Indexing: ~1s per 1000 lines of code
- Query: <100ms on CPU
- Accuracy: >80% precision@5 vs manual search

### Phase 7: Package & Tutorial
**Deliverables**:
1. PyPI package: `latent-code-search`
2. Tutorial notebook: `demo_codebase_search.ipynb`
3. Documentation: How trajectory planning improves search
4. Comparison: vs grep, vs embedding search

## Novel Contributions

### 1. Trajectory-Based Ranking
Unlike static embeddings, we use the **entire planning trajectory**:
- Early timesteps: syntactic features
- Middle timesteps: semantic concepts
- Late timesteps: high-level purpose

### 2. Continual Indexing
Don't retrain on every code change:
- Add new files: just train on new file + replay some memory
- Edit file: fine-tune on changes + experience replay
- No catastrophic forgetting thanks to Raccoon memory

### 3. Query Planning
Model can "plan" how to search:
- Different queries may follow different trajectories
- More complex queries → longer planning horizon
- Simple queries → short trajectory

## Next Steps

1. ✅ **NOW**: Run baseline test to get performance metrics
2. **Phase 2**: Implement flow matching (rectified flows)
3. **Phase 3**: Build code tokenizer and crawler
4. **Phase 4**: Adapt model for query-code similarity
5. **Phase 5**: Build CLI interface
6. **Phase 6**: Demo on this repo
7. **Phase 7**: Package and release

## Questions to Explore

1. **Trajectory length**: How many SDE steps optimal for code search?
2. **Flow matching benefit**: Does it improve search quality?
3. **Memory size**: How many code chunks to store for good coverage?
4. **CPU optimization**: Can we use quantization, pruning, or distillation?

---

**This is what makes latent trajectory planning exciting**: We're not just matching keywords or static embeddings—we're actually **planning a search through conceptual space**! 🚀
