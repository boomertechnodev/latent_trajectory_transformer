# Transformer Architecture Optimization - Advanced Skill Module

## Mathematical Foundations

### Attention as Kernel Methods

**Generalized Attention Framework:**

The attention mechanism can be viewed through the lens of kernel methods:

1. **Standard Dot-Product Attention:**
   ```
   Attention(Q,K,V) = softmax(QK^T/√d)V
   ```
   Corresponds to kernel: k(q,k) = exp(⟨q,k⟩/√d)

2. **Generalized Kernel Attention:**
   ```
   Attention(Q,K,V) = D^(-1)φ(Q)φ(K)^T V
   ```
   Where φ is a feature map, D is normalization

3. **Linear Attention via Random Features:**
   ```
   φ(x) = exp(ωx - ||ω||²/2) for ω ~ N(0,I)
   ```
   Approximates RBF kernel with O(nd) complexity

### Information-Theoretic View of Attention

**Attention as Information Bottleneck:**

1. **Mutual Information Maximization:**
   ```
   max I(Output; Value) - β·I(Output; Query)
   ```
   Balance between preserving value information and compressing query

2. **Entropy Regularization:**
   ```
   H(Attention Weights) = -Σᵢⱼ αᵢⱼ log αᵢⱼ
   ```
   Higher entropy = more uniform attention

3. **Rank Analysis:**
   ```
   rank(Attention Matrix) ≤ min(sequence_length, d_model)
   ```
   Low-rank attention leads to information bottleneck

### Gradient Flow Analysis

**Deep Transformer Gradient Dynamics:**

1. **Gradient Norm Through Layers:**
   ```
   ||∂L/∂x_l|| = ||∂L/∂x_L|| · ∏ᵢ₌ₗ₊₁^L ||J_i||
   ```
   Where J_i is Jacobian of layer i

2. **Residual Connection Effect:**
   ```
   x_{l+1} = x_l + F(x_l)
   ∂x_{l+1}/∂x_l = I + ∂F/∂x_l
   ```
   Ensures gradient magnitude ≥ 1

3. **Layer Normalization Impact:**
   ```
   γ(x) = (x - μ)/σ
   ||∂γ/∂x|| ≈ 1/σ (orthogonal to mean direction)
   ```
   Stabilizes gradient magnitudes

## Advanced Implementation Techniques

### 1. State-of-the-Art Attention Variants

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import triton
import triton.language as tl

class FlashAttention2(nn.Module):
    """
    Flash Attention 2.0 with improved work partitioning.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        window_size: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.window_size = window_size

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(self, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        # Flash Attention forward
        if cu_seqlens is not None:
            # Variable length sequences
            out = flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens,
                max_seqlen=N,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.causal,
                window_size=self.window_size
            )
        else:
            # Fixed length sequences
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.causal,
                window_size=self.window_size
            )

        out = out.reshape(B, N, C)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """
    Linear complexity attention using kernel feature maps.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feature_dim: int = 64,
        eps: float = 1e-6,
        kernel: str = 'elu'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim
        self.eps = eps

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feature map
        if kernel == 'elu':
            self.feature_map = lambda x: F.elu(x) + 1
        elif kernel == 'relu':
            self.feature_map = lambda x: F.relu(x)
        elif kernel == 'squared_relu':
            self.feature_map = lambda x: F.relu(x) ** 2

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, N, D)

        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Compute KV (D x D matrix per head)
        if mask is not None:
            k = k * mask.unsqueeze(1).unsqueeze(-1)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)

        # Compute normalization
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + self.eps)

        # Compute output
        out = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention with hierarchical pooling.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        scales: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.n_scales = len(scales)

        # Scale-specific projections
        self.scale_heads = n_heads // self.n_scales
        assert n_heads % self.n_scales == 0

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Process each scale
        outputs = []
        for i, scale in enumerate(self.scales):
            start_head = i * self.scale_heads
            end_head = (i + 1) * self.scale_heads

            q_scale = q[:, start_head:end_head]
            k_scale = k[:, start_head:end_head]
            v_scale = v[:, start_head:end_head]

            if scale > 1:
                # Pooling for keys and values
                k_scale = F.avg_pool1d(
                    k_scale.flatten(1, 2).transpose(-1, -2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(-1, -2).reshape(B, self.scale_heads, -1, C // self.n_heads)

                v_scale = F.avg_pool1d(
                    v_scale.flatten(1, 2).transpose(-1, -2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(-1, -2).reshape(B, self.scale_heads, -1, C // self.n_heads)

            # Compute attention
            scores = torch.matmul(q_scale, k_scale.transpose(-2, -1)) / math.sqrt(C // self.n_heads)

            if mask is not None and scale == 1:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out_scale = torch.matmul(attn, v_scale)
            outputs.append(out_scale)

        # Concatenate scales
        output = torch.cat(outputs, dim=1)
        output = output.transpose(1, 2).reshape(B, N, C)

        return self.out_proj(output)
```

### 2. Advanced Positional Encoding Techniques

```python
class XPos(nn.Module):
    """
    Extrapolatable Position Embedding (xPos).
    Combines RoPE with exponential decay for length extrapolation.
    """
    def __init__(
        self,
        head_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        scale_base: float = 512.0
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.scale_base = scale_base

        # RoPE frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Exponential decay for extrapolation
        self.register_buffer(
            'scale',
            (torch.arange(0, max_position, 1.0) + scale_base) / scale_base
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        seq_len = q.shape[2]

        # Compute RoPE
        t = torch.arange(seq_len, device=q.device) + offset
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        cos = freqs.cos()
        sin = freqs.sin()

        # Apply rotation with scaling
        scale = self.scale[offset:offset + seq_len]
        scale = scale.unsqueeze(-1)

        q_rot = self.apply_rotary_emb(q, cos, sin)
        k_rot = self.apply_rotary_emb(k, cos, sin)

        # Apply exponential decay to keys
        k_rot = k_rot / scale.unsqueeze(0).unsqueeze(0)

        return q_rot, k_rot

    @staticmethod
    def apply_rotary_emb(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional encoding with interpolation for variable lengths.
    """
    def __init__(
        self,
        d_model: int,
        max_position: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_position = max_position

        self.embeddings = nn.Parameter(torch.randn(max_position, d_model))
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        if N <= self.max_position:
            # Direct indexing
            pos_emb = self.embeddings[:N]
        else:
            # Interpolation for longer sequences
            pos_emb = F.interpolate(
                self.embeddings.unsqueeze(0).transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2).squeeze(0)

        return self.dropout(x + pos_emb)


class ConditionalPositionalEncoding(nn.Module):
    """
    Conditional positional encoding that adapts based on content.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_position: int = 8192
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Content-dependent position computation
        self.pos_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads)  # Per-head position bias
        )

        # Learnable relative position embeddings
        self.rel_pos_emb = nn.Parameter(
            torch.randn(2 * max_position - 1, n_heads)
        )

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        # Compute content-dependent positions
        pos_weights = self.pos_net(x)  # (B, N, H)

        # Create relative position matrix
        positions = torch.arange(N, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos + (N - 1)  # Shift to positive indices

        # Get relative position embeddings
        rel_emb = self.rel_pos_emb[rel_pos]  # (N, N, H)

        # Combine with content-dependent weights
        pos_bias = torch.einsum('bnh,nnh->bhn', pos_weights, rel_emb)

        return pos_bias.unsqueeze(2)  # (B, H, 1, N)
```

### 3. FFN Variants and Gating Mechanisms

```python
class SwiGLU(nn.Module):
    """
    SwiGLU activation function from GLU variants paper.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN with top-k routing.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(d_model, n_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(n_experts)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        # Compute routing weights
        router_logits = self.router(x)  # (B, N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        # Process through experts
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # (B, N)
            gate = top_k_gates[..., k:k+1]  # (B, N, 1)

            # Gather samples for each expert
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += gate[mask] * expert_output

        return self.dropout(output)
```

### 4. Training Stability and Optimization

```python
class StableTransformer(nn.Module):
    """
    Transformer with multiple stability improvements.
    """
    def __init__(self, config):
        super().__init__()

        # Embedding with scaled initialization
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embed.weight, std=config.d_model ** -0.5)

        # Positional encoding
        self.pos_enc = XPos(config.d_model // config.n_heads)

        # Transformer blocks with careful initialization
        self.blocks = nn.ModuleList([
            self._init_block(config, layer_idx)
            for layer_idx in range(config.n_layers)
        ])

        # Output with small initialization
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-6)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Apply special scaled init to output
        self.apply(self._init_weights)

        # Scale down residual projections
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                std=0.02 / math.sqrt(2 * config.n_layers)
            )

    def _init_block(self, config, layer_idx):
        """Initialize a transformer block with layer-dependent scaling."""
        block = OptimizedTransformerBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout * (layer_idx / config.n_layers),  # Layer-wise dropout
            activation='swiglu',
            norm_type='pre'
        )
        return block

    def _init_weights(self, module):
        """Custom weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        # Embeddings with gradient scaling
        x = self.embed(input_ids)
        x = x * math.sqrt(self.config.d_model)

        # Process through blocks with gradient checkpointing
        for i, block in enumerate(self.blocks):
            if self.training and i > len(self.blocks) // 2:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x, _ = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1  # Improves generalization
            )

        return logits, loss
```

## Debugging Strategies

### Common Issues and Solutions

1. **Attention Pattern Collapse**
   ```python
   def diagnose_attention_collapse(model, dataloader):
       """Check if attention patterns are collapsing to single positions."""
       attention_entropy = []

       for batch in dataloader:
           with torch.no_grad():
               # Hook to capture attention weights
               attn_weights = []

               def hook_fn(module, input, output):
                   attn_weights.append(output[1])  # Attention weights

               hooks = []
               for block in model.blocks:
                   hooks.append(block.attn.register_forward_hook(hook_fn))

               model(batch)

               for hook in hooks:
                   hook.remove()

           # Compute entropy
           for weights in attn_weights:
               entropy = -(weights * weights.log()).sum(-1).mean()
               attention_entropy.append(entropy.item())

       return np.mean(attention_entropy)
   ```

2. **Gradient Vanishing/Explosion**
   ```python
   def monitor_gradient_flow(model):
       """Monitor gradient norms through layers."""
       grad_norms = {}

       for name, param in model.named_parameters():
           if param.grad is not None:
               grad_norms[name] = param.grad.norm().item()

       # Check for issues
       max_grad = max(grad_norms.values())
       min_grad = min(grad_norms.values())

       if max_grad > 100:
           print(f"Warning: Gradient explosion detected (max={max_grad})")
       if min_grad < 1e-6:
           print(f"Warning: Gradient vanishing detected (min={min_grad})")

       return grad_norms
   ```

3. **Position Encoding Failures**
   ```python
   def test_positional_encoding(pos_encoding, max_len=10000):
       """Test if positional encoding maintains necessary properties."""
       # Test different sequence lengths
       for seq_len in [10, 100, 1000, max_len]:
           x = torch.randn(1, seq_len, pos_encoding.d_model)

           # Check if positions are distinguishable
           pos_emb = pos_encoding(x)
           similarity = torch.cosine_similarity(
               pos_emb[0].unsqueeze(1),
               pos_emb[0].unsqueeze(0),
               dim=-1
           )

           # Positions should have decreasing similarity with distance
           for i in range(min(10, seq_len)):
               assert similarity[i, i] == 1.0  # Self-similarity
               if i > 0:
                   assert similarity[0, i] < similarity[0, i-1]  # Monotonic decay
   ```

## Performance Optimization

### Memory-Efficient Training
```python
def setup_efficient_training(model, config):
    """Configure model for memory-efficient training."""
    # Gradient accumulation
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size

    # Mixed precision
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()

    # Gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer with memory efficiency
    from bitsandbytes.optim import AdamW8bit
    optimizer = AdamW8bit(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    return optimizer, scaler, gradient_accumulation_steps
```

### Inference Optimization
```python
@torch.inference_mode()
def optimized_generate(model, prompt, max_length=100):
    """Optimized generation with KV-cache."""
    model.eval()

    # Initialize KV-cache
    past_kv = None

    tokens = prompt
    for _ in range(max_length):
        # Only process new tokens
        if past_kv is not None:
            input_tokens = tokens[-1:]
        else:
            input_tokens = tokens

        logits, past_kv = model(input_tokens, past_kv=past_kv, use_cache=True)

        # Sample next token
        next_token = torch.multinomial(
            F.softmax(logits[0, -1] / temperature, dim=-1), 1
        )

        tokens = torch.cat([tokens, next_token])

        if next_token == eos_token:
            break

    return tokens
```

## Literature References

### Foundational Papers
- **Attention is All You Need**: Vaswani et al. (2017)
- **BERT**: Devlin et al. (2019) - Bidirectional pretraining
- **GPT**: Radford et al. (2018, 2019) - Autoregressive pretraining
- **T5**: Raffel et al. (2020) - Text-to-text unified framework

### Efficiency Improvements
- **Flash Attention**: Dao et al. (2022) - IO-aware exact attention
- **Flash Attention 2**: Dao (2023) - Improved work partitioning
- **Linear Attention**: Katharopoulos et al. (2020) - Kernel feature maps
- **Linformer**: Wang et al. (2020) - Low-rank factorization

### Positional Encodings
- **RoPE**: Su et al. (2021) - Rotary position embedding
- **ALiBi**: Press et al. (2021) - Attention with linear biases
- **xPos**: Sun et al. (2023) - Extrapolatable position embedding

### Architecture Innovations
- **Pre-LN**: Xiong et al. (2020) - Pre-normalization for stability
- **ReZero**: Bachlechner et al. (2021) - Zero-initialized residuals
- **SwiGLU**: Shazeer (2020) - GLU variants improve transformers

## Quick Reference Tables

### Attention Complexity Comparison

| Method | Time Complexity | Memory Complexity | Quality |
|--------|----------------|-------------------|---------|
| Full Attention | O(n²d) | O(n²) | Exact |
| Flash Attention | O(n²d) | O(n) | Exact |
| Linear Attention | O(nd²) | O(d²) | Approximate |
| Local Attention | O(nwd) | O(nw) | Local only |
| Sparse Attention | O(n√n·d) | O(n√n) | Approximate |

### Position Encoding Properties

| Method | Extrapolation | Learnable | Relative | Complexity |
|--------|--------------|-----------|----------|------------|
| Sinusoidal | Good | No | No | O(1) |
| Learned | Poor | Yes | No | O(n) |
| RoPE | Good | No | Yes | O(1) |
| ALiBi | Excellent | No | Yes | O(1) |
| T5 Bias | Good | Yes | Yes | O(n²) |

### Initialization Guidelines

| Component | Initialization | Std Dev |
|-----------|---------------|---------|
| Embeddings | Normal | 1/√d_model |
| QKV Projections | Xavier | √(2/d_model) |
| Output Projections | Normal | 0.02/√(2L) |
| FFN | Xavier | √(2/d_ff) |
| LayerNorm | γ=1, β=0 | - |

Remember: Transformer architecture is both art and science. Every design choice affects expressivity, efficiency, and trainability. Build with intention, optimize with measurement, and always validate empirically.