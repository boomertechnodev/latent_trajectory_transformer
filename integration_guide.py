"""
INTEGRATION GUIDE: Applying Sequence Modeling Improvements to latent_drift_trajectory.py
=========================================================================================

This file demonstrates how to integrate the sequence modeling improvements
into the existing latent_drift_trajectory.py codebase.

Author: Sequence Modeling Specialist Agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Import the improvements
from sequence_modeling_improvements import (
    AdvancedSampler,
    BeamSearchDecoder,
    ImprovedDiscreteObservation,
    SequenceMetrics
)


# ============================================================================
# MODIFIED DISCRETE OBSERVATION WITH MINIMAL CHANGES
# ============================================================================

class EnhancedDiscreteObservation(nn.Module):
    """
    Drop-in replacement for DiscreteObservation with sampling improvements.
    Maintains backward compatibility while adding new features.
    """

    def __init__(
        self,
        latent_size: int,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        nb_heads: int = 4,
        dropout: float = 0.0,
        # New parameters with defaults for backward compatibility
        enable_scheduled_sampling: bool = False,
        schedule_type: str = "linear",
        schedule_k: float = 5000.0,
        num_layers: int = 1  # Default to 1 for compatibility
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Token embeddings (same as original)
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.latent_proj = nn.Linear(latent_size, hidden_size)
        self.token_proj = nn.Linear(embed_size, hidden_size)

        # Use existing AddPositionalEncoding from original code
        # self.pos_enc = AddPositionalEncoding(len_max=1e5)

        # For compatibility, we'll use simple sinusoidal encoding
        self.register_buffer('pos_enc', self._create_sinusoidal_encoding(1000, hidden_size))

        # Transformer blocks (support multiple layers)
        self.blocks = nn.ModuleList([
            self._create_transformer_block(hidden_size, nb_heads, dropout)
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Linear(hidden_size, vocab_size)

        # Scheduled sampling support
        self.enable_scheduled_sampling = enable_scheduled_sampling
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k
        self.training_step = 0

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _create_transformer_block(self, d_model: int, n_heads: int, dropout: float):
        """Create a transformer block similar to original TransformerBlock."""
        return nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'ffn': nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
        })

    def get_scheduled_sampling_prob(self) -> float:
        """Get probability for scheduled sampling."""
        if not self.enable_scheduled_sampling:
            return 0.0

        step = self.training_step
        if self.schedule_type == "linear":
            return min(1.0, step / self.schedule_k)
        elif self.schedule_type == "exponential":
            return 1.0 - math.exp(-step / self.schedule_k)
        elif self.schedule_type == "inverse_sigmoid":
            return self.schedule_k / (self.schedule_k + math.exp(step / self.schedule_k))
        return 0.0

    def get_logits(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compatible with original interface, with optional scheduled sampling.

        Args:
            z: Latent trajectory (batch, seq_len, latent_dim)
            tokens: Target tokens (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Handle scheduled sampling during training
        if self.training and self.enable_scheduled_sampling:
            sample_prob = self.get_scheduled_sampling_prob()
            self.training_step += 1

            # Progressive generation with scheduled sampling
            current_tokens = tokens.clone()
            all_logits = []

            for t in range(seq_len):
                # Get logits up to current position
                logits_t = self._forward_impl(z[:, :t+1], current_tokens[:, :t+1])
                all_logits.append(logits_t[:, -1:, :])

                if t < seq_len - 1 and torch.rand(1).item() < sample_prob:
                    # Sample from model
                    probs = F.softmax(logits_t[:, -1, :], dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    current_tokens[:, t+1] = sampled
                else:
                    # Use ground truth
                    current_tokens[:, t+1] = tokens[:, t+1] if t < seq_len - 1 else 0

            return torch.cat(all_logits, dim=1)

        else:
            # Standard forward pass (teacher forcing or inference)
            return self._forward_impl(z, tokens)

    def _forward_impl(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Internal forward implementation.
        """
        batch_size, seq_len = tokens.shape

        # Shift tokens right
        tokens_in = tokens.roll(1, dims=1)
        tokens_in[:, 0] = 0  # Start token

        # Embeddings
        tok_emb = self.token_emb(tokens_in)
        h = self.latent_proj(z) + self.token_proj(tok_emb)

        # Add positional encoding
        h = h + self.pos_enc[:, :seq_len, :h.size(-1)]

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device),
            diagonal=1
        ).bool()

        # Pass through transformer blocks
        for block in self.blocks:
            # Self-attention with causal mask
            h_norm = block['norm1'](h)
            h_attn, _ = block['self_attn'](h_norm, h_norm, h_norm,
                                          attn_mask=causal_mask)
            h = h + h_attn

            # Feed-forward
            h_norm = block['norm2'](h)
            h = h + block['ffn'](h_norm)

        # Output projection
        logits = self.proj_out(h)
        return logits

    def forward(self, z: torch.Tensor, tokens: torch.Tensor):
        """Maintain compatibility with original interface."""
        logits = self.get_logits(z, tokens)
        import torch.distributions as D
        return D.Categorical(logits=logits.reshape(-1, self.vocab_size))


# ============================================================================
# ENHANCED SAMPLING FUNCTION
# ============================================================================

def sample_sequences_ode_enhanced(
    model,  # DeterministicLatentODE
    seq_len: int,
    n_samples: int,
    device: torch.device,
    # New sampling parameters
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_beam_search: bool = False,
    beam_size: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced sampling function with advanced strategies.

    This is a drop-in replacement for sample_sequences_ode with better generation.
    """
    p_ode = model.p_ode
    p_observe = model.p_observe

    # Generate latent trajectories (same as original)
    z0_fixed = torch.randn(1, model.latent_size, device=device).repeat(n_samples, 1)
    z0_random = torch.randn(n_samples, model.latent_size, device=device)

    with torch.no_grad():
        # Fixed z0 trajectories
        from latent_drift_trajectory import solve_ode
        zs_fixed = solve_ode(p_ode, z0_fixed, 0.0, 1.0, n_steps=seq_len - 1)
        zs_fixed = zs_fixed.permute(1, 0, 2)  # (batch, seq_len, latent)

        # Random z0 trajectories
        zs_random = solve_ode(p_ode, z0_random, 0.0, 1.0, n_steps=seq_len - 1)
        zs_random = zs_random.permute(1, 0, 2)

        if use_beam_search:
            # Use beam search for higher quality
            decoder = BeamSearchDecoder(
                beam_size=beam_size,
                length_penalty=0.6,
                early_stopping=True
            )

            def forward_fn(token_ids):
                current_len = token_ids.shape[1]
                z_truncated = zs_fixed[:, :current_len, :]
                logits = p_observe.get_logits(z_truncated, token_ids)
                return logits

            tokens_fixed, _ = decoder.search(
                forward_fn,
                torch.zeros(n_samples, 1, dtype=torch.long, device=device),
                seq_len,
                eos_token_id=0,
                pad_token_id=0
            )

            # Repeat for random z
            def forward_fn_random(token_ids):
                current_len = token_ids.shape[1]
                z_truncated = zs_random[:, :current_len, :]
                logits = p_observe.get_logits(z_truncated, token_ids)
                return logits

            tokens_random, _ = decoder.search(
                forward_fn_random,
                torch.zeros(n_samples, 1, dtype=torch.long, device=device),
                seq_len,
                eos_token_id=0,
                pad_token_id=0
            )

        else:
            # Use sampling strategies
            tokens_fixed = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)
            tokens_random = torch.zeros(n_samples, seq_len, dtype=torch.long, device=device)

            for t in range(seq_len):
                # Fixed z sampling
                logits_fixed = p_observe.get_logits(zs_fixed, tokens_fixed)[:, t, :]

                tokens_fixed[:, t] = AdvancedSampler.sample_with_strategies(
                    logits_fixed,
                    generated_ids=tokens_fixed[:, :t] if t > 0 else None,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True
                ).squeeze(-1)

                # Random z sampling
                logits_random = p_observe.get_logits(zs_random, tokens_random)[:, t, :]

                tokens_random[:, t] = AdvancedSampler.sample_with_strategies(
                    logits_random,
                    generated_ids=tokens_random[:, :t] if t > 0 else None,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True
                ).squeeze(-1)

    return tokens_fixed, tokens_random


# ============================================================================
# MODIFIED TRAINING LOOP
# ============================================================================

def train_ode_enhanced(
    model,  # DeterministicLatentODE with enhanced decoder
    dataloader,
    n_iter: int,
    device: torch.device,
    loss_weights: tuple = (1.0, 0.05, 1.0),
    # New parameters
    use_scheduled_sampling: bool = True,
    log_metrics_interval: int = 100
):
    """
    Enhanced training loop with scheduled sampling and better metrics.
    """
    from tqdm import trange

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Metrics tracking
    metrics_history = {
        'loss': [],
        'perplexity': [],
        'recon_loss': [],
        'ep_loss': [],
        'ode_loss': []
    }

    pbar = trange(n_iter)
    data_iter = iter(dataloader)

    initial_ep = 0.0005
    final_ep = loss_weights[1]
    warmup_steps = 10000

    for step in pbar:
        try:
            tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            tokens = next(data_iter)

        tokens = tokens.to(device)

        # Warmup for EP term
        if step < warmup_steps:
            interp = step / warmup_steps
            current_ep = initial_ep + interp * (final_ep - initial_ep)
        else:
            current_ep = final_ep

        weights = (loss_weights[0], current_ep, loss_weights[2])

        model.train()

        # Enable scheduled sampling if decoder supports it
        if hasattr(model.p_observe, 'enable_scheduled_sampling'):
            model.p_observe.enable_scheduled_sampling = use_scheduled_sampling

        loss, loss_dict = model(tokens, loss_weights=weights)

        optim.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()

        # Track metrics
        metrics_history['loss'].append(loss.item())
        metrics_history['recon_loss'].append(loss_dict['recon'].item())
        metrics_history['ep_loss'].append(loss_dict['latent_ep'].item())
        metrics_history['ode_loss'].append(loss_dict['ode_reg'].item())

        # Compute perplexity periodically
        if step % log_metrics_interval == 0:
            model.eval()
            with torch.no_grad():
                z = model.encoder(tokens)
                logits = model.p_observe.get_logits(z, tokens)
                perplexity = SequenceMetrics.compute_perplexity(logits, tokens)
                metrics_history['perplexity'].append(perplexity)

            # Enhanced sampling with multiple strategies
            n_samples = 8
            seq_len = tokens.shape[1]

            # Try different sampling strategies
            print(f"\n[Step {step}] Testing different sampling strategies:")

            strategies = [
                {"temperature": 0.8, "top_k": 40, "name": "Conservative"},
                {"temperature": 1.0, "top_p": 0.95, "name": "Nucleus"},
                {"use_beam_search": True, "beam_size": 3, "name": "Beam"}
            ]

            for strategy in strategies:
                name = strategy.pop("name")
                samples_fixed, samples_random = sample_sequences_ode_enhanced(
                    model,
                    seq_len=seq_len,
                    n_samples=2,
                    device=device,
                    **strategy
                )

                # Decode and print one sample
                from latent_drift_trajectory import decode
                print(f"\n{name} sampling:")
                print(f"  Fixed z: {decode(samples_fixed[0].cpu())}")
                print(f"  Random z: {decode(samples_random[0].cpu())}")

            # Compute diversity metrics
            all_samples = torch.cat([samples_fixed, samples_random], dim=0)
            diversity = SequenceMetrics.compute_diversity_metrics([all_samples])
            self_bleu = SequenceMetrics.compute_self_bleu(
                [s for s in all_samples]
            )

            desc = (
                f"Loss: {loss.item():.4f} | "
                f"PPL: {perplexity:.2f} | "
                f"Div-1: {diversity['distinct_1']:.3f} | "
                f"BLEU: {self_bleu:.3f}"
            )
        else:
            desc = (
                f"Loss: {loss.item():.4f} | "
                f"Rec: {loss_dict['recon']:.3f} | "
                f"EP: {loss_dict['latent_ep']:.3f} | "
                f"ODE: {loss_dict['ode_reg']:.3f}"
            )

        pbar.set_description(desc)

    return metrics_history


# ============================================================================
# INTEGRATION EXAMPLE: MINIMAL CHANGES TO ORIGINAL
# ============================================================================

def integrate_improvements_minimal():
    """
    Minimal integration example - shows how to add improvements
    with minimal changes to the original codebase.
    """
    import sys
    import os

    # Add path to find original module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import original components
    from latent_drift_trajectory import (
        DeterministicLatentODE,
        SyntheticTargetDataset,
        vocab_size,
        decode
    )
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("MINIMAL INTEGRATION EXAMPLE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dataset (same as original)
    ds = SyntheticTargetDataset(n_samples=100_000)
    dataloader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

    # Create model (same as original)
    model = DeterministicLatentODE(
        vocab_size=vocab_size,
        latent_size=64,
        hidden_size=128,
        embed_size=64,
        num_slices=1024
    ).to(device)

    # MODIFICATION 1: Replace decoder with enhanced version
    print("\n1. Replacing decoder with enhanced version...")
    model.p_observe = EnhancedDiscreteObservation(
        latent_size=64,
        vocab_size=vocab_size,
        embed_size=64,
        hidden_size=128,
        nb_heads=4,
        dropout=0.1,  # Add dropout
        enable_scheduled_sampling=True,  # Enable scheduled sampling
        schedule_type="linear",
        schedule_k=5000.0,
        num_layers=2  # Use 2 layers instead of 1
    ).to(device)

    print(f"   - Enhanced decoder created with scheduled sampling")
    print(f"   - Using 2 transformer layers with dropout=0.1")

    # MODIFICATION 2: Use enhanced training loop
    print("\n2. Training with enhanced loop...")
    train_ode_enhanced(
        model,
        dataloader,
        n_iter=100,  # Short demo
        device=device,
        use_scheduled_sampling=True,
        log_metrics_interval=50
    )

    # MODIFICATION 3: Use enhanced sampling
    print("\n3. Generating with advanced sampling strategies...")

    model.eval()
    seq_len = 64
    n_samples = 4

    # Compare different strategies
    print("\nComparing generation strategies:")
    print("-" * 40)

    # Original sampling (for comparison)
    from latent_drift_trajectory import sample_sequences_ode
    orig_fixed, orig_random = sample_sequences_ode(model, seq_len, n_samples, device)
    print("Original sampling:")
    print(f"  {decode(orig_fixed[0].cpu())}")

    # Enhanced sampling with temperature
    enh_fixed, enh_random = sample_sequences_ode_enhanced(
        model, seq_len, n_samples, device,
        temperature=0.8,
        top_k=50
    )
    print("\nTop-k sampling (k=50, T=0.8):")
    print(f"  {decode(enh_fixed[0].cpu())}")

    # Nucleus sampling
    nuc_fixed, nuc_random = sample_sequences_ode_enhanced(
        model, seq_len, n_samples, device,
        temperature=0.9,
        top_p=0.95
    )
    print("\nNucleus sampling (p=0.95, T=0.9):")
    print(f"  {decode(nuc_fixed[0].cpu())}")

    # Beam search
    beam_fixed, beam_random = sample_sequences_ode_enhanced(
        model, seq_len, n_samples, device,
        use_beam_search=True,
        beam_size=3
    )
    print("\nBeam search (size=3):")
    print(f"  {decode(beam_fixed[0].cpu())}")

    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    integrate_improvements_minimal()