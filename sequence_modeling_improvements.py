"""
SEQUENCE MODELING IMPROVEMENTS FOR LATENT TRAJECTORY TRANSFORMER
================================================================

Expert analysis and implementations by the Sequence Modeling Specialist Agent.
These improvements address exposure bias, sampling quality, and generation diversity.

Author: Sequence Modeling Agent
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# ============================================================================
# PART 1: ADVANCED SAMPLING STRATEGIES
# ============================================================================

class AdvancedSampler:
    """
    Collection of state-of-the-art sampling strategies for autoregressive generation.

    Key innovations:
    - Temperature-controlled sampling for diversity tuning
    - Top-k filtering to avoid low-probability tail
    - Nucleus (top-p) sampling for dynamic vocabulary size
    - Repetition penalty to prevent loops
    - Beam search for high-quality generation
    """

    @staticmethod
    def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Temperature controls the trade-off between:
        - Low temp (< 1.0): More conservative, higher quality, less diverse
        - High temp (> 1.0): More diverse, potentially lower quality

        Args:
            logits: Raw model outputs (batch, vocab_size)
            temperature: Scaling factor (typical range: 0.5 to 1.5)

        Returns:
            Scaled logits
        """
        # Avoid division by zero
        if temperature < 1e-5:
            temperature = 1e-5

        return logits / temperature

    @staticmethod
    def top_k_filtering(
        logits: torch.Tensor,
        k: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Filter logits to only top-k tokens, then sample.

        Top-k sampling prevents the model from selecting very unlikely tokens
        that could derail generation quality.

        Args:
            logits: Raw model outputs (batch, vocab_size)
            k: Number of top tokens to consider
            temperature: Temperature for diversity control

        Returns:
            Filtered logits with -inf for tokens outside top-k
        """
        batch_size, vocab_size = logits.shape

        # Apply temperature first
        logits = AdvancedSampler.apply_temperature(logits, temperature)

        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

        # Create mask for tokens not in top-k
        # This is more numerically stable than setting to -inf directly
        min_value = top_k_values[:, -1:].expand_as(logits)
        logits = torch.where(logits < min_value,
                             torch.full_like(logits, -float('inf')),
                             logits)

        return logits

    @staticmethod
    def nucleus_filtering(
        logits: torch.Tensor,
        p: float = 0.95,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Nucleus (top-p) sampling: sample from smallest set whose cumulative probability > p.

        This is often superior to top-k as it adapts the vocabulary size dynamically
        based on the confidence of the model.

        Args:
            logits: Raw model outputs (batch, vocab_size)
            p: Cumulative probability threshold (typical: 0.9-0.95)
            temperature: Temperature for diversity control

        Returns:
            Filtered logits
        """
        # Apply temperature
        logits = AdvancedSampler.apply_temperature(logits, temperature)

        # Sort in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find cutoff index where cumulative probability exceeds p
        # We want to include tokens until cumulative prob > p
        cutoff_mask = cumulative_probs > p

        # Shift mask to include the token that crosses threshold
        cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
        cutoff_mask[:, 0] = False

        # Set filtered positions to -inf
        sorted_logits[cutoff_mask] = -float('inf')

        # Restore original order
        # Create scatter indices
        batch_indices = torch.arange(logits.shape[0]).unsqueeze(-1).expand_as(sorted_indices)
        original_logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        return original_logits

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Apply repetition penalty to discourage repeating already generated tokens.

        Based on CTRL paper: https://arxiv.org/abs/1909.05858

        Args:
            logits: Current logits (batch, vocab_size)
            generated_ids: Previously generated token ids (batch, seq_len)
            penalty: Repetition penalty factor (> 1.0 penalizes repetition)

        Returns:
            Modified logits with repetition penalty applied
        """
        batch_size = logits.shape[0]

        # For each batch, penalize tokens that have appeared
        for b in range(batch_size):
            for token_id in generated_ids[b].unique():
                if logits[b, token_id] < 0:
                    # If logit is negative, make more negative
                    logits[b, token_id] *= penalty
                else:
                    # If logit is positive, make less positive
                    logits[b, token_id] /= penalty

        return logits

    @staticmethod
    def sample_with_strategies(
        logits: torch.Tensor,
        generated_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Combined sampling with multiple strategies.

        Args:
            logits: Model output logits (batch, vocab_size)
            generated_ids: Previously generated tokens for repetition penalty
            temperature: Temperature scaling
            top_k: If set, use top-k filtering
            top_p: If set, use nucleus filtering
            repetition_penalty: Penalty for repeating tokens
            do_sample: If False, use greedy decoding

        Returns:
            Sampled token ids (batch, 1)
        """
        # Apply repetition penalty if we have history
        if generated_ids is not None and repetition_penalty != 1.0:
            logits = AdvancedSampler.apply_repetition_penalty(
                logits, generated_ids, repetition_penalty
            )

        # Apply filtering strategies
        if top_k is not None and top_k > 0:
            logits = AdvancedSampler.top_k_filtering(logits, top_k, temperature)
        elif top_p is not None and top_p < 1.0:
            logits = AdvancedSampler.nucleus_filtering(logits, top_p, temperature)
        else:
            # Just apply temperature
            logits = AdvancedSampler.apply_temperature(logits, temperature)

        # Sample or greedy decode
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

        return next_tokens


# ============================================================================
# PART 2: BEAM SEARCH IMPLEMENTATION
# ============================================================================

class BeamSearchDecoder:
    """
    Beam search for high-quality sequence generation.

    Maintains multiple hypotheses and selects the best based on likelihood.
    Includes length normalization and diverse beam search variants.
    """

    def __init__(
        self,
        beam_size: int = 5,
        length_penalty: float = 0.6,
        early_stopping: bool = True,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0
    ):
        """
        Initialize beam search decoder.

        Args:
            beam_size: Number of beams to maintain
            length_penalty: Wu et al. (2016) length normalization factor
            early_stopping: Stop when best sequence is found
            num_beam_groups: Number of groups for diverse beam search
            diversity_penalty: Penalty for similarity within groups
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty

        assert beam_size % num_beam_groups == 0, "beam_size must be divisible by num_beam_groups"
        self.beams_per_group = beam_size // num_beam_groups

    def length_normalize(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Apply length normalization to prevent bias toward shorter sequences.

        Uses Wu et al. (2016) formula: score / ((5 + length) / 6) ^ alpha

        Args:
            scores: Cumulative log probabilities (batch * beam_size)
            lengths: Sequence lengths (batch * beam_size)

        Returns:
            Length-normalized scores
        """
        length_penalty_tensor = ((5.0 + lengths.float()) / 6.0) ** self.length_penalty
        return scores / length_penalty_tensor

    def search(
        self,
        model_forward_fn,
        initial_ids: torch.Tensor,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform beam search decoding.

        Args:
            model_forward_fn: Function that takes token ids and returns logits
            initial_ids: Starting token ids (batch_size, initial_length)
            max_length: Maximum generation length
            eos_token_id: End-of-sequence token id
            pad_token_id: Padding token id

        Returns:
            sequences: Best sequences (batch_size, max_length)
            scores: Sequence scores (batch_size,)
        """
        batch_size = initial_ids.shape[0]
        device = initial_ids.device
        vocab_size = None  # Will be inferred from first forward pass

        # Expand for beam search
        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
        beam_scores[:, 1:] = -float('inf')  # Only first beam is active initially

        # Initialize beams
        beam_seqs = initial_ids.unsqueeze(1).expand(-1, self.beam_size, -1)
        beam_seqs = beam_seqs.reshape(batch_size * self.beam_size, -1)

        # Track finished sequences
        finished_seqs = []
        finished_scores = []

        for step in range(max_length - initial_ids.shape[1]):
            # Get model predictions
            logits = model_forward_fn(beam_seqs)  # (batch * beam_size, vocab_size)

            if vocab_size is None:
                vocab_size = logits.shape[-1]

            # Get log probabilities
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (batch * beam, vocab)

            # Add to beam scores
            log_probs = log_probs.view(batch_size, self.beam_size, vocab_size)

            # Apply diversity penalty for diverse beam search
            if self.num_beam_groups > 1:
                for group_idx in range(self.num_beam_groups):
                    group_start = group_idx * self.beams_per_group
                    group_end = (group_idx + 1) * self.beams_per_group

                    if group_idx > 0:
                        # Penalize tokens used by previous groups
                        for prev_group in range(group_idx):
                            prev_start = prev_group * self.beams_per_group
                            prev_end = (prev_group + 1) * self.beams_per_group
                            prev_tokens = beam_seqs[prev_start:prev_end, -1]

                            # Apply diversity penalty
                            for token in prev_tokens.unique():
                                log_probs[:, group_start:group_end, token] -= self.diversity_penalty

            # Compute new scores
            new_scores = beam_scores.unsqueeze(-1) + log_probs  # (batch, beam, vocab)

            # Flatten to select top-k
            new_scores_flat = new_scores.view(batch_size, -1)  # (batch, beam * vocab)

            # Select top-k candidates
            top_scores, top_indices = torch.topk(new_scores_flat, self.beam_size, dim=-1)

            # Recover beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beams
            new_beam_seqs = []
            for b in range(batch_size):
                for k in range(self.beam_size):
                    beam_idx = beam_indices[b, k]
                    token_idx = token_indices[b, k]

                    # Get sequence from selected beam
                    seq_idx = b * self.beam_size + beam_idx
                    seq = beam_seqs[seq_idx]

                    # Append new token
                    new_seq = torch.cat([seq, token_idx.unsqueeze(0)])
                    new_beam_seqs.append(new_seq)

                    # Check for EOS
                    if token_idx == eos_token_id:
                        finished_seqs.append(new_seq)
                        finished_scores.append(top_scores[b, k])

            # Update beam sequences
            beam_seqs = torch.nn.utils.rnn.pad_sequence(new_beam_seqs, batch_first=True, padding_value=pad_token_id)
            beam_scores = top_scores

            # Early stopping check
            if self.early_stopping and len(finished_seqs) >= batch_size:
                break

        # Return best sequences
        if finished_seqs:
            # Pad finished sequences
            max_finished_len = max(len(seq) for seq in finished_seqs)
            padded_seqs = torch.full((len(finished_seqs), max_finished_len), pad_token_id, device=device)
            for i, seq in enumerate(finished_seqs):
                padded_seqs[i, :len(seq)] = seq

            # Apply length normalization to scores
            lengths = torch.tensor([len(seq) for seq in finished_seqs], device=device)
            normalized_scores = self.length_normalize(
                torch.tensor(finished_scores, device=device), lengths
            )

            # Select best for each batch
            best_indices = normalized_scores.topk(batch_size)[1]
            return padded_seqs[best_indices], normalized_scores[best_indices]
        else:
            # No sequences finished, return current beams
            return beam_seqs[:batch_size], beam_scores[:, 0]


# ============================================================================
# PART 3: SCHEDULED SAMPLING FOR EXPOSURE BIAS MITIGATION
# ============================================================================

class ScheduledSamplingMixin:
    """
    Mixin for implementing scheduled sampling during training.

    Scheduled sampling gradually transitions from teacher forcing to using
    model predictions during training, reducing exposure bias.

    Reference: Bengio et al. (2015) "Scheduled Sampling for Sequence Prediction"
    """

    def __init__(self, schedule_type: str = "linear", k: float = 2000.0):
        """
        Initialize scheduled sampling.

        Args:
            schedule_type: Type of schedule ("linear", "exponential", "inverse_sigmoid")
            k: Schedule parameter (interpretation depends on schedule_type)
        """
        self.schedule_type = schedule_type
        self.k = k
        self.training_step = 0

    def get_sampling_probability(self) -> float:
        """
        Get probability of using model prediction vs ground truth.

        Returns:
            Probability in [0, 1] where 0 = always use ground truth,
            1 = always use model prediction
        """
        step = self.training_step

        if self.schedule_type == "linear":
            # Linear increase from 0 to 1 over k steps
            return min(1.0, step / self.k)

        elif self.schedule_type == "exponential":
            # Exponential schedule: 1 - exp(-step/k)
            return 1.0 - math.exp(-step / self.k)

        elif self.schedule_type == "inverse_sigmoid":
            # Inverse sigmoid: k/(k + exp(step/k))
            return self.k / (self.k + math.exp(step / self.k))

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def scheduled_sampling_forward(
        self,
        model,
        tokens: torch.Tensor,
        z: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with scheduled sampling.

        Args:
            model: Decoder model with get_logits method
            tokens: Target tokens (batch, seq_len)
            z: Latent trajectories (batch, seq_len, latent_dim)
            training: Whether in training mode

        Returns:
            logits: Model predictions (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        if not training:
            # Regular forward pass during evaluation
            return model.get_logits(z, tokens)

        # Get sampling probability
        sample_prob = self.get_sampling_probability()

        # Initialize with start token
        current_tokens = tokens.clone()
        current_tokens[:, 1:] = 0  # Will be filled progressively
        current_tokens[:, 0] = tokens[:, 0]  # Keep start token

        all_logits = []

        for t in range(seq_len):
            # Get logits up to current position
            logits = model.get_logits(z[:, :t+1], current_tokens[:, :t+1])
            current_logits = logits[:, t:t+1, :]  # (batch, 1, vocab)
            all_logits.append(current_logits)

            if t < seq_len - 1:  # Don't sample for last position
                # Decide whether to use model prediction or ground truth
                use_sample = torch.rand(1).item() < sample_prob

                if use_sample:
                    # Sample from model prediction
                    probs = F.softmax(current_logits.squeeze(1), dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    current_tokens[:, t+1] = sampled
                else:
                    # Use ground truth (teacher forcing)
                    current_tokens[:, t+1] = tokens[:, t+1]

        # Concatenate all logits
        logits = torch.cat(all_logits, dim=1)  # (batch, seq_len, vocab)

        # Increment training step
        self.training_step += 1

        return logits


# ============================================================================
# PART 4: IMPROVED DISCRETE OBSERVATION MODEL
# ============================================================================

class ImprovedDiscreteObservation(nn.Module):
    """
    Enhanced autoregressive decoder with:
    - Multiple transformer layers
    - Scheduled sampling support
    - Advanced generation strategies
    - Cross-attention to latent trajectory
    """

    def __init__(
        self,
        latent_size: int,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 3,  # More layers for expressiveness
        num_heads: int = 8,   # More attention heads
        dropout: float = 0.1,  # Add dropout for regularization
        max_seq_len: int = 1000,
        schedule_type: str = "linear",
        schedule_k: float = 5000.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.positional_emb = nn.Embedding(max_seq_len, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Projection layers
        self.token_proj = nn.Linear(embed_size, hidden_size)

        # Latent cross-attention layers
        self.latent_cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Self-attention layers (causal)
        self.self_attention = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),  # GELU activation (better than ReLU for transformers)
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.ln1 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ln3 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        # Latent projection for cross-attention
        self.latent_proj = nn.Linear(latent_size, hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        # Scheduled sampling
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k
        self.training_step = 0

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_scheduled_sampling_prob(self) -> float:
        """Get probability for scheduled sampling."""
        step = self.training_step

        if self.schedule_type == "linear":
            return min(1.0, step / self.schedule_k)
        elif self.schedule_type == "exponential":
            return 1.0 - math.exp(-step / self.schedule_k)
        elif self.schedule_type == "inverse_sigmoid":
            return self.schedule_k / (self.schedule_k + math.exp(step / self.schedule_k))
        else:
            return 0.0  # Default to teacher forcing

    def forward_step(
        self,
        tokens: torch.Tensor,
        z: torch.Tensor,
        use_scheduled_sampling: bool = False
    ) -> torch.Tensor:
        """
        Single forward pass through the decoder.

        Args:
            tokens: Input tokens (batch, seq_len)
            z: Latent trajectory (batch, seq_len, latent_dim)
            use_scheduled_sampling: Whether to use scheduled sampling

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Token embeddings
        token_emb = self.token_emb(tokens)  # (batch, seq_len, embed_size)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.positional_emb(positions)

        # Combine embeddings and project
        x = self.dropout(token_emb + pos_emb)
        x = self.token_proj(x)  # (batch, seq_len, hidden_size)

        # Project latent trajectory for cross-attention
        z_proj = self.latent_proj(z)  # (batch, seq_len, hidden_size)

        # Create causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_len, device)

        # Pass through transformer layers
        for i in range(self.num_layers):
            # Self-attention with causal mask
            residual = x
            x = self.ln1[i](x)
            x, _ = self.self_attention[i](x, x, x, attn_mask=causal_mask)
            x = self.dropout(x)
            x = residual + x

            # Cross-attention to latent trajectory
            residual = x
            x = self.ln2[i](x)
            x, _ = self.latent_cross_attention[i](x, z_proj, z_proj)
            x = self.dropout(x)
            x = residual + x

            # Feed-forward network
            residual = x
            x = self.ln3[i](x)
            x = self.ffn[i](x)
            x = self.dropout(x)
            x = residual + x

        # Output projection
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        return logits

    def get_logits(
        self,
        z: torch.Tensor,
        tokens: torch.Tensor,
        use_scheduled_sampling: bool = False
    ) -> torch.Tensor:
        """
        Get logits with optional scheduled sampling during training.

        Args:
            z: Latent trajectory (batch, seq_len, latent_dim)
            tokens: Target tokens (batch, seq_len)
            use_scheduled_sampling: Enable scheduled sampling in training

        Returns:
            logits: Model predictions (batch, seq_len, vocab_size)
        """
        if not self.training or not use_scheduled_sampling:
            # Standard teacher forcing
            return self.forward_step(tokens, z, use_scheduled_sampling=False)

        # Scheduled sampling implementation
        batch_size, seq_len = tokens.shape
        device = tokens.device
        vocab_size = self.vocab_size

        # Get sampling probability
        sample_prob = self.get_scheduled_sampling_prob()

        # Initialize with start token
        current_tokens = torch.zeros_like(tokens)
        current_tokens[:, 0] = tokens[:, 0]  # Copy start token

        all_logits = []

        for t in range(seq_len):
            # Get predictions up to current position
            if t == 0:
                # First position always uses start token
                logits = self.forward_step(
                    current_tokens[:, :1],
                    z[:, :1],
                    use_scheduled_sampling=False
                )
                all_logits.append(logits)
            else:
                # Forward pass with current tokens
                logits = self.forward_step(
                    current_tokens[:, :t+1],
                    z[:, :t+1],
                    use_scheduled_sampling=False
                )
                # Take only the last position's logits
                all_logits.append(logits[:, -1:, :])

                if t < seq_len - 1:  # Don't sample for last position
                    # Decide whether to use model prediction or ground truth
                    use_sample = torch.rand(1, device=device).item() < sample_prob

                    if use_sample:
                        # Sample from model prediction
                        last_logits = logits[:, -1, :]  # (batch, vocab_size)
                        probs = F.softmax(last_logits / 0.8, dim=-1)  # Temperature for diversity
                        sampled = torch.multinomial(probs, 1).squeeze(-1)
                        current_tokens[:, t+1] = sampled
                    else:
                        # Use ground truth (teacher forcing)
                        current_tokens[:, t+1] = tokens[:, t+1]

        # Concatenate all logits
        final_logits = torch.cat(all_logits, dim=1)  # (batch, seq_len, vocab_size)

        # Increment training step for schedule
        self.training_step += 1

        return final_logits

    def generate(
        self,
        z: torch.Tensor,
        max_length: int,
        start_token_id: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0,
        use_beam_search: bool = False,
        beam_size: int = 5
    ) -> torch.Tensor:
        """
        Generate sequences using advanced sampling strategies.

        Args:
            z: Latent trajectory (batch, latent_seq_len, latent_dim)
            max_length: Maximum generation length
            start_token_id: Starting token
            eos_token_id: End-of-sequence token (optional)
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            use_beam_search: Whether to use beam search
            beam_size: Beam size for beam search

        Returns:
            generated: Generated token sequences (batch, max_length)
        """
        batch_size = z.shape[0]
        device = z.device

        if use_beam_search:
            # Use beam search for higher quality
            decoder = BeamSearchDecoder(
                beam_size=beam_size,
                length_penalty=0.6,
                early_stopping=True
            )

            # Create forward function for beam search
            def forward_fn(token_ids):
                with torch.no_grad():
                    # Ensure z matches token sequence length
                    current_len = token_ids.shape[1]
                    z_truncated = z[:, :current_len, :]
                    logits = self.forward_step(token_ids, z_truncated)
                return logits

            # Initialize with start token
            initial_ids = torch.full((batch_size, 1), start_token_id, device=device)

            sequences, scores = decoder.search(
                forward_fn,
                initial_ids,
                max_length,
                eos_token_id if eos_token_id is not None else start_token_id,
                pad_token_id=0
            )

            return sequences

        else:
            # Use sampling strategies
            generated = torch.full((batch_size, max_length), 0, dtype=torch.long, device=device)
            generated[:, 0] = start_token_id

            unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

            for t in range(1, max_length):
                if not unfinished.any():
                    break

                # Get logits for next position
                with torch.no_grad():
                    current_z = z[:, :t, :] if t <= z.shape[1] else z
                    logits = self.forward_step(generated[:, :t], current_z)
                    next_logits = logits[:, -1, :]  # (batch, vocab_size)

                # Apply sampling strategies
                next_tokens = AdvancedSampler.sample_with_strategies(
                    next_logits,
                    generated_ids=generated[:, :t],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True
                )

                generated[:, t] = next_tokens.squeeze(-1)

                # Check for EOS
                if eos_token_id is not None:
                    finished = (next_tokens.squeeze(-1) == eos_token_id)
                    unfinished = unfinished & ~finished

            return generated


# ============================================================================
# PART 5: EVALUATION METRICS
# ============================================================================

class SequenceMetrics:
    """
    Comprehensive metrics for evaluating sequence generation quality.
    """

    @staticmethod
    def compute_perplexity(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute perplexity of predictions.

        Perplexity = exp(average cross entropy)
        Lower is better, 1.0 is perfect.

        Args:
            logits: Model predictions (batch, seq_len, vocab_size)
            targets: Target tokens (batch, seq_len)
            mask: Optional mask for padding (batch, seq_len)

        Returns:
            perplexity: Float value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none'
        )

        # Apply mask if provided
        if mask is not None:
            ce_loss = ce_loss * mask.reshape(-1)
            num_tokens = mask.sum()
        else:
            num_tokens = batch_size * seq_len

        # Average cross entropy
        avg_ce = ce_loss.sum() / num_tokens

        # Perplexity
        perplexity = torch.exp(avg_ce)

        return perplexity.item()

    @staticmethod
    def compute_diversity_metrics(
        generated: List[torch.Tensor],
        n_grams: List[int] = [1, 2, 3]
    ) -> dict:
        """
        Compute diversity metrics for generated sequences.

        Args:
            generated: List of generated sequences
            n_grams: N-gram sizes to compute

        Returns:
            Dictionary with diversity metrics
        """
        metrics = {}

        for n in n_grams:
            all_ngrams = set()
            total_ngrams = 0

            for seq in generated:
                seq_list = seq.tolist() if isinstance(seq, torch.Tensor) else seq

                for i in range(len(seq_list) - n + 1):
                    ngram = tuple(seq_list[i:i+n])
                    all_ngrams.add(ngram)
                    total_ngrams += 1

            # Distinct n-grams ratio
            if total_ngrams > 0:
                metrics[f'distinct_{n}'] = len(all_ngrams) / total_ngrams
            else:
                metrics[f'distinct_{n}'] = 0.0

        return metrics

    @staticmethod
    def compute_self_bleu(
        generated: List[torch.Tensor],
        n_gram: int = 4
    ) -> float:
        """
        Compute Self-BLEU to measure diversity.
        Lower Self-BLEU = more diverse generation.

        Args:
            generated: List of generated sequences
            n_gram: N-gram size for BLEU

        Returns:
            Average Self-BLEU score
        """
        from collections import Counter
        import math

        def ngrams(sequence, n):
            """Extract n-grams from sequence."""
            return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

        def compute_bleu(reference, hypothesis, n):
            """Simple BLEU implementation."""
            ref_ngrams = Counter(ngrams(reference, n))
            hyp_ngrams = Counter(ngrams(hypothesis, n))

            # Count matches
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())

            if total == 0:
                return 0.0

            # Precision
            precision = matches / total

            # Brevity penalty
            ref_len = len(reference)
            hyp_len = len(hypothesis)

            if hyp_len < ref_len:
                bp = math.exp(1 - ref_len / hyp_len)
            else:
                bp = 1.0

            return bp * precision

        scores = []
        for i, seq1 in enumerate(generated):
            seq1_list = seq1.tolist() if isinstance(seq1, torch.Tensor) else seq1

            # Compare with all other sequences
            seq_scores = []
            for j, seq2 in enumerate(generated):
                if i != j:
                    seq2_list = seq2.tolist() if isinstance(seq2, torch.Tensor) else seq2
                    score = compute_bleu(seq2_list, seq1_list, n_gram)
                    seq_scores.append(score)

            if seq_scores:
                scores.append(sum(seq_scores) / len(seq_scores))

        return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# PART 6: INTEGRATION EXAMPLE
# ============================================================================

def create_improved_model(
    vocab_size: int,
    latent_size: int,
    hidden_size: int,
    embed_size: int,
    num_decoder_layers: int = 3,
    use_scheduled_sampling: bool = True
) -> ImprovedDiscreteObservation:
    """
    Factory function to create improved decoder with all enhancements.

    Args:
        vocab_size: Vocabulary size
        latent_size: Latent dimension size
        hidden_size: Hidden layer size
        embed_size: Embedding size
        num_decoder_layers: Number of transformer layers
        use_scheduled_sampling: Whether to use scheduled sampling

    Returns:
        ImprovedDiscreteObservation model
    """
    return ImprovedDiscreteObservation(
        latent_size=latent_size,
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_decoder_layers,
        num_heads=8,
        dropout=0.1,
        max_seq_len=1000,
        schedule_type="linear" if use_scheduled_sampling else "none",
        schedule_k=5000.0
    )


def demonstrate_improvements():
    """
    Demonstration of all improvements with example usage.
    """
    print("=" * 80)
    print("SEQUENCE MODELING IMPROVEMENTS DEMONSTRATION")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 64
    latent_size = 64
    vocab_size = 29
    hidden_size = 128
    embed_size = 64

    # Create improved model
    print("\n1. Creating improved decoder with scheduled sampling...")
    model = create_improved_model(
        vocab_size=vocab_size,
        latent_size=latent_size,
        hidden_size=hidden_size,
        embed_size=embed_size,
        num_decoder_layers=3,
        use_scheduled_sampling=True
    ).to(device)

    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Using {model.num_layers} transformer layers")
    print(f"   - Scheduled sampling: {model.schedule_type}")

    # Example latent trajectory and tokens
    z = torch.randn(batch_size, seq_len, latent_size, device=device)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Test forward pass with scheduled sampling
    print("\n2. Testing forward pass with scheduled sampling...")
    model.train()
    logits = model.get_logits(z, tokens, use_scheduled_sampling=True)
    print(f"   - Output shape: {logits.shape}")
    print(f"   - Sampling probability: {model.get_scheduled_sampling_prob():.3f}")

    # Test generation with advanced strategies
    print("\n3. Testing generation with advanced sampling...")
    model.eval()

    # Generate with different strategies
    strategies = [
        {"temperature": 0.7, "top_k": 50, "name": "Top-k (k=50, T=0.7)"},
        {"temperature": 0.9, "top_p": 0.95, "name": "Nucleus (p=0.95, T=0.9)"},
        {"temperature": 1.0, "repetition_penalty": 1.2, "name": "Rep. penalty (1.2)"},
        {"use_beam_search": True, "beam_size": 5, "name": "Beam search (size=5)"},
    ]

    generated_sequences = []
    for strategy in strategies:
        name = strategy.pop("name")
        print(f"\n   Strategy: {name}")

        generated = model.generate(
            z[:1],  # Use first sample
            max_length=seq_len,
            start_token_id=0,
            **strategy
        )
        generated_sequences.append(generated[0])
        print(f"   - Generated shape: {generated.shape}")

    # Compute diversity metrics
    print("\n4. Computing generation quality metrics...")
    metrics = SequenceMetrics.compute_diversity_metrics(generated_sequences)
    self_bleu = SequenceMetrics.compute_self_bleu(generated_sequences)

    print(f"   - Distinct-1: {metrics['distinct_1']:.3f}")
    print(f"   - Distinct-2: {metrics['distinct_2']:.3f}")
    print(f"   - Self-BLEU: {self_bleu:.3f} (lower = more diverse)")

    # Test perplexity computation
    print("\n5. Testing perplexity computation...")
    with torch.no_grad():
        logits = model.get_logits(z, tokens, use_scheduled_sampling=False)
        perplexity = SequenceMetrics.compute_perplexity(logits, tokens)
        print(f"   - Perplexity: {perplexity:.2f}")

    print("\n" + "=" * 80)
    print("IMPROVEMENTS SUCCESSFULLY DEMONSTRATED!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_improvements()