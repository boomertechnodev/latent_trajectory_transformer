# Sequence Modeling Expert Skill

Comprehensive implementation of autoregressive models, advanced decoding strategies, and training techniques for sequence generation. This skill provides state-of-the-art implementations for language modeling and sequence-to-sequence tasks.

## Core Implementations

### 1. Advanced Teacher Forcing Strategies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import math
from dataclasses import dataclass
from collections import defaultdict
import heapq


class ScheduledSamplingStrategy:
    """Collection of scheduled sampling strategies for reducing exposure bias."""

    def __init__(self, strategy: str = 'linear', k: float = 2000.0, c: float = 0.0):
        """
        Args:
            strategy: 'linear', 'exponential', 'inverse_sigmoid', 'constant'
            k: Schedule parameter (speed of transition)
            c: Constant probability (for 'constant' strategy)
        """
        self.strategy = strategy
        self.k = k
        self.c = c
        self.step = 0

    def get_sampling_probability(self) -> float:
        """Get probability of using model prediction vs ground truth."""
        if self.strategy == 'constant':
            return self.c

        elif self.strategy == 'linear':
            return min(1.0, self.step / self.k)

        elif self.strategy == 'exponential':
            return 1.0 - math.exp(-self.step / self.k)

        elif self.strategy == 'inverse_sigmoid':
            return self.k / (self.k + math.exp(self.step / self.k))

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def step_forward(self):
        """Increment training step."""
        self.step += 1

    def reset(self):
        """Reset schedule."""
        self.step = 0


class ProfessorForcingModule(nn.Module):
    """
    Professor Forcing implementation that uses future information to guide generation.
    Based on "Professor Forcing: A New Algorithm for Training Recurrent Networks" (Lamb et al., 2016)
    """

    def __init__(self, hidden_dim: int, future_weight: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_weight = future_weight

        # Future encoder (bidirectional)
        self.future_encoder = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

        # Discriminator for adversarial training
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Hidden state adapter
        self.adapter = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, past_hidden: torch.Tensor, future_sequence: torch.Tensor) -> torch.Tensor:
        """
        Combine past and future information.

        Args:
            past_hidden: Hidden states from autoregressive generation (B, T, H)
            future_sequence: Future sequence embeddings (B, T, H)

        Returns:
            Adapted hidden states
        """
        # Encode future information
        future_hidden, _ = self.future_encoder(future_sequence)

        # Combine past and future
        combined = torch.cat([past_hidden, future_hidden], dim=-1)
        adapted = self.adapter(combined)

        # Mix with original based on weight
        output = (1 - self.future_weight) * past_hidden + self.future_weight * adapted

        return output

    def discriminator_loss(self, teacher_forced: torch.Tensor, free_running: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss for professor forcing."""
        # Label teacher-forced as real (1), free-running as fake (0)
        real_scores = self.discriminator(teacher_forced)
        fake_scores = self.discriminator(free_running.detach())

        real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
        fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))

        return real_loss + fake_loss

    def generator_loss(self, free_running: torch.Tensor) -> torch.Tensor:
        """Compute generator loss for professor forcing."""
        # Try to fool discriminator
        fake_scores = self.discriminator(free_running)
        return F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))


class MixedTeacherForcing(nn.Module):
    """
    Advanced teacher forcing that mixes multiple strategies.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Core LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Embedding and projection
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.projection = nn.Linear(hidden_dim, vocab_size)

        # Scheduled sampling
        self.sampling_schedule = ScheduledSamplingStrategy('exponential')

        # Curriculum learning
        self.max_length = 100
        self.current_max_length = 10
        self.length_increment = 0.01

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with mixed teacher forcing strategies.
        """
        batch_size, seq_len = input_ids.shape

        # Truncate to current curriculum length
        if self.training:
            seq_len = min(seq_len, int(self.current_max_length))
            input_ids = input_ids[:, :seq_len]
            if targets is not None:
                targets = targets[:, :seq_len]

        # Initialize hidden state
        hidden = None
        outputs = []

        # Embed input
        embeddings = self.embedding(input_ids)

        for t in range(seq_len):
            # Get input for this timestep
            if t == 0 or targets is None or not self.training:
                # Use actual input
                input_t = embeddings[:, t:t+1]
            else:
                # Scheduled sampling
                use_teacher = torch.rand(1) > self.sampling_schedule.get_sampling_probability()

                if use_teacher:
                    input_t = embeddings[:, t:t+1]
                else:
                    # Use previous prediction
                    prev_logits = outputs[-1]
                    prev_tokens = torch.argmax(prev_logits, dim=-1)
                    input_t = self.embedding(prev_tokens)

            # LSTM step
            output, hidden = self.lstm(input_t, hidden)

            # Project to vocabulary
            logits = self.projection(output)
            outputs.append(logits)

        # Update schedules
        if self.training:
            self.sampling_schedule.step_forward()
            self.current_max_length = min(self.max_length,
                                         self.current_max_length + self.length_increment)

        return torch.cat(outputs, dim=1)


### 2. Advanced Decoding Algorithms

@dataclass
class BeamSearchNode:
    """Node in beam search tree."""
    tokens: List[int]
    score: float
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
    attention_weights: Optional[torch.Tensor] = None

    def __lt__(self, other):
        return self.score < other.score


class AdvancedDecoder:
    """Collection of advanced decoding strategies."""

    @staticmethod
    def beam_search(model: nn.Module, input_ids: torch.Tensor, beam_size: int = 5,
                   max_length: int = 100, length_penalty: float = 0.6,
                   coverage_penalty: float = 0.0, no_repeat_ngram_size: int = 0,
                   temperature: float = 1.0) -> List[List[int]]:
        """
        Advanced beam search with length normalization and coverage penalty.

        Args:
            model: Language model with generate_step method
            input_ids: Input token IDs (batch_size, seq_len)
            beam_size: Number of beams
            max_length: Maximum generation length
            length_penalty: Wu et al. (2016) length penalty factor
            coverage_penalty: Coverage penalty for attention-based models
            no_repeat_ngram_size: Block repeated n-grams
            temperature: Sampling temperature

        Returns:
            List of best sequences for each batch item
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        results = []

        for batch_idx in range(batch_size):
            # Initialize beams
            start_tokens = input_ids[batch_idx:batch_idx+1]
            beams = [BeamSearchNode(
                tokens=start_tokens.squeeze().tolist(),
                score=0.0,
                hidden=None
            )]

            # Generate tokens
            for step in range(max_length - len(start_tokens[0])):
                candidates = []

                for beam in beams:
                    # Get model predictions
                    current_tokens = torch.tensor([beam.tokens], device=device)
                    logits, hidden = model.generate_step(current_tokens, beam.hidden)

                    # Apply temperature
                    logits = logits / temperature

                    # Apply no-repeat-ngram blocking
                    if no_repeat_ngram_size > 0:
                        logits = AdvancedDecoder._block_ngrams(
                            beam.tokens, logits, no_repeat_ngram_size
                        )

                    # Get top-k tokens
                    log_probs = F.log_softmax(logits, dim=-1)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                    # Create new candidates
                    for k in range(beam_size):
                        new_tokens = beam.tokens + [top_indices[0, k].item()]
                        new_score = beam.score + top_log_probs[0, k].item()

                        # Apply length penalty
                        length = len(new_tokens)
                        length_factor = ((5 + length) / 6) ** length_penalty
                        normalized_score = new_score / length_factor

                        candidates.append(BeamSearchNode(
                            tokens=new_tokens,
                            score=normalized_score,
                            hidden=hidden
                        ))

                # Select top beams
                candidates.sort(reverse=True)
                beams = candidates[:beam_size]

                # Check for early stopping
                if all(beam.tokens[-1] == model.eos_token_id for beam in beams):
                    break

            # Return best sequence
            results.append(beams[0].tokens)

        return results

    @staticmethod
    def diverse_beam_search(model: nn.Module, input_ids: torch.Tensor,
                           num_beams: int = 5, num_groups: int = 5,
                           diversity_penalty: float = 0.5, **kwargs) -> List[List[int]]:
        """
        Diverse beam search with group-based diversity.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        beams_per_group = num_beams // num_groups
        results = []

        for batch_idx in range(batch_size):
            group_beams = [[] for _ in range(num_groups)]

            # Initialize each group
            start_tokens = input_ids[batch_idx:batch_idx+1]
            for g in range(num_groups):
                group_beams[g].append(BeamSearchNode(
                    tokens=start_tokens.squeeze().tolist(),
                    score=0.0,
                    hidden=None
                ))

            # Generate tokens
            for step in range(kwargs.get('max_length', 100)):
                for g in range(num_groups):
                    candidates = []

                    for beam in group_beams[g]:
                        # Get predictions
                        current_tokens = torch.tensor([beam.tokens], device=device)
                        logits, hidden = model.generate_step(current_tokens, beam.hidden)

                        # Apply diversity penalty based on previous groups
                        if g > 0:
                            for prev_g in range(g):
                                for prev_beam in group_beams[prev_g]:
                                    if len(prev_beam.tokens) > step:
                                        prev_token = prev_beam.tokens[step]
                                        logits[0, prev_token] -= diversity_penalty

                        # Get top candidates
                        log_probs = F.log_softmax(logits, dim=-1)
                        top_log_probs, top_indices = torch.topk(log_probs, beams_per_group)

                        # Create candidates
                        for k in range(beams_per_group):
                            new_tokens = beam.tokens + [top_indices[0, k].item()]
                            new_score = beam.score + top_log_probs[0, k].item()

                            candidates.append(BeamSearchNode(
                                tokens=new_tokens,
                                score=new_score,
                                hidden=hidden
                            ))

                    # Select best for this group
                    candidates.sort(reverse=True)
                    group_beams[g] = candidates[:beams_per_group]

            # Select overall best
            all_beams = [beam for group in group_beams for beam in group]
            all_beams.sort(reverse=True)
            results.append(all_beams[0].tokens)

        return results

    @staticmethod
    def nucleus_sampling(logits: torch.Tensor, p: float = 0.95, temperature: float = 1.0,
                        min_tokens_to_keep: int = 1) -> torch.Tensor:
        """
        Top-p (nucleus) sampling with temperature.
        """
        batch_size, vocab_size = logits.shape

        # Apply temperature
        logits = logits / temperature

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find cutoff index
        cutoff_index = torch.searchsorted(cumulative_probs, p, right=False)
        cutoff_index = torch.clamp(cutoff_index, min=min_tokens_to_keep - 1)

        # Create mask
        indices_to_remove = sorted_indices.clone()
        for b in range(batch_size):
            indices_to_remove[b, cutoff_index[b] + 1:] = -1

        # Apply mask
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove == -1] = -float('inf')

        # Sample
        probs = F.softmax(filtered_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)

        return sampled

    @staticmethod
    def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.95,
                             temperature: float = 1.0) -> torch.Tensor:
        """
        Combined top-k and top-p filtering.
        """
        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')

        return logits

    @staticmethod
    def _block_ngrams(tokens: List[int], logits: torch.Tensor, n: int) -> torch.Tensor:
        """Block repeated n-grams in logits."""
        if len(tokens) < n:
            return logits

        # Get n-grams in existing sequence
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n-1])
            ngrams.add(ngram)

        # Check if current context matches any n-gram prefix
        current_ngram = tuple(tokens[-(n-1):])
        if current_ngram in ngrams:
            # Find the token that would complete this n-gram
            for i in range(len(tokens) - n + 1):
                if tuple(tokens[i:i+n-1]) == current_ngram:
                    blocked_token = tokens[i+n-1] if i+n-1 < len(tokens) else None
                    if blocked_token is not None:
                        logits[0, blocked_token] = -float('inf')

        return logits


### 3. Efficient Training Techniques

class EfficientSequenceTrainer:
    """Memory and compute efficient training techniques for sequence models."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.gradient_checkpointing = False
        self.mixed_precision = False
        self.gradient_accumulation_steps = 1
        self.current_accumulation_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor], loss_fn: callable) -> Dict[str, float]:
        """
        Single training step with efficiency optimizations.
        """
        metrics = {}

        # Mixed precision context
        if self.mixed_precision:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()

            with autocast():
                outputs = self._forward_with_checkpointing(batch)
                loss = loss_fn(outputs, batch['labels'])
        else:
            outputs = self._forward_with_checkpointing(batch)
            loss = loss_fn(outputs, batch['labels'])

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights after accumulation
        self.current_accumulation_step += 1
        if self.current_accumulation_step >= self.gradient_accumulation_steps:
            if self.mixed_precision:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.current_accumulation_step = 0

            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        metrics['loss'] = loss.item() * self.gradient_accumulation_steps
        return metrics

    def _forward_with_checkpointing(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.model.training:
            from torch.utils.checkpoint import checkpoint

            # Checkpoint transformer layers
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            # Apply checkpointing to each layer
            hidden_states = batch['input_ids']
            for layer in self.model.layers:
                hidden_states = checkpoint(create_custom_forward(layer), hidden_states)
            return hidden_states
        else:
            return self.model(batch['input_ids'])


class CurriculumLearningScheduler:
    """Curriculum learning for sequence models."""

    def __init__(self, min_length: int = 10, max_length: int = 512,
                 warmup_steps: int = 10000, strategy: str = 'exponential'):
        self.min_length = min_length
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.strategy = strategy
        self.current_step = 0

    def get_max_length(self) -> int:
        """Get current maximum sequence length."""
        if self.current_step >= self.warmup_steps:
            return self.max_length

        progress = self.current_step / self.warmup_steps

        if self.strategy == 'linear':
            current_max = self.min_length + (self.max_length - self.min_length) * progress

        elif self.strategy == 'exponential':
            # Exponential growth
            ratio = self.max_length / self.min_length
            current_max = self.min_length * (ratio ** progress)

        elif self.strategy == 'stepwise':
            # Discrete steps
            n_steps = 5
            step_size = (self.max_length - self.min_length) / n_steps
            current_step = int(progress * n_steps)
            current_max = self.min_length + step_size * current_step

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return int(current_max)

    def step(self):
        """Increment training step."""
        self.current_step += 1

    def should_increase_length(self) -> bool:
        """Check if length should be increased."""
        old_length = self.get_max_length()
        self.step()
        new_length = self.get_max_length()
        return new_length > old_length


### 4. Sequence Evaluation Metrics

class SequenceMetrics:
    """Comprehensive metrics for sequence generation evaluation."""

    @staticmethod
    def compute_perplexity(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                          pad_token_id: int = 0) -> float:
        """
        Compute perplexity on a dataset.
        """
        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                # Shift for next-token prediction
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                mask = attention_mask[:, 1:]

                # Get model predictions
                logits = model(inputs)

                # Compute cross-entropy
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction='none',
                    ignore_index=pad_token_id
                )

                # Apply mask and accumulate
                loss = loss * mask.reshape(-1)
                total_loss += loss.sum().item()
                total_tokens += mask.sum().item()

        # Calculate perplexity
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(avg_loss)

        return perplexity

    @staticmethod
    def compute_bleu(hypotheses: List[List[str]], references: List[List[List[str]]],
                     max_n: int = 4, smooth: bool = True) -> Dict[str, float]:
        """
        Compute BLEU score with multiple n-gram precisions.
        """
        from collections import Counter

        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Get n-grams from token list."""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)

        def compute_precision(hypothesis: List[str], references: List[List[str]], n: int) -> Tuple[int, int]:
            """Compute n-gram precision."""
            hyp_ngrams = get_ngrams(hypothesis, n)
            ref_ngrams = [get_ngrams(ref, n) for ref in references]

            # Count matches
            matches = 0
            total = sum(hyp_ngrams.values())

            for ngram, count in hyp_ngrams.items():
                max_ref_count = max(ref_ngram.get(ngram, 0) for ref_ngram in ref_ngrams)
                matches += min(count, max_ref_count)

            return matches, total

        # Compute BLEU for each n
        precisions = []
        for n in range(1, max_n + 1):
            total_matches = 0
            total_predicted = 0

            for hyp, refs in zip(hypotheses, references):
                matches, predicted = compute_precision(hyp, refs, n)
                total_matches += matches
                total_predicted += predicted

            # Smoothing
            if smooth and total_matches == 0:
                precision = 1 / (2 ** n)
            else:
                precision = total_matches / max(1, total_predicted)

            precisions.append(precision)

        # Brevity penalty
        total_hyp_len = sum(len(hyp) for hyp in hypotheses)
        total_ref_len = sum(
            min(len(ref) for ref in refs) for refs in references
        )
        brevity_penalty = min(1.0, math.exp(1 - total_ref_len / max(1, total_hyp_len)))

        # Geometric mean of precisions
        log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
        geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))

        # Final BLEU score
        bleu = brevity_penalty * geometric_mean

        return {
            'bleu': bleu,
            'brevity_penalty': brevity_penalty,
            **{f'precision_{i+1}': p for i, p in enumerate(precisions)}
        }

    @staticmethod
    def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        def get_ngrams(text: str, n: int) -> set:
            tokens = text.split()
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        def lcs_length(x: List[str], y: List[str]) -> int:
            """Longest common subsequence."""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        scores = defaultdict(list)

        for hyp, ref in zip(hypotheses, references):
            # ROUGE-1 (unigram)
            hyp_unigrams = get_ngrams(hyp, 1)
            ref_unigrams = get_ngrams(ref, 1)
            if len(hyp_unigrams) > 0 and len(ref_unigrams) > 0:
                precision = len(hyp_unigrams & ref_unigrams) / len(hyp_unigrams)
                recall = len(hyp_unigrams & ref_unigrams) / len(ref_unigrams)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                scores['rouge1'].append(f1)

            # ROUGE-2 (bigram)
            hyp_bigrams = get_ngrams(hyp, 2)
            ref_bigrams = get_ngrams(ref, 2)
            if len(hyp_bigrams) > 0 and len(ref_bigrams) > 0:
                precision = len(hyp_bigrams & ref_bigrams) / len(hyp_bigrams)
                recall = len(hyp_bigrams & ref_bigrams) / len(ref_bigrams)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                scores['rouge2'].append(f1)

            # ROUGE-L (LCS)
            hyp_tokens = hyp.split()
            ref_tokens = ref.split()
            lcs_len = lcs_length(hyp_tokens, ref_tokens)
            if len(hyp_tokens) > 0 and len(ref_tokens) > 0:
                precision = lcs_len / len(hyp_tokens)
                recall = lcs_len / len(ref_tokens)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                scores['rougeL'].append(f1)

        return {k: np.mean(v) for k, v in scores.items()}

    @staticmethod
    def compute_self_bleu(generated_texts: List[List[str]], n: int = 4) -> float:
        """
        Compute Self-BLEU for diversity measurement.
        Lower scores indicate more diverse generation.
        """
        if len(generated_texts) < 2:
            return 0.0

        self_bleu_scores = []

        for i, hypothesis in enumerate(generated_texts):
            # Use all other texts as references
            references = [generated_texts[j] for j in range(len(generated_texts)) if j != i]

            # Compute BLEU
            bleu = SequenceMetrics.compute_bleu([hypothesis], [references], max_n=n)
            self_bleu_scores.append(bleu['bleu'])

        return np.mean(self_bleu_scores)

    @staticmethod
    def compute_distinct_n(texts: List[str], n: int = 2) -> float:
        """
        Compute Distinct-n metric for diversity.
        Ratio of unique n-grams to total n-grams.
        """
        all_ngrams = []

        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                all_ngrams.append(tuple(tokens[i:i+n]))

        if not all_ngrams:
            return 0.0

        return len(set(all_ngrams)) / len(all_ngrams)


### 5. Specialized Architectures

class CachedTransformer(nn.Module):
    """
    Transformer with KV-cache for efficient generation.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            CachedTransformerLayer(d_model, nhead)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size)
        self.cache = {}

    def forward(self, input_ids: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        # Check if we're continuing from cache
        if use_cache and 'position' in self.cache:
            start_pos = self.cache['position']
            input_ids = input_ids[:, start_pos:]
        else:
            start_pos = 0
            self.cache = {'position': 0}

        # Embed and add positions
        x = self.embedding(input_ids)
        positions = torch.arange(start_pos, start_pos + x.shape[1], device=x.device)
        x = self.pos_encoding(x, positions)

        # Process through layers
        for i, layer in enumerate(self.layers):
            if use_cache:
                layer_cache = self.cache.get(f'layer_{i}', {})
                x, layer_cache = layer(x, cache=layer_cache)
                self.cache[f'layer_{i}'] = layer_cache
            else:
                x, _ = layer(x, cache=None)

        # Update position
        if use_cache:
            self.cache['position'] = start_pos + input_ids.shape[1]

        return self.output_projection(x)

    def clear_cache(self):
        """Clear KV-cache."""
        self.cache = {}


class CachedTransformerLayer(nn.Module):
    """Single transformer layer with KV-cache support."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cache: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        # Self-attention with cache
        if cache is not None and 'k' in cache and 'v' in cache:
            # Compute new K, V
            new_k = self.self_attn.in_proj_weight[x.shape[-1]:2*x.shape[-1]] @ x.transpose(-2, -1)
            new_v = self.self_attn.in_proj_weight[2*x.shape[-1]:] @ x.transpose(-2, -1)

            # Concatenate with cached
            k = torch.cat([cache['k'], new_k.transpose(-2, -1)], dim=1)
            v = torch.cat([cache['v'], new_v.transpose(-2, -1)], dim=1)

            # Update cache
            cache = {'k': k, 'v': v}
        else:
            k = v = x
            cache = {}

        # Attention
        attn_output, _ = self.self_attn(x, k, v)
        x = self.norm1(x + attn_output)

        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, cache


class PositionalEncoding(nn.Module):
    """Flexible positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is None:
            positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe[positions]


### 6. Controllable Generation

class ControllableGenerator:
    """Methods for controllable text generation."""

    @staticmethod
    def pplm_generation(model: nn.Module, input_ids: torch.Tensor,
                       attribute_model: nn.Module, target_class: int,
                       num_iterations: int = 3, step_size: float = 0.01,
                       temperature: float = 1.0) -> torch.Tensor:
        """
        Plug and Play Language Model (PPLM) for controllable generation.
        """
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(50):  # Max generation length
            # Forward pass
            logits = model(generated)[:, -1, :]

            # Perturb with gradients toward target attribute
            for _ in range(num_iterations):
                # Get gradients
                logits.requires_grad_(True)
                probs = F.softmax(logits / temperature, dim=-1)

                # Sample token
                sampled = torch.multinomial(probs, 1)

                # Get attribute score
                attribute_input = torch.cat([generated, sampled], dim=1)
                attribute_score = attribute_model(attribute_input)
                loss = -attribute_score[:, target_class].mean()

                # Compute gradients
                grad = torch.autograd.grad(loss, logits)[0]

                # Update logits
                logits = logits - step_size * grad
                logits = logits.detach()

            # Sample final token
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if next_token.item() == model.config.eos_token_id:
                break

        return generated

    @staticmethod
    def weighted_decoding(model: nn.Module, input_ids: torch.Tensor,
                         positive_words: List[int], negative_words: List[int],
                         alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
        """
        Weighted decoding with word lists.
        """
        generated = input_ids.clone()

        for _ in range(50):
            logits = model(generated)[:, -1, :]

            # Boost positive words
            for word in positive_words:
                logits[:, word] += alpha

            # Suppress negative words
            for word in negative_words:
                logits[:, word] -= beta

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == model.config.eos_token_id:
                break

        return generated


# Usage Example
if __name__ == "__main__":
    # Example model setup
    vocab_size = 10000
    hidden_dim = 256
    model = MixedTeacherForcing(vocab_size, hidden_dim)

    # Example training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = EfficientSequenceTrainer(model, optimizer)
    trainer.gradient_accumulation_steps = 4
    trainer.gradient_checkpointing = True

    # Example batch
    batch = {
        'input_ids': torch.randint(0, vocab_size, (32, 100)),
        'labels': torch.randint(0, vocab_size, (32, 100))
    }

    # Training step
    def loss_fn(outputs, targets):
        return F.cross_entropy(outputs.reshape(-1, vocab_size), targets.reshape(-1))

    metrics = trainer.train_step(batch, loss_fn)
    print(f"Training metrics: {metrics}")

    # Example generation
    decoder = AdvancedDecoder()
    input_text = torch.randint(0, vocab_size, (1, 10))

    # Beam search
    sequences = decoder.beam_search(model, input_text, beam_size=5)
    print(f"Generated sequences: {sequences}")

    # Compute metrics
    metrics_computer = SequenceMetrics()

    # Example texts for evaluation
    hypotheses = [["this", "is", "a", "test"], ["another", "example", "text"]]
    references = [[["this", "is", "the", "test"]], [["another", "sample", "text"]]]

    bleu = metrics_computer.compute_bleu(hypotheses, references)
    print(f"BLEU scores: {bleu}")