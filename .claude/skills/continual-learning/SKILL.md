# Continual Learning Systems Engineering - Advanced Skill Module

## Mathematical Foundations

### Information-Theoretic View of Forgetting

**Catastrophic Forgetting as Information Loss:**

Given a sequence of tasks T₁, T₂, ..., Tₙ with corresponding data distributions p₁, p₂, ..., pₙ:

1. **Information Retention Metric:**
   ```
   I(θₜ; Dᵢ) = H(Dᵢ) - H(Dᵢ|θₜ)
   ```
   Where θₜ are parameters at time t, Dᵢ is data from task i

2. **Forgetting Measure:**
   ```
   F(i,t) = max_{s≤i} I(θₛ; Dᵢ) - I(θₜ; Dᵢ)
   ```
   Quantifies information loss about task i at time t

3. **Plasticity-Stability Dilemma:**
   ```
   L_total = L_current + λ·KL(p(θ|D_old) || p(θ|D_all))
   ```
   Trade-off between current task performance and parameter drift

### Memory Theory for Neural Networks

**Hopfield Energy Perspective:**

1. **Memory Capacity Bound (Hopfield):**
   ```
   P_max ≈ 0.138·N  (for N neurons)
   ```

2. **Modern Dense Associative Memory:**
   ```
   P_max ≈ 2^(αN) for α < 1
   ```
   Exponential capacity with interaction vertex models

3. **Gradient Episodic Memory Constraints:**
   ```
   ⟨g_current, g_memory⟩ ≥ -γ  ∀ memory samples
   ```
   Ensures gradients don't conflict with past tasks

### Optimal Sampling Theory

**Priority Sampling Mathematics:**

1. **TD-Error Priority (from RL):**
   ```
   p(i) = (|δᵢ| + ε)^α / Σⱼ(|δⱼ| + ε)^α
   ```
   Where δᵢ is temporal difference error

2. **Gradient Magnitude Priority:**
   ```
   p(i) = ||∇_θL(xᵢ,yᵢ;θ)||₂^β / Σⱼ||∇_θL(xⱼ,yⱼ;θ)||₂^β
   ```

3. **Uncertainty-Based Priority:**
   ```
   p(i) = H(p(y|xᵢ,θ)) / Σⱼ H(p(y|xⱼ,θ))
   ```
   Entropy of predictive distribution

## Advanced Implementation Techniques

### 1. Sophisticated Memory Buffers

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import heapq

class AdvancedExperienceBuffer:
    """
    Multi-strategy experience buffer with adaptive sampling.
    """
    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        strategies: List[str] = ['gradient', 'uncertainty', 'diversity']
    ):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.strategies = strategies

        # Core storage
        self.buffer = []
        self.features = torch.zeros((capacity, feature_dim))
        self.scores = {}
        for strategy in strategies:
            self.scores[strategy] = np.zeros(capacity)

        # Coverage tracking
        self.coverage_tree = self._build_coverage_tree()
        self.write_index = 0
        self.full = False

        # Statistics
        self.age = np.zeros(capacity)
        self.access_count = np.zeros(capacity)

    def _build_coverage_tree(self):
        """Build KD-tree for coverage-based sampling."""
        from scipy.spatial import KDTree
        return None  # Lazy initialization

    def add(
        self,
        sample: Dict,
        model: nn.Module,
        compute_scores: bool = True
    ):
        """Add sample with multi-criteria scoring."""
        idx = self.write_index

        # Store sample
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[idx] = sample
            self.full = True

        # Extract features
        with torch.no_grad():
            features = self._extract_features(sample, model)
            self.features[idx] = features

        # Compute scores
        if compute_scores:
            if 'gradient' in self.strategies:
                self.scores['gradient'][idx] = self._gradient_score(sample, model)
            if 'uncertainty' in self.strategies:
                self.scores['uncertainty'][idx] = self._uncertainty_score(sample, model)
            if 'diversity' in self.strategies:
                self.scores['diversity'][idx] = self._diversity_score(features)

        # Update metadata
        self.age[idx] = 0
        self.age[self.age > 0] += 1

        # Update write index
        self.write_index = (idx + 1) % self.capacity

        # Rebuild coverage tree periodically
        if self.write_index % 100 == 0:
            self._rebuild_coverage_tree()

    def _gradient_score(self, sample: Dict, model: nn.Module) -> float:
        """Compute gradient-based importance score."""
        model.eval()

        # Forward pass
        x, y = sample['input'], sample['target']
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        # Compute gradient norm
        grad_norm = 0
        for p in model.parameters():
            if p.requires_grad:
                grad = torch.autograd.grad(loss, p, retain_graph=True)[0]
                grad_norm += grad.norm().item() ** 2

        return np.sqrt(grad_norm)

    def _uncertainty_score(self, sample: Dict, model: nn.Module) -> float:
        """Compute uncertainty using MC Dropout."""
        model.train()  # Enable dropout

        x = sample['input']
        predictions = []

        # Multiple forward passes
        with torch.no_grad():
            for _ in range(10):
                output = model(x)
                predictions.append(torch.softmax(output, dim=-1))

        # Compute entropy of mean prediction
        mean_pred = torch.stack(predictions).mean(0)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum().item()

        return entropy

    def _diversity_score(self, features: torch.Tensor) -> float:
        """Compute diversity w.r.t. current buffer."""
        if len(self.buffer) < 2:
            return 1.0

        # Compute minimum distance to existing samples
        current_features = self.features[:len(self.buffer)]
        distances = torch.cdist(features.unsqueeze(0), current_features)
        min_dist = distances.min().item()

        return min_dist

    def sample(
        self,
        batch_size: int,
        strategy: str = 'mixed',
        temperature: float = 1.0
    ) -> List[Dict]:
        """Sample batch using specified strategy."""
        if len(self.buffer) == 0:
            return []

        n = min(batch_size, len(self.buffer))

        if strategy == 'uniform':
            indices = np.random.choice(len(self.buffer), n, replace=False)

        elif strategy in self.strategies:
            scores = self.scores[strategy][:len(self.buffer)]
            # Temperature-scaled softmax
            probs = np.exp(scores / temperature)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), n, replace=False, p=probs)

        elif strategy == 'mixed':
            # Combine multiple strategies
            combined_score = np.zeros(len(self.buffer))
            for s in self.strategies:
                scores = self.scores[s][:len(self.buffer)]
                # Normalize to [0, 1]
                if scores.max() > scores.min():
                    normalized = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    normalized = np.ones_like(scores)
                combined_score += normalized

            combined_score /= len(self.strategies)
            probs = np.exp(combined_score / temperature)
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), n, replace=False, p=probs)

        elif strategy == 'coreset':
            indices = self._coreset_selection(n)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Update access counts
        self.access_count[indices] += 1

        return [self.buffer[i] for i in indices]

    def _coreset_selection(self, k: int) -> np.ndarray:
        """K-center coreset selection for representative sampling."""
        n = len(self.buffer)
        if k >= n:
            return np.arange(n)

        features = self.features[:n].numpy()

        # Initialize with random center
        centers = [np.random.randint(n)]
        distances = np.full(n, np.inf)

        for _ in range(k - 1):
            # Update distances to nearest center
            new_distances = np.linalg.norm(
                features - features[centers[-1]], axis=1
            )
            distances = np.minimum(distances, new_distances)

            # Select farthest point as new center
            next_center = np.argmax(distances)
            centers.append(next_center)

        return np.array(centers)

    def evict(self, num_evict: int = 1):
        """Evict samples based on combined criteria."""
        if len(self.buffer) <= num_evict:
            return

        # Compute eviction scores (lower is worse)
        eviction_score = np.zeros(len(self.buffer))

        # Age penalty (older samples less valuable)
        age_penalty = 1.0 - np.exp(-self.age[:len(self.buffer)] / 1000)

        # Access bonus (frequently accessed samples more valuable)
        access_bonus = np.log1p(self.access_count[:len(self.buffer)])

        # Combine scores
        for strategy in self.strategies:
            scores = self.scores[strategy][:len(self.buffer)]
            if scores.max() > scores.min():
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized = np.ones_like(scores)
            eviction_score += normalized

        eviction_score = eviction_score / len(self.strategies)
        eviction_score = eviction_score - age_penalty + access_bonus

        # Select samples to evict
        evict_indices = np.argpartition(eviction_score, num_evict)[:num_evict]

        # Remove evicted samples
        keep_mask = np.ones(len(self.buffer), dtype=bool)
        keep_mask[evict_indices] = False

        self.buffer = [s for i, s in enumerate(self.buffer) if keep_mask[i]]
        self.features = self.features[keep_mask]
        for strategy in self.strategies:
            self.scores[strategy] = self.scores[strategy][keep_mask]
        self.age = self.age[keep_mask]
        self.access_count = self.access_count[keep_mask]
```

### 2. Advanced Regularization Techniques

```python
class AdaptiveEWC:
    """
    Elastic Weight Consolidation with adaptive importance weighting.
    """
    def __init__(
        self,
        model: nn.Module,
        decay_rate: float = 0.95,
        threshold: float = 0.01
    ):
        self.model = model
        self.decay_rate = decay_rate
        self.threshold = threshold

        # Running Fisher Information
        self.fisher = {}
        self.optimal_params = {}
        self.task_count = 0

    def update_fisher(self, dataloader, num_samples: int = 200):
        """Update Fisher Information Matrix with new task."""
        self.task_count += 1

        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

        # Compute Fisher for current task
        new_fisher = {}
        for name, param in self.model.named_parameters():
            new_fisher[name] = torch.zeros_like(param)

        self.model.eval()
        samples_seen = 0

        for batch in dataloader:
            if samples_seen >= num_samples:
                break

            x, y = batch
            output = self.model(x)

            # Sample from output distribution
            sampled_y = torch.multinomial(
                torch.softmax(output, dim=-1), 1
            ).squeeze()

            # Compute log-likelihood
            loss = nn.functional.cross_entropy(output, sampled_y)

            # Compute gradients
            self.model.zero_grad()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.data ** 2

            samples_seen += x.size(0)

        # Average Fisher
        for name in new_fisher:
            new_fisher[name] /= samples_seen

        # Update running Fisher with decay
        if self.task_count == 1:
            self.fisher = new_fisher
        else:
            for name in self.fisher:
                self.fisher[name] = (
                    self.decay_rate * self.fisher[name] +
                    (1 - self.decay_rate) * new_fisher[name]
                )

        # Threshold small values for numerical stability
        for name in self.fisher:
            self.fisher[name][self.fisher[name] < self.threshold] = 0

    def penalty(self, lambda_ewc: float = 1000) -> torch.Tensor:
        """Compute EWC penalty."""
        if self.task_count == 0:
            return torch.tensor(0.0)

        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                fisher_diag = self.fisher[name]
                optimal = self.optimal_params[name]

                loss += (fisher_diag * (param - optimal) ** 2).sum()

        return lambda_ewc * loss / 2


class PathIntegralRegularizer:
    """
    Synaptic Intelligence: Online computation of synaptic importance.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.omega = {}  # Importance weights
        self.params_old = {}
        self.gradients = {}

        # Initialize
        for name, param in model.named_parameters():
            self.omega[name] = torch.zeros_like(param)
            self.params_old[name] = param.data.clone()
            self.gradients[name] = torch.zeros_like(param)

    def update_omega(self):
        """Update importance weights after task."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                delta = param.data - self.params_old[name]
                self.omega[name] += self.gradients[name] * delta

        # Reset for next task
        for name, param in self.model.named_parameters():
            self.params_old[name] = param.data.clone()
            self.gradients[name].zero_()

    def accumulate_gradients(self):
        """Accumulate gradients during training."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.gradients[name] += param.grad.data.abs()

    def penalty(self, lambda_si: float = 1.0) -> torch.Tensor:
        """Compute SI penalty."""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.omega:
                loss += (self.omega[name] *
                        (param - self.params_old[name]) ** 2).sum()

        return lambda_si * loss
```

### 3. Meta-Continual Learning

```python
class MetaContinualLearner(nn.Module):
    """
    Meta-learning approach to continual learning using MAML-style updates.
    """
    def __init__(
        self,
        base_model: nn.Module,
        meta_lr: float = 0.01,
        inner_lr: float = 0.001,
        inner_steps: int = 5
    ):
        super().__init__()
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        # Meta-parameters
        self.meta_model = self._create_meta_model()

    def _create_meta_model(self):
        """Create meta-model with same architecture."""
        import copy
        return copy.deepcopy(self.base_model)

    def inner_loop(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        fast_weights: Optional[Dict] = None
    ) -> Dict:
        """Inner loop adaptation."""
        if fast_weights is None:
            fast_weights = {
                name: param.clone()
                for name, param in self.meta_model.named_parameters()
            }

        x_support, y_support = support_data

        for _ in range(self.inner_steps):
            # Forward with fast weights
            logits = self.functional_forward(x_support, fast_weights)
            loss = nn.functional.cross_entropy(logits, y_support)

            # Compute gradients w.r.t. fast weights
            grads = torch.autograd.grad(
                loss, fast_weights.values(),
                create_graph=True
            )

            # Update fast weights
            fast_weights = {
                name: weight - self.inner_lr * grad
                for (name, weight), grad in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def functional_forward(
        self,
        x: torch.Tensor,
        weights: Dict
    ) -> torch.Tensor:
        """Forward pass with given weights."""
        # Implement functional forward pass
        # This depends on your model architecture
        pass

    def outer_loop(
        self,
        task_batch: List[Tuple[torch.Tensor, torch.Tensor]],
        memory_batch: Optional[List] = None
    ):
        """Outer loop meta-update."""
        meta_loss = 0

        for task_data in task_batch:
            # Split into support and query
            x, y = task_data
            split = len(x) // 2
            support = (x[:split], y[:split])
            query = (x[split:], y[split:])

            # Inner loop adaptation
            fast_weights = self.inner_loop(support)

            # Compute loss on query set
            query_logits = self.functional_forward(query[0], fast_weights)
            task_loss = nn.functional.cross_entropy(query_logits, query[1])

            meta_loss += task_loss

        # Add memory constraint if available
        if memory_batch is not None:
            memory_weights = self.inner_loop(memory_batch)
            memory_logits = self.functional_forward(
                memory_batch[0], memory_weights
            )
            memory_loss = nn.functional.cross_entropy(
                memory_logits, memory_batch[1]
            )
            meta_loss += memory_loss

        # Meta-update
        meta_grads = torch.autograd.grad(meta_loss, self.meta_model.parameters())
        for param, grad in zip(self.meta_model.parameters(), meta_grads):
            param.data -= self.meta_lr * grad
```

## Debugging Strategies

### Common Issues and Solutions

1. **Memory Buffer Not Improving Performance**
   - Check priority scores distribution
   - Verify samples are diverse
   - Ensure buffer isn't dominated by easy examples
   - Monitor gradient alignment between tasks

2. **Catastrophic Forgetting Despite Regularization**
   - Fisher Information might be poorly estimated
   - Lambda values might be too small
   - Check if importance weights are updating
   - Verify old task data is being sampled

3. **Online Adaptation Too Slow**
   - Increase adaptation learning rate
   - Use momentum-based optimizers
   - Reduce memory replay batch size
   - Implement warm-start strategies

4. **Memory Overflow**
   - Implement better eviction policies
   - Use gradient accumulation instead of storing all samples
   - Compress features using autoencoders
   - Implement hierarchical memory structures

## Performance Optimization

### GPU Memory Management
```python
def optimize_memory_usage():
    # Clear cache periodically
    torch.cuda.empty_cache()

    # Use gradient checkpointing
    from torch.utils.checkpoint import checkpoint

    # Mixed precision training
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
```

### Efficient Sampling
```python
@torch.no_grad()
def fast_reservoir_sampling(stream, k):
    """O(n) reservoir sampling."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

## Literature References

### Foundational Papers
- **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
- **SI**: Zenke et al., "Continual Learning Through Synaptic Intelligence" (2017)
- **GEM**: Lopez-Paz & Ranzato, "Gradient Episodic Memory" (2017)
- **A-GEM**: Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (2019)

### Recent Advances
- **DER**: Buzzega et al., "Dark Experience for General Continual Learning" (2020)
- **ER-Ring**: Chaudhry et al., "On Tiny Episodic Memories in Continual Learning" (2019)
- **GSS**: Aljundi et al., "Gradient-based sample selection" (2019)
- **MER**: Riemer et al., "Learning to Learn without Forgetting by Maximizing Transfer" (2019)

### Theoretical Foundations
- **Information Theory**: Cover & Thomas, "Elements of Information Theory"
- **Statistical Learning**: Vapnik, "Statistical Learning Theory"
- **Online Learning**: Shalev-Shwartz, "Online Learning and Online Convex Optimization"

## Quick Reference

### Key Equations

| Method | Objective Function | Complexity |
|--------|-------------------|------------|
| EWC | L + λ/2 Σᵢ Fᵢ(θᵢ - θ*ᵢ)² | O(P) |
| SI | L + λ Σᵢ Ωᵢ(θᵢ - θ*ᵢ)² | O(P) |
| GEM | L s.t. ⟨g, gₖ⟩ ≥ 0 ∀k | O(KP) |
| A-GEM | L s.t. ⟨g, ḡ⟩ ≥ 0 | O(P) |

Where P = number of parameters, K = number of stored examples

### Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Memory Size | 200-5000 | Task-dependent |
| Replay Batch | 10-128 | Balance with new data |
| EWC Lambda | 100-10000 | Dataset-specific |
| Learning Rate | 1e-4 to 1e-2 | Lower for online |
| Inner Steps (Meta) | 1-10 | Computation trade-off |

### Decision Tree

```
Continual Learning Scenario?
├── Few Tasks + Large Memory? → Use Full Rehearsal
├── Many Tasks + Limited Memory?
│   ├── Can Store Gradients? → Use GEM/A-GEM
│   └── Can't Store Gradients? → Use EWC/SI
├── Non-Stationary Distribution? → Use Online EWC + Replay
└── Need Fast Adaptation? → Use Meta-Continual Learning
```

Remember: The key to successful continual learning is balancing the stability-plasticity tradeoff while maintaining computational efficiency.