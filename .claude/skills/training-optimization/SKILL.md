# Training Optimization Skill

Advanced techniques for optimizing neural network training, including warmup strategies, mixed precision (AMP), hyperparameter search (Bayesian optimization, Hyperband), and advanced optimizer implementations.

## Core Competencies

### 1. Advanced Warmup Strategies

#### Linear Warmup
```python
class LinearWarmup:
    """Linear learning rate warmup."""

    def __init__(self, optimizer, warmup_steps, base_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (
                self.current_step / self.warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1
```

#### Exponential Warmup
```python
class ExponentialWarmup:
    """Exponential learning rate warmup for faster convergence."""

    def __init__(self, optimizer, warmup_steps, base_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_step = 0
        self.warmup_factor = (target_lr / base_lr) ** (1 / warmup_steps)

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.warmup_factor ** self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1
```

#### RAdam-style Warmup
```python
class RAdamWarmup:
    """RAdam-style adaptive warmup based on variance adaptation."""

    def __init__(self, optimizer, beta2=0.999):
        self.optimizer = optimizer
        self.beta2 = beta2
        self.current_step = 0

    def get_variance_rectification_term(self):
        beta2_t = self.beta2 ** self.current_step
        N_sma_max = 2 / (1 - self.beta2) - 1
        N_sma = N_sma_max - 2 * self.current_step * beta2_t / (1 - beta2_t)

        if N_sma >= 5:
            step_size = math.sqrt(
                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
            )
        else:
            step_size = 1.0

        return step_size

    def step(self):
        self.current_step += 1
        rect_term = self.get_variance_rectification_term()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * rect_term
```

### 2. Mixed Precision Training (AMP)

#### Custom AMP Trainer
```python
import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any

class AMPTrainer:
    """Advanced mixed precision trainer with dynamic loss scaling."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        initial_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        dtype: torch.dtype = torch.float16
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler(
            init_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        )
        self.dtype = dtype
        self.overflow_count = 0
        self.effective_steps = 0

    def train_step(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_fn: callable,
        clip_grad_norm: Optional[float] = 1.0
    ) -> Dict[str, Any]:
        """Single training step with mixed precision."""

        self.optimizer.zero_grad()

        # Forward pass with autocast
        with autocast(enabled=True, dtype=self.dtype):
            outputs = self.model(**inputs)
            loss = loss_fn(outputs, labels)

        # Scale loss and backward
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()

        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            clip_grad_norm
        )

        # Optimizer step with overflow check
        self.scaler.step(self.optimizer)
        scale_before = self.scaler.get_scale()
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        # Track overflow
        if scale_after < scale_before:
            self.overflow_count += 1
        else:
            self.effective_steps += 1

        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'scale': scale_after,
            'overflow_rate': self.overflow_count / max(1, self.effective_steps)
        }

    def benchmark_dtype_performance(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_steps: int = 100
    ) -> Dict[str, Dict]:
        """Benchmark performance across different dtypes."""

        import time

        results = {}
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue

            torch.cuda.synchronize()
            start_time = time.time()
            total_loss = 0

            for step, (inputs, labels) in enumerate(dataloader):
                if step >= num_steps:
                    break

                with autocast(enabled=(dtype != torch.float32), dtype=dtype):
                    outputs = self.model(**inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            results[str(dtype)] = {
                'time': elapsed,
                'throughput': num_steps / elapsed,
                'avg_loss': total_loss / num_steps,
                'memory': torch.cuda.max_memory_allocated() / 1024**3
            }

            torch.cuda.reset_peak_memory_stats()

        return results
```

#### Gradient Accumulation with AMP
```python
class GradientAccumulationAMPTrainer:
    """Gradient accumulation with mixed precision for large effective batch sizes."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        dtype: torch.dtype = torch.bfloat16
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.dtype = dtype
        self.scaler = GradScaler() if dtype == torch.float16 else None
        self.accumulated_loss = 0
        self.current_step = 0

    def accumulate_gradients(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_fn: callable
    ) -> Dict[str, Any]:
        """Accumulate gradients over multiple steps."""

        # Scale loss by accumulation steps
        with autocast(enabled=True, dtype=self.dtype):
            outputs = self.model(**inputs)
            loss = loss_fn(outputs, labels) / self.accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.accumulated_loss += loss.item()
        self.current_step += 1

        # Optimizer step when accumulation complete
        if self.current_step % self.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                self.optimizer.step()

            self.optimizer.zero_grad()

            result = {
                'loss': self.accumulated_loss,
                'grad_norm': grad_norm.item(),
                'effective_batch_size': self.accumulation_steps * inputs['input_ids'].size(0)
            }
            self.accumulated_loss = 0
            return result

        return None
```

### 3. Bayesian Optimization for Hyperparameters

#### Gaussian Process Based Optimization
```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimizer:
    """Bayesian optimization for hyperparameter search."""

    def __init__(
        self,
        bounds: Dict[str, tuple],
        n_random_starts: int = 5,
        acquisition: str = 'ei'  # Expected Improvement
    ):
        self.bounds = bounds
        self.n_random_starts = n_random_starts
        self.acquisition = acquisition

        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )

        self.X_observed = []
        self.y_observed = []
        self.iteration = 0

    def _acquisition_function(self, x: np.ndarray) -> float:
        """Compute acquisition function value."""

        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)

        if self.acquisition == 'ei':  # Expected Improvement
            if len(self.y_observed) > 0:
                y_best = np.max(self.y_observed)
                z = (mu - y_best) / (sigma + 1e-9)
                ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei[0]  # Minimize negative EI
            else:
                return -mu[0]

        elif self.acquisition == 'ucb':  # Upper Confidence Bound
            kappa = 2.0
            return -(mu + kappa * sigma)[0]

        elif self.acquisition == 'pi':  # Probability of Improvement
            if len(self.y_observed) > 0:
                y_best = np.max(self.y_observed)
                z = (mu - y_best) / (sigma + 1e-9)
                return -norm.cdf(z)[0]
            else:
                return -mu[0]

    def suggest_next(self) -> Dict[str, float]:
        """Suggest next hyperparameters to evaluate."""

        if self.iteration < self.n_random_starts:
            # Random exploration
            suggestion = {}
            for param, (low, high) in self.bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    suggestion[param] = np.random.randint(low, high + 1)
                else:
                    suggestion[param] = np.random.uniform(low, high)
        else:
            # Bayesian optimization
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

            # Optimize acquisition function
            dim = len(self.bounds)
            x0 = np.random.uniform(0, 1, size=(10, dim))

            best_x = None
            best_acq = float('inf')

            for start_point in x0:
                res = minimize(
                    self._acquisition_function,
                    start_point,
                    bounds=[(0, 1)] * dim,
                    method='L-BFGS-B'
                )

                if res.fun < best_acq:
                    best_acq = res.fun
                    best_x = res.x

            # Convert back to original space
            suggestion = {}
            for i, (param, (low, high)) in enumerate(self.bounds.items()):
                value = best_x[i] * (high - low) + low
                if isinstance(low, int) and isinstance(high, int):
                    suggestion[param] = int(np.round(value))
                else:
                    suggestion[param] = value

        self.iteration += 1
        return suggestion

    def update(self, params: Dict[str, float], score: float):
        """Update observations with new result."""

        # Normalize parameters to [0, 1]
        x = []
        for param, (low, high) in self.bounds.items():
            value = (params[param] - low) / (high - low)
            x.append(value)

        self.X_observed.append(x)
        self.y_observed.append(score)

    def get_best(self) -> tuple:
        """Get best hyperparameters found so far."""

        if len(self.y_observed) == 0:
            return None, None

        best_idx = np.argmax(self.y_observed)
        best_score = self.y_observed[best_idx]

        # Convert back to original space
        best_params = {}
        for i, (param, (low, high)) in enumerate(self.bounds.items()):
            value = self.X_observed[best_idx][i] * (high - low) + low
            if isinstance(low, int) and isinstance(high, int):
                best_params[param] = int(np.round(value))
            else:
                best_params[param] = value

        return best_params, best_score
```

### 4. Hyperband Algorithm

```python
class Hyperband:
    """Hyperband algorithm for efficient hyperparameter optimization."""

    def __init__(
        self,
        max_budget: int = 81,  # Maximum epochs
        eta: int = 3,  # Downsampling rate
        random_sampler: callable = None
    ):
        self.max_budget = max_budget
        self.eta = eta
        self.random_sampler = random_sampler

        # Compute bracket configurations
        self.s_max = int(np.log(max_budget) / np.log(eta))
        self.B = (self.s_max + 1) * max_budget

        self.results = []
        self.best_config = None
        self.best_score = float('-inf')

    def _get_bracket_config(self, s: int) -> tuple:
        """Get configuration for a bracket."""
        n = int(np.ceil(self.B / self.max_budget / (s + 1) * self.eta ** s))
        r = self.max_budget * self.eta ** (-s)
        return n, r

    def run_bracket(
        self,
        s: int,
        train_fn: callable,
        eval_fn: callable
    ) -> Dict:
        """Run one bracket of successive halving."""

        n, r = self._get_bracket_config(s)

        # Sample n random configurations
        configs = [self.random_sampler() for _ in range(n)]

        # Successive halving
        for i in range(s + 1):
            # Current budget
            r_i = r * self.eta ** i
            n_i = int(n * self.eta ** (-i))

            # Train and evaluate configurations
            scores = []
            for config in configs[:n_i]:
                # Train model
                model = train_fn(config, epochs=int(r_i))

                # Evaluate
                score = eval_fn(model)
                scores.append((score, config))

                # Track best
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config

            # Keep top k configurations
            if i < s:
                scores.sort(reverse=True)
                configs = [config for _, config in scores[:int(n_i / self.eta)]]

        return {
            'bracket': s,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'num_evaluations': sum(int(n * self.eta ** (-j)) for j in range(i + 1))
        }

    def optimize(
        self,
        train_fn: callable,
        eval_fn: callable
    ) -> Dict:
        """Run full Hyperband optimization."""

        for s in reversed(range(self.s_max + 1)):
            result = self.run_bracket(s, train_fn, eval_fn)
            self.results.append(result)

            print(f"Bracket {s}: Best score = {result['best_score']:.4f}")

        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'brackets_results': self.results
        }
```

### 5. Advanced Optimizers

#### Lion Optimizer Implementation
```python
class Lion(torch.optim.Optimizer):
    """Lion optimizer - Evolved Sign Momentum."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)

                # Update parameters
                p.data.add_(update.sign(), alpha=-group['lr'])

                # Update exponential average
                exp_avg.mul_(beta2).add(grad, alpha=1 - beta2)

        return loss
```

#### Lookahead Optimizer Wrapper
```python
class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper for any base optimizer."""

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0

        self.param_groups = base_optimizer.param_groups
        self.state = defaultdict(dict)

        # Store slow weights
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['slow_weight'] = p.data.clone()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = self.base_optimizer.step(closure)
        self.step_count += 1

        if self.step_count % self.k == 0:
            # Update slow weights
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    slow = state['slow_weight']

                    # Lookahead update
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)

        return loss

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        state_dict['step_count'] = self.step_count
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
```

### 6. Learning Rate Scheduling Strategies

#### Cosine Annealing with Restarts
```python
class CosineAnnealingWithRestarts:
    """Cosine annealing with warm restarts (SGDR)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 0.0,
        warmup_steps: int = 0,
        gamma: float = 1.0
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.current_step = 0
        self.T_cur = 0
        self.cycle = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_mult = self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine annealing with restarts
            if self.T_cur >= self.T_0 * (self.T_mult ** self.cycle):
                self.cycle += 1
                self.T_cur = 0
                self.base_lrs = [lr * self.gamma for lr in self.base_lrs]

            T_i = self.T_0 * (self.T_mult ** self.cycle)
            lr_mult = self.eta_min + 0.5 * (1 - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / T_i)
            )

            self.T_cur += 1

        # Apply learning rate
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult

        self.current_step += 1
```

#### Cyclical Learning Rate
```python
class CyclicalLR:
    """Cyclical learning rate with triangular/triangular2/exp_range policies."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-4,
        max_lr: float = 1e-2,
        step_size: int = 2000,
        mode: str = 'triangular',
        gamma: float = 1.0
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        self.current_step = 0

    def _triangular(self, cycle_position: float) -> float:
        """Triangular policy."""
        return self.base_lr + (self.max_lr - self.base_lr) * max(
            0, 1 - abs(1 - 2 * cycle_position)
        )

    def _triangular2(self, cycle_position: float, cycle: int) -> float:
        """Triangular2 policy with amplitude scaling."""
        scale = 1 / (2 ** (cycle - 1))
        return self.base_lr + (self.max_lr - self.base_lr) * max(
            0, 1 - abs(1 - 2 * cycle_position)
        ) * scale

    def _exp_range(self, cycle_position: float, cycle: int) -> float:
        """Exponential range policy."""
        scale = self.gamma ** (cycle - 1)
        return self.base_lr + (self.max_lr - self.base_lr) * max(
            0, 1 - abs(1 - 2 * cycle_position)
        ) * scale

    def step(self):
        """Update learning rate."""

        cycle = 1 + self.current_step // (2 * self.step_size)
        cycle_position = self.current_step / self.step_size - 2 * (cycle - 1)

        if self.mode == 'triangular':
            lr = self._triangular(cycle_position)
        elif self.mode == 'triangular2':
            lr = self._triangular2(cycle_position, cycle)
        elif self.mode == 'exp_range':
            lr = self._exp_range(cycle_position, cycle)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
```

### 7. Training Stability and Monitoring

```python
class TrainingMonitor:
    """Comprehensive training monitoring and stability detection."""

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.metrics_history = defaultdict(list)
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.gradient_norms = []

    def update(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update metrics and check training health."""

        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # Check for improvement
        current = metrics.get('loss', 0)
        if self.mode == 'min':
            improved = current < self.best_metric - self.min_delta
        else:
            improved = current > self.best_metric + self.min_delta

        if improved:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Compute statistics
        stats = {}
        for key, values in self.metrics_history.items():
            if len(values) >= 10:
                recent = values[-10:]
                stats[f'{key}_mean'] = np.mean(recent)
                stats[f'{key}_std'] = np.std(recent)
                stats[f'{key}_trend'] = np.polyfit(range(10), recent, 1)[0]

        # Check for issues
        warnings = []
        if 'loss' in metrics:
            if np.isnan(metrics['loss']) or np.isinf(metrics['loss']):
                warnings.append("Loss is NaN or Inf!")
            elif len(self.metrics_history['loss']) > 20:
                recent_losses = self.metrics_history['loss'][-20:]
                if np.std(recent_losses) > 10 * np.mean(recent_losses):
                    warnings.append("Loss is oscillating wildly!")

        if 'grad_norm' in metrics:
            self.gradient_norms.append(metrics['grad_norm'])
            if metrics['grad_norm'] > 100:
                warnings.append("Gradient explosion detected!")
            elif metrics['grad_norm'] < 1e-7:
                warnings.append("Vanishing gradients detected!")

        return {
            'should_stop': self.patience_counter >= self.patience,
            'warnings': warnings,
            'statistics': stats
        }
```

This skill provides comprehensive training optimization capabilities including advanced warmup strategies, mixed precision training, Bayesian optimization and Hyperband for hyperparameter search, custom optimizer implementations, and sophisticated learning rate scheduling. Each component is production-ready with proper error handling and monitoring capabilities.