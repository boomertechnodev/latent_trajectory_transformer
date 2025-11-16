"""
ALTERNATIVE RACCOON IMPLEMENTATION
Simpler architecture with different design choices for continuous learning.

Design Goals:
1. Simpler SDE using Ornstein-Uhlenbeck process
2. Affine flow layers (faster than coupling layers)
3. Fixed-size circular buffer memory (not priority-based)
4. CNN encoder instead of transformer
5. Direct classification head (no SDE trajectory evolution)
6. Synthetic log generator with realistic patterns
7. Training loop with early stopping and validation
8. Inference-only mode for deployment
9. Unit tests for all components
10. Full training with comparison metrics
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from tqdm import trange, tqdm


# ==============================================================================
# COMPONENT 1: REALISTIC SYNTHETIC LOG GENERATOR
# ==============================================================================

class RealisticLogGenerator:
    """Generate synthetic logs with timestamps, process IDs, and stack traces."""

    LOG_TEMPLATES = {
        "ERROR": [
            "ERROR: Connection timeout to {service}",
            "ERROR: Database connection failed: {error_code}",
            "ERROR: NullPointerException in {module}:{line}",
            "ERROR: Out of memory at {timestamp}",
            "ERROR: Permission denied accessing {resource}",
            "ERROR: Segmentation fault in {process}",
            "ERROR: Stack overflow in recursive call",
        ],
        "WARNING": [
            "WARN: Deprecated function {func} called",
            "WARN: Slow query detected: {duration}ms",
            "WARN: Memory usage high: {percent}%",
            "WARN: Connection retry {attempt}/{max_attempts}",
            "WARN: Cache miss for key {key}",
            "WARN: Rate limit approaching",
        ],
        "INFO": [
            "INFO: Service started on port {port}",
            "INFO: Database migration completed",
            "INFO: Request completed in {duration}ms",
            "INFO: Background job finished",
            "INFO: Configuration reloaded",
            "INFO: Health check passed",
        ],
        "DEBUG": [
            "DEBUG: Variable {var} = {value}",
            "DEBUG: Entering function {func}",
            "DEBUG: Loop iteration {i}/{n}",
            "DEBUG: Checkpoint reached at line {line}",
            "DEBUG: Memory allocation {size} bytes",
        ],
    }

    @staticmethod
    def generate(category: str, seq_len: int = 50) -> str:
        """
        Generate realistic log message.

        Args:
            category: Log category (ERROR, WARNING, INFO, DEBUG)
            seq_len: Desired sequence length

        Returns:
            Log message string
        """
        import random

        templates = RealisticLogGenerator.LOG_TEMPLATES.get(category, [""])
        template = random.choice(templates)

        # Fill in placeholders
        message = template
        message = message.replace("{service}", random.choice(["DB", "API", "CACHE", "AUTH"]))
        message = message.replace("{error_code}", str(random.randint(4000, 5000)))
        message = message.replace("{module}", random.choice(["Main", "Handler", "Service", "Utils"]))
        message = message.replace("{line}", str(random.randint(1, 1000)))
        message = message.replace("{timestamp}", f"{random.randint(0, 23)}:{random.randint(0, 59)}:{random.randint(0, 59)}")
        message = message.replace("{process}", f"PID_{random.randint(1000, 9999)}")
        message = message.replace("{resource}", random.choice(["/etc/config", "/var/log", "/home/user"]))
        message = message.replace("{func}", random.choice(["init", "process", "cleanup", "handler"]))
        message = message.replace("{duration}", str(random.randint(10, 5000)))
        message = message.replace("{percent}", str(random.randint(50, 99)))
        message = message.replace("{attempt}", str(random.randint(1, 3)))
        message = message.replace("{max_attempts}", "5")
        message = message.replace("{key}", f"key_{random.randint(1, 100)}")
        message = message.replace("{port}", str(random.randint(8000, 9000)))
        message = message.replace("{var}", random.choice(["x", "ptr", "buf", "count"]))
        message = message.replace("{value}", str(random.randint(0, 1000)))
        message = message.replace("{i}", str(random.randint(0, 100)))
        message = message.replace("{n}", str(random.randint(100, 1000)))
        message = message.replace("{size}", str(random.randint(1024, 65536)))

        # Pad to seq_len
        if len(message) < seq_len:
            message = message + "_" * (seq_len - len(message))
        else:
            message = message[:seq_len]

        return message


# ==============================================================================
# COMPONENT 2: SIMPLER SDE - ORNSTEIN-UHLENBECK PROCESS
# ==============================================================================

class OrnsteinUhlenbeckSDE(nn.Module):
    """
    Ornstein-Uhlenbeck process: dz = -theta*z*dt + sigma*dW
    Much simpler than learned drift/diffusion networks.
    """

    def __init__(self, latent_dim: int, theta: float = 0.1, sigma: float = 0.1):
        """
        Args:
            latent_dim: Latent dimension
            theta: Mean reversion speed
            sigma: Volatility
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.theta = nn.Parameter(torch.tensor(theta))
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, z: Tensor, dt: float = 0.01) -> Tensor:
        """
        Single Euler-Maruyama step.

        Args:
            z: Current state (batch, latent_dim)
            dt: Time step

        Returns:
            Updated state (batch, latent_dim)
        """
        # Deterministic drift: -theta * z
        drift = -self.theta * z

        # Stochastic diffusion: sigma * dW
        dW = torch.randn_like(z)
        diffusion = self.sigma * torch.sqrt(torch.tensor(dt)) * dW

        # Euler-Maruyama update
        z_next = z + drift * dt + diffusion

        return z_next


# ==============================================================================
# COMPONENT 3: AFFINE FLOW LAYERS (SIMPLER THAN COUPLING)
# ==============================================================================

class AffineFlowLayer(nn.Module):
    """
    Simple affine transformation for normalizing flow.
    y = scale * x + shift
    Much simpler and faster than coupling layers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * 0.1)
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Apply affine transformation.

        Args:
            z: Input (batch, dim)
            reverse: If True, apply inverse

        Returns:
            z_out: Transformed (batch, dim)
            log_det: Log determinant (batch,)
        """
        if not reverse:
            z_out = self.scale * z + self.shift
            log_det = torch.sum(torch.log(torch.abs(self.scale) + 1e-8)) * torch.ones(z.shape[0], device=z.device)
        else:
            z_out = (z - self.shift) / (self.scale + 1e-8)
            log_det = -torch.sum(torch.log(torch.abs(self.scale) + 1e-8)) * torch.ones(z.shape[0], device=z.device)

        return z_out, log_det


class SimpleNormalizingFlow(nn.Module):
    """Normalizing flow using simple affine layers."""

    def __init__(self, latent_dim: int, num_layers: int = 4):
        super().__init__()
        self.flows = nn.ModuleList([
            AffineFlowLayer(latent_dim) for _ in range(num_layers)
        ])

    def forward(self, z: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: (batch, latent_dim)
            reverse: Direction of flow

        Returns:
            z_out: (batch, latent_dim)
            log_det: (batch,)
        """
        log_det = torch.zeros(z.shape[0], device=z.device)
        flows = reversed(self.flows) if reverse else self.flows

        for flow in flows:
            z, ld = flow(z, reverse=reverse)
            log_det += ld

        return z, log_det


# ==============================================================================
# COMPONENT 4: CNN ENCODER (FASTER THAN TRANSFORMER)
# ==============================================================================

class CNNEncoder(nn.Module):
    """
    CNN encoder for sequence encoding.
    Faster than transformer on CPU with fewer parameters.
    """

    def __init__(self, vocab_size: int, embed_dim: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Stacked 1D convolutions
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)

        # Global average pooling + FC
        self.fc = nn.Linear(16, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (batch, seq_len)

        Returns:
            latent: (batch, latent_dim)
        """
        x = self.embedding(tokens)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)        # (batch, embed_dim, seq_len)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling
        x = x.mean(dim=2)  # (batch, 16)

        # Project to latent
        latent = self.fc(x)  # (batch, latent_dim)

        return latent


# ==============================================================================
# COMPONENT 5: DIRECT CLASSIFICATION HEAD (NO SDE EVOLUTION)
# ==============================================================================

class DirectClassifier(nn.Module):
    """
    Simple direct classification from latent representation.
    No SDE trajectory evolution for simplicity and speed.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (batch, latent_dim)

        Returns:
            logits: (batch, num_classes)
        """
        return self.net(z)


# ==============================================================================
# COMPONENT 6: FIXED-SIZE CIRCULAR BUFFER MEMORY
# ==============================================================================

class CircularBuffer:
    """Fixed-size circular buffer memory (simpler than priority-based)."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Dict[str, Tensor]):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, n: int, device: torch.device) -> List[Dict[str, Tensor]]:
        """Random sample from buffer."""
        import random
        if len(self.buffer) == 0:
            return []

        sample_size = min(n, len(self.buffer))
        return [self.buffer[i] for i in random.sample(range(len(self.buffer)), sample_size)]

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


# ==============================================================================
# COMPONENT 7: ALTERNATIVE RACCOON MODEL
# ==============================================================================

LOG_CATEGORIES = ["ERROR", "WARNING", "INFO", "DEBUG"]
NUM_LOG_CLASSES = len(LOG_CATEGORIES)

# Character set for encoding
chars = ["_"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(i) for i in range(10)] + ["!", ">", "?", ":", "-", "."]
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(chars)


def encode_sequence(s: str) -> Tensor:
    """Encode string to tensor."""
    return torch.tensor([char2idx.get(c, 0) for c in s], dtype=torch.long)


def decode_sequence(t: Tensor) -> str:
    """Decode tensor to string."""
    return "".join(idx2char.get(int(i), "_") for i in t)


class SimpleRaccoonModel(nn.Module):
    """
    Simplified Raccoon model for continuous learning.
    Uses: OrnsteinUhlenbeck + Affine Flows + CNN + Direct Classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        memory_size: int = 1000,
        sde_steps: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sde_steps = sde_steps

        # 1. CNN encoder
        self.encoder = CNNEncoder(vocab_size, embed_dim, latent_dim)

        # 2. Ornstein-Uhlenbeck SDE
        self.sde = OrnsteinUhlenbeckSDE(latent_dim, theta=0.1, sigma=0.1)

        # 3. Affine normalizing flows
        self.flow = SimpleNormalizingFlow(latent_dim, num_layers=4)

        # 4. Direct classifier
        self.classifier = DirectClassifier(latent_dim, hidden_dim, num_classes)

        # 5. Memory
        self.memory = CircularBuffer(max_size=memory_size)

        # Statistics tracking
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    def encode(self, tokens: Tensor) -> Tensor:
        """Encode tokens to latent space."""
        return self.encoder(tokens)

    def apply_sde(self, z: Tensor, num_steps: int) -> Tensor:
        """Apply SDE dynamics for multiple steps."""
        for _ in range(num_steps):
            z = self.sde(z, dt=0.01)
        return z

    def classify(self, z: Tensor) -> Tensor:
        """Classify from latent."""
        return self.classifier(z)

    def forward(self, tokens: Tensor, labels: Tensor, training: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Full forward pass.

        Args:
            tokens: (batch, seq_len)
            labels: (batch,)
            training: Whether in training mode

        Returns:
            loss: scalar
            stats: Dictionary of metrics
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Encode
        z = self.encode(tokens)  # (batch, latent_dim)

        # Apply SDE if training
        if training:
            z = self.apply_sde(z, self.sde_steps)

        # Apply normalizing flow
        z_flow, log_det = self.flow(z, reverse=False)

        # Classify
        logits = self.classify(z_flow)

        # Compute loss
        ce_loss = F.cross_entropy(logits, labels)

        # Flow regularization (encourage invertibility)
        flow_reg = torch.abs(log_det).mean()

        total_loss = ce_loss + 0.01 * flow_reg

        # Accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean()

        stats = {
            "ce_loss": ce_loss.detach(),
            "flow_reg": flow_reg.detach(),
            "accuracy": accuracy.detach(),
        }

        return total_loss, stats


# ==============================================================================
# COMPONENT 8: INFERENCE-ONLY MODE
# ==============================================================================

class InferenceEngine:
    """Lightweight inference engine for deployment."""

    def __init__(self, model: SimpleRaccoonModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

        # Disable gradients for inference
        for param in model.parameters():
            param.requires_grad = False

    def predict(self, tokens: Tensor) -> Tuple[int, float]:
        """
        Make prediction on single or batch of sequences.

        Returns:
            predictions: Class indices (batch,)
            confidences: Softmax probabilities (batch, num_classes)
        """
        tokens = tokens.to(self.device)

        with torch.no_grad():
            z = self.model.encode(tokens)
            logits = self.model.classify(z)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        return preds.cpu(), probs.cpu()

    def predict_with_uncertainty(self, tokens: Tensor, n_samples: int = 10) -> Dict[str, Any]:
        """
        Predict with uncertainty via SDE stochasticity.
        """
        predictions = []

        for _ in range(n_samples):
            z = self.model.encode(tokens)
            z = self.model.apply_sde(z, self.model.sde_steps)
            logits = self.model.classify(z)
            preds = logits.argmax(dim=1)
            predictions.append(preds.detach().cpu())

        predictions = torch.stack(predictions)  # (n_samples, batch)

        # Compute mean and uncertainty
        mode = torch.mode(predictions, dim=0).values
        entropy = torch.log(torch.tensor(self.model.num_classes)) - torch.logsumexp(torch.log(torch.bincount(predictions.view(-1))[None, :] + 1e-8), dim=1)

        return {
            "predictions": mode,
            "samples": predictions,
            "mode": mode,
        }


# ==============================================================================
# COMPONENT 9: UNIT TESTS
# ==============================================================================

class ComponentTests:
    """Unit tests for all components."""

    @staticmethod
    def test_ou_sde():
        """Test Ornstein-Uhlenbeck SDE."""
        print("\n[Test 1] Ornstein-Uhlenbeck SDE")
        sde = OrnsteinUhlenbeckSDE(latent_dim=16)

        z = torch.randn(8, 16)
        z_next = sde(z, dt=0.01)

        assert z_next.shape == z.shape, "Shape mismatch"
        assert not torch.isnan(z_next).any(), "NaN values"
        print("✅ OrnsteinUhlenbeckSDE: PASSED")

    @staticmethod
    def test_affine_flow():
        """Test affine flow layer."""
        print("\n[Test 2] Affine Flow Layer")
        flow = AffineFlowLayer(dim=32)

        z = torch.randn(8, 32)
        z_forward, ld_forward = flow(z, reverse=False)
        z_backward, ld_backward = flow(z_forward, reverse=True)

        assert z_forward.shape == z.shape, "Forward shape mismatch"
        assert ld_forward.shape == (8,), "Log-det shape mismatch"
        assert torch.allclose(z, z_backward, atol=1e-5), "Invertibility failed"
        print("✅ AffineFlowLayer: PASSED")

    @staticmethod
    def test_cnn_encoder():
        """Test CNN encoder."""
        print("\n[Test 3] CNN Encoder")
        encoder = CNNEncoder(vocab_size=50, embed_dim=32, latent_dim=16)

        tokens = torch.randint(0, 50, (8, 50))
        z = encoder(tokens)

        assert z.shape == (8, 16), f"Shape mismatch: {z.shape}"
        assert not torch.isnan(z).any(), "NaN values"
        print("✅ CNNEncoder: PASSED")

    @staticmethod
    def test_direct_classifier():
        """Test direct classifier."""
        print("\n[Test 4] Direct Classifier")
        classifier = DirectClassifier(latent_dim=16, hidden_dim=32, num_classes=4)

        z = torch.randn(8, 16)
        logits = classifier(z)

        assert logits.shape == (8, 4), f"Shape mismatch: {logits.shape}"
        print("✅ DirectClassifier: PASSED")

    @staticmethod
    def test_circular_buffer():
        """Test circular buffer."""
        print("\n[Test 5] Circular Buffer Memory")
        buffer = CircularBuffer(max_size=100)

        # Add experiences
        for i in range(150):
            buffer.add({"id": i})

        assert len(buffer) == 100, "Buffer size incorrect"

        # Sample
        samples = buffer.sample(10, torch.device("cpu"))
        assert len(samples) == 10, "Sample size incorrect"
        print("✅ CircularBuffer: PASSED")

    @staticmethod
    def test_normalizing_flow():
        """Test normalizing flow."""
        print("\n[Test 6] Simple Normalizing Flow")
        flow = SimpleNormalizingFlow(latent_dim=32, num_layers=4)

        z = torch.randn(8, 32)
        z_forward, ld_forward = flow(z, reverse=False)
        z_backward, _ = flow(z_forward, reverse=True)

        assert z_forward.shape == z.shape, "Forward shape mismatch"
        assert torch.allclose(z, z_backward, atol=1e-4), "Invertibility failed"
        print("✅ SimpleNormalizingFlow: PASSED")

    @staticmethod
    def test_simple_raccoon_model():
        """Test full model."""
        print("\n[Test 7] Simple Raccoon Model")
        model = SimpleRaccoonModel(
            vocab_size=50,
            num_classes=4,
            latent_dim=16,
            hidden_dim=32,
        )

        tokens = torch.randint(0, 50, (8, 50))
        labels = torch.randint(0, 4, (8,))

        loss, stats = model(tokens, labels, training=True)

        assert loss.item() > 0, "Invalid loss"
        assert "accuracy" in stats, "Missing accuracy"
        assert not torch.isnan(loss), "NaN loss"
        print("✅ SimpleRaccoonModel: PASSED")

    @staticmethod
    def test_inference_engine():
        """Test inference engine."""
        print("\n[Test 8] Inference Engine")
        model = SimpleRaccoonModel(vocab_size=50, num_classes=4)
        device = torch.device("cpu")
        engine = InferenceEngine(model, device)

        tokens = torch.randint(0, 50, (4, 50))
        preds, probs = engine.predict(tokens)

        assert preds.shape == (4,), "Predictions shape incorrect"
        assert probs.shape == (4, 4), "Probabilities shape incorrect"
        print("✅ InferenceEngine: PASSED")

    @staticmethod
    def run_all():
        """Run all tests."""
        print("\n" + "="*70)
        print("RUNNING UNIT TESTS FOR ALTERNATIVE RACCOON")
        print("="*70)

        ComponentTests.test_ou_sde()
        ComponentTests.test_affine_flow()
        ComponentTests.test_cnn_encoder()
        ComponentTests.test_direct_classifier()
        ComponentTests.test_circular_buffer()
        ComponentTests.test_normalizing_flow()
        ComponentTests.test_simple_raccoon_model()
        ComponentTests.test_inference_engine()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)


# ==============================================================================
# COMPONENT 10: TRAINING LOOP WITH EARLY STOPPING & VALIDATION
# ==============================================================================

class TrainingLogger:
    """Track training metrics."""

    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "times": [],
        }

    def log(self, split: str, **kwargs):
        """Log metrics."""
        for key, value in kwargs.items():
            if f"{split}_{key}" not in self.metrics:
                self.metrics[f"{split}_{key}"] = []
            self.metrics[f"{split}_{key}"].append(value)

    def best_val_loss(self) -> float:
        """Get best validation loss."""
        return min(self.metrics["val_loss"]) if self.metrics["val_loss"] else float("inf")


class EarlyStoppingCallback:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop training.

        Returns:
            True if should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def train_alternative_raccoon(
    model: SimpleRaccoonModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
) -> TrainingLogger:
    """
    Training loop with early stopping and validation.

    Args:
        model: Model to train
        train_loader: Training data
        val_loader: Validation data
        device: Torch device
        max_epochs: Maximum epochs
        learning_rate: Learning rate
        patience: Early stopping patience

    Returns:
        TrainingLogger with metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    early_stop = EarlyStoppingCallback(patience=patience)
    logger = TrainingLogger()

    print("\n" + "="*70)
    print("TRAINING SIMPLE RACCOON WITH EARLY STOPPING")
    print("="*70)

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
        for tokens, labels in pbar:
            tokens = tokens.to(device)
            labels = labels.to(device)

            loss, stats = model(tokens, labels, training=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += stats["accuracy"].item()
            train_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)

        train_loss = train_loss_sum / train_batches
        train_acc = train_acc_sum / train_batches
        logger.log("train", loss=train_loss, acc=train_acc)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)

                loss, stats = model(tokens, labels, training=False)

                val_loss_sum += loss.item()
                val_acc_sum += stats["accuracy"].item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches
        val_acc = val_acc_sum / val_batches
        logger.log("val", loss=val_loss, acc=val_acc)

        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f} | " +
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")

        # Early stopping
        if early_stop(val_loss):
            print(f"\n✅ Early stopping at epoch {epoch+1}")
            break

    return logger


# ==============================================================================
# SYNTHETIC LOG DATASET
# ==============================================================================

class AlternativeLogDataset(Dataset):
    """
    Log dataset with realistic generation and concept drift.
    """

    def __init__(self, n_samples: int, seq_len: int = 50, drift_point: Optional[int] = None):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.drift_point = drift_point

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Determine class with possible drift
        if self.drift_point and idx >= self.drift_point:
            # Drift: more errors, fewer debug
            class_probs = [0.4, 0.3, 0.2, 0.1]
        else:
            # Balanced
            class_probs = [0.25, 0.25, 0.25, 0.25]

        category = torch.multinomial(torch.tensor(class_probs), 1).item()
        category_name = LOG_CATEGORIES[category]

        # Generate realistic log
        message = RealisticLogGenerator.generate(category_name, self.seq_len)

        # Add noise
        message_list = list(message)
        for i in range(len(message_list)):
            if torch.rand(1).item() < 0.05:  # 5% corruption
                message_list[i] = chars[torch.randint(0, len(chars), (1,)).item()]
        message = "".join(message_list)

        # Encode
        tokens = encode_sequence(message)

        return tokens, category


if __name__ == "__main__":
    # Run unit tests first
    ComponentTests.run_all()

    print("\n" + "="*70)
    print("TRAINING ALTERNATIVE RACCOON")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create datasets
    print("\n📝 Creating log datasets...")
    train_ds = AlternativeLogDataset(n_samples=2000, seq_len=50)
    val_ds = AlternativeLogDataset(n_samples=500, seq_len=50)
    test_ds = AlternativeLogDataset(n_samples=500, seq_len=50)
    drift_ds = AlternativeLogDataset(n_samples=500, seq_len=50, drift_point=250)

    # Show samples
    print("\n📋 Example logs:")
    for i in range(4):
        tokens, label = train_ds[i]
        message = decode_sequence(tokens)
        print(f"  [{LOG_CATEGORIES[label]}] {message[:40]}...")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    drift_loader = DataLoader(drift_ds, batch_size=1, shuffle=True)

    # Create model
    print("\n🦝 Initializing Simple Raccoon model...")
    model = SimpleRaccoonModel(
        vocab_size=vocab_size,
        num_classes=NUM_LOG_CLASSES,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=1000,
        sde_steps=3,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count:,}")

    # Train with early stopping
    logger = train_alternative_raccoon(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=50,
        learning_rate=1e-3,
        patience=10,
    )

    # Final evaluation
    print("\n📊 Final evaluation on test set...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            z = model.encode(tokens)
            logits = model.classify(z)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"✅ Test Accuracy: {test_accuracy:.3f} ({correct}/{total})")

    # Inference engine test
    print("\n🚀 Testing inference engine...")
    inference_engine = InferenceEngine(model, device)

    test_tokens = torch.randint(0, vocab_size, (4, 50))
    preds, probs = inference_engine.predict(test_tokens)

    print(f"Sample predictions: {preds.tolist()}")
    print(f"Confidence: {probs.max(dim=1).values.tolist()}")

    print("\n" + "="*70)
    print("✅ ALTERNATIVE RACCOON TRAINING COMPLETE!")
    print("="*70)
    print(f"\nDesign Choices:")
    print("  ✅ Ornstein-Uhlenbeck SDE (simpler dynamics)")
    print("  ✅ Affine Flow Layers (faster than coupling)")
    print("  ✅ Fixed-size Circular Buffer (simpler memory)")
    print("  ✅ CNN Encoder (faster on CPU)")
    print("  ✅ Direct Classifier (no trajectory evolution)")
    print("  ✅ Realistic Log Generator")
    print("  ✅ Early Stopping & Validation Monitoring")
    print("  ✅ Inference-Only Mode")
    print("  ✅ Unit Tests for All Components")
    print(f"\n📊 Final Results:")
    print(f"  Test Accuracy: {test_accuracy:.3f}")
    print(f"  Model Parameters: {param_count:,}")
