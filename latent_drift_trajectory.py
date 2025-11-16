import math
from typing import Any, Callable
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch import distributions as D
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist

from tqdm import trange
from torch.nn import functional as F 

# ──────────────────────────────────────────────────────────────────────────────
#  Character-level dataset: Σ = {_, A..Z, !, >}
# ──────────────────────────────────────────────────────────────────────────────

chars = ["_"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["!", ">", "?"]
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(chars)  # 29


def encode(s: str) -> Tensor:
    return torch.tensor([char2idx[c] for c in s], dtype=torch.long)


def decode(t: Tensor) -> str:
    return "".join(idx2char[int(i)] for i in t)


class SyntheticTargetDataset(Dataset):
    # Each sample: length-66 tensor (2-char prompt + 64 seq)
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.T = 64  # base length
        self.L = 8  # target block length
        self.p_noise = 1.0 / 16.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        T = self.T
        L = self.L

        # 1. start with a string of 64 underscores
        seq = torch.full((T,), char2idx["_"], dtype=torch.long)

        # 2. pick uppercase letter and position, insert 8×letter
        letter = chr(ord("A") + torch.randint(0, 26, (1,)).item())
        letter_token = char2idx[letter]
        start = torch.randint(0, T - L + 1, (1,)).item()
        seq[start : start + L] = letter_token

        # 3. replace any character with '!' with probability 1/16
        noise_mask = torch.rand(T) < self.p_noise
        seq[noise_mask] = char2idx["!"]

        # 4. concatenate prompt: target letter + '>'
        prompt = torch.tensor([char2idx["?"], letter_token, char2idx[">"]], dtype=torch.long)
        full_seq = torch.cat([prompt, seq], dim=0)  # shape: (66,)

        return full_seq

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def all_reduce(x: Tensor, op: str = "AVG") -> Tensor:
    if is_dist_avail_and_initialized():
        op_enum = getattr(dist.ReduceOp, op.upper())
        dist.all_reduce(x, op=op_enum)
        if op.upper() == "AVG":
            x /= dist.get_world_size()
    return x


class UnivariateTest(nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0.0, 1.0)

    def prepare_data(self, x: Tensor) -> Tensor:
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x: Tensor) -> Tensor:
        return all_reduce(x, op="AVG")

    @property
    def world_size(self) -> int:
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1


class FastEppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley test statistic for univariate normality.

    Expects input x of shape (*, N, K) where N is the sample dimension
    and K is the "slice" dimension. Returns (*, K) test statistics.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17, integration: str = "trapezoid"):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)  # positive half, incl 0
        dt = t_max / (n_points - 1)

        # FIXED: Separate quadrature weights from weight function
        # Trapezoid quadrature weights (NOT multiplied by phi)
        quad_weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        quad_weights[0] = dt
        quad_weights[-1] = dt

        # Weight function (uniform for standard EP test)
        w = torch.ones_like(t)

        phi = (-0.5 * t.square()).exp()

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        # FIXED: Correctly combine quadrature and weight function
        self.register_buffer("weights", quad_weights * w)

    def forward(self, x: Tensor) -> Tensor:
        # x: (*, N, K)
        N = x.size(-2)

        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # Mean across samples (dimension -3 = N)
        cos_mean = cos_vals.mean(-3)  # (*, K, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, K, n_points)

        # DDP reduction
        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        # Error against standard normal characteristic function
        err = (cos_mean - self.phi) ** 2 + sin_mean**2  # (*, K, n_points)

        # Weighted integration over t (last dim)
        stats = err @ self.weights  # (*, K)

        return stats * N * self.world_size


class EppsPulleyCF(UnivariateTest):
    """
    Alternative Epps-Pulley normality test via characteristic functions.
    """

    def __init__(self, t_range=(-3, 3), n_points=10, weight_type="gaussian"):
        super().__init__()
        self.t_range = t_range
        self.n_points = n_points
        self.weight_type = weight_type

    def empirical_cf(self, x: Tensor, t: Tensor) -> Tensor:
        # x: (..., N), t: (M,)
        x_expanded = x.unsqueeze(-1)  # (..., N, 1)
        t_expanded = t.view(*([1] * x.ndim), -1)  # (..., 1, M)

        real_part = torch.cos(t_expanded * x_expanded)
        imag_part = torch.sin(t_expanded * x_expanded)

        empirical_real = real_part.mean(dim=-2)  # (..., M)
        empirical_imag = imag_part.mean(dim=-2)  # (..., M)

        return torch.complex(empirical_real.float(), empirical_imag.float())

    def normal_cf(self, t: Tensor, mu: float, sigma: float) -> Tensor:
        magnitude = torch.exp(-0.5 * (sigma**2) * t**2)
        phase = mu * t

        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        return torch.complex(real_part.float(), imag_part.float())

    def weight_function(self, t: Tensor) -> Tensor:
        if self.weight_type == "gaussian":
            return torch.exp(-(t**2) / 2)
        elif self.weight_type == "uniform":
            return torch.ones_like(t)
        else:
            raise ValueError(f"Unknown weight type: {self.weight_type}")

    def forward(self, x: Tensor) -> Tensor:
        device = x.device

        with torch.no_grad():
            t_min, t_max = self.t_range
            t = torch.linspace(t_min, t_max, self.n_points, device=device)

            phi_normal = self.normal_cf(t, mu=0.0, sigma=1.0)
            weights = self.weight_function(t)

            # Broadcast over extra dims of x (all except last sample dim)
            for _ in range(x.ndim - 1):
                phi_normal = phi_normal.unsqueeze(-1)
                weights = weights.unsqueeze(-1)

        phi_emp = self.empirical_cf(x, t)
        diff = phi_emp - phi_normal
        squared_diff = torch.real(diff * torch.conj(diff))

        integrand = squared_diff * weights
        integral = torch.trapz(integrand, t, dim=-1)

        return integral


class SlicingUnivariateTest(nn.Module):
    """
    Multivariate test by random slicing + univariate test.

    Input: (*, N, D)
    Output:
        - scalar if reduction='mean' or 'sum'
        - (*, num_slices) if reduction=None
    """

    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float | None = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed: int):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: Tensor) -> Tensor:
        # x: (*, N, D)
        with torch.no_grad():
            # Synchronize global_step across ranks
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = int(global_step_sync.item())

            dev = dict(device=x.device)
            g = self._get_generator(x.device, seed)

            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, **dev, generator=g)
            A /= A.norm(p=2, dim=0) + 1e-12

            self.global_step.add_(1)

        # Project and run univariate test
        # x @ A: (*, N, num_slices)
        stats = self.univariate_test(x @ A)

        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, stats.new_zeros(()), stats)

        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")



# ──────────────────────────────────────────────────────────────────────────────
#  ODE base class + solver (deterministic latent dynamics)
# ──────────────────────────────────────────────────────────────────────────────


class ODE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        # so we can call ode(z, t)
        return self.drift(z, t, *args)


def solve_ode(
    ode: Callable[[Tensor, Tensor], Tensor] | ODE,
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int,
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)
    dt = (tf - ts) / n_steps

    path = [z]
    for t in tt[:-1]:
        f = ode(z, t)
        z = z + f  * dt
        path.append(z)

    return torch.stack(path)  # (L, B, latent)


# ──────────────────────────────────────────────────────────────────────────────
#  Prior (generative) process in latent space
# ──────────────────────────────────────────────────────────────────────────────


class PriorInitDistribution(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()

        self.m = nn.Parameter(torch.zeros(1, latent_size))
        self.log_s = nn.Parameter(torch.zeros(1, latent_size))

    def forward(self) -> D.Distribution:
        m = self.m
        s = torch.exp(self.log_s)
        return D.Independent(D.Normal(m, s), 1)


class PriorODE(ODE):
    """
    IMPROVED: Shallower network (5 layers instead of 11) with better initialization
    for improved gradient flow and numerical stability.
    """
    def __init__(self, latent_size: int, hidden_size: int, depth: int = 5):
        super().__init__()

        layers = []
        input_dim = latent_size + 1
        # FIXED: Use depth=5 instead of 11 for better gradient flow
        for i in range(depth):
            linear = nn.Linear(input_dim, hidden_size)
            # IMPROVED: Better initialization gain for gradient flow (from numerical-stability agent)
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            input_dim = hidden_size

        # FIXED: Add normalization before final layer
        layers.append(nn.LayerNorm(hidden_size))
        final_linear = nn.Linear(hidden_size, latent_size)
        # FIXED: Use orthogonal init for output layer
        nn.init.orthogonal_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        layers.append(final_linear)
        self.drift_net = nn.Sequential(*layers)


    def drift(self, z: Tensor, t: Tensor, *args) -> Tensor:
        if t.ndim == 0:
            t = t.reshape(1, 1).expand(z.shape[0], 1)
        return self.drift_net(torch.cat([z, t], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
#  Discrete observation model: z_t → categorical over Σ
# ──────────────────────────────────────────────────────────────────────────────
class DiscreteObservation(nn.Module):
    """
    Autoregressive transformer decoder:
        p(x_t | x_{<t}, z_{0:t})

    Uses a single causal TransformerBlock, conditioned on:
        - latent path z_t
        - embeddings of previous tokens
    """

    def __init__(
        self,
        latent_size: int,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        nb_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, embed_size)

        self.latent_proj = nn.Linear(latent_size, hidden_size)
        self.token_proj = nn.Linear(embed_size, hidden_size)

        self.pos_enc = AddPositionalEncoding(len_max=1e5)

        self.block = TransformerBlock(
            dim_model=hidden_size,
            dim_keys=hidden_size // nb_heads,
            dim_hidden=hidden_size,
            nb_heads=nb_heads,
            causal=True,          # AR
            dropout=dropout,
        )

        self.proj_out = nn.Linear(hidden_size, vocab_size)

    def get_logits(self, z: Tensor, tokens: Tensor) -> Tensor:
        """
        z:      (B, L, D)
        tokens: (B, L)  – target tokens (teacher forcing)

        At position t, we predict tokens[:, t] based on:
            - latents z[:, :t+1]
            - previous tokens tokens[:, :t]
        """
        B, L, D = z.shape

        # Shift tokens right: input at position t is token at t-1
        tokens_in = tokens.roll(1, dims=1)
        start_token_id = char2idx["_"]
        tokens_in[:, 0] = start_token_id

        tok_emb = self.token_emb(tokens_in)      # (B, L, E)

        h = self.latent_proj(z) + self.token_proj(tok_emb)  # (B, L, H)
        h = self.pos_enc(h)
        h = self.block(h)
        logits = self.proj_out(h)                # (B, L, V)

        return logits

    def forward(self, z: Tensor, tokens: Tensor) -> D.Distribution:
        logits = self.get_logits(z, tokens)      # (B, L, V)
        return D.Categorical(logits=logits.reshape(-1, self.vocab_size))

# ──────────────────────────────────────────────────────────────────────────────
#  Posterior encoder (used as deterministic sequence encoder)
# ──────────────────────────────────────────────────────────────────────────────


class PosteriorEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, embed_size)

        # Project embedding dimension → transformer model dimension if needed
        if embed_size != hidden_size:
            self.in_proj = nn.Linear(embed_size, hidden_size)
        else:
            self.in_proj = nn.Identity()

        # Pico-style transformer defaults
        nb_heads = 4
        nb_blocks = 4
        dropout = 0.0
        causal = False 
        self.pos_enc = AddPositionalEncoding(len_max=1e5)

        self.trunk = nn.Sequential(
            *[
                TransformerBlock(
                    dim_model=hidden_size,
                    dim_keys=hidden_size // nb_heads,
                    dim_hidden=hidden_size,
                    nb_heads=nb_heads,
                    causal=causal,
                    dropout=dropout,
                )
                for _ in range(nb_blocks)
            ]
        )

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (B, L)
        x = self.emb(tokens)          # (B, L, E)
        x = self.in_proj(x)           # (B, L, H)
        x = self.pos_enc(x)           # (B, L, H)
        x = self.trunk(x)             # (B, L, H)


        return x


import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DeterministicEncoder(nn.Module):
    """
    Deterministic latent encoder: tokens → latent sequence z_{0:L-1}.
    Uses the GRU encoder above and projects to latent_size.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        latent_size: int,
        smear_kernel_size: int =3,
        smear_sigma: float = 2.0,
    ):
        super().__init__()
        self.ctx_encoder = PosteriorEncoder(vocab_size, embed_size, hidden_size)
        self.proj = nn.Linear(hidden_size, latent_size)

        # Build a fixed Gaussian kernel on index distance: k(d) ∝ exp(-d² / (2σ²))
        t = torch.arange(smear_kernel_size, dtype=torch.float32)
        center = (smear_kernel_size - 1) / 2.0
        kernel = torch.exp(-0.5 * ((t - center) / smear_sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.register_buffer("smear_kernel", kernel)
        self.smear_pad = smear_kernel_size // 2

    def local_smooth(self, z: Tensor) -> Tensor:
        # z: (B, L, D); smear along L, independently per feature dim
        B, L, D = z.shape

        z_t = z.permute(0, 2, 1)  # (B, D, L)
        z_t = F.pad(z_t, (self.smear_pad, self.smear_pad), mode="reflect")  # (B, D, L+2p)

        k = self.smear_kernel.view(1, 1, -1)          # (1, 1, K)
        k = k.expand(D, 1, -1)                        # (D, 1, K) – depthwise
        z_s = F.conv1d(z_t, k, groups=D)              # (B, D, L)

        return z_s.permute(0, 2, 1)                   # (B, L, D)

    def forward(self, tokens: Tensor) -> Tensor:
        ctx = self.ctx_encoder(tokens)  # (B, L+1, H) in your comment

        z = self.proj(ctx)              # (B, L+1, latent)
        #z = self.local_smooth(z)        # smear along sequence indices

        return z


# ──────────────────────────────────────────────────────────────────────────────
#  Deterministic latent ODE model: encoder + ODE regression + EP regulariser
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  Pico-style transformer encoder blocks
# ──────────────────────────────────────────────────────────────────────────────


class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max: float):
        super().__init__()
        self.len_max = len_max

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        u = torch.arange(x.size(1), device=x.device)[:, None]        # (T, 1)
        j = torch.arange(x.size(2), device=x.device)[None, :]        # (1, C)
        k = j % 2
        t = u / (self.len_max ** ((j - k) / x.size(2))) + math.pi / 2 * k
        return x + torch.sin(t)                                      # broadcast to (T, C)


class QKVAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, nb_heads=1, causal=False, dropout=0.0):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.dropout = dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, T, C)
        q = torch.einsum("ntc,hdc->nhtd", x, self.w_q)
        k = torch.einsum("ntc,hdc->nhtd", x, self.w_k)
        v = torch.einsum("ntc,hdc->nhtd", x, self.w_v)

        a = torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(self.w_q.size(1))

        if self.causal:
            t = torch.arange(x.size(1), device=x.device)
            attzero = t[None, None, :, None] < t[None, None, None, :]
            a = a.masked_fill(attzero, float("-inf"))

        a = a.softmax(dim=3)
        a = F.dropout(a, self.dropout, self.training)
        y = torch.einsum("nhts,nhsd->nthd", a, v).flatten(2)  # (N, T, H*dim_v)

        y = y @ self.w_o  # (N, T, C)

        return y


class TransformerBlock(nn.Module):
    def __init__(self, dim_model, dim_keys, dim_hidden, nb_heads, causal, dropout):
        super().__init__()
        self.att_ln = nn.LayerNorm((dim_model,))
        self.att_mh = QKVAttention(
            dim_in=dim_model,
            dim_qk=dim_keys,
            dim_v=dim_model // nb_heads,
            nb_heads=nb_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ffn_ln = nn.LayerNorm((dim_model,))
        self.ffn_fc1 = nn.Linear(dim_model, dim_hidden)
        self.ffn_fc2 = nn.Linear(dim_hidden, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C)
        r = x

        x = self.att_ln(r)
        x = self.att_mh(x)
        r = r + x

        x = self.ffn_ln(r)
        x = self.ffn_fc1(x)
        x = F.relu(x)
        x = self.ffn_fc2(x)
        r = r + x

        return r

class Predictor(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int):
        super().__init__()
        self.pos_enc = AddPositionalEncoding(len_max=1e5)
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_size),
        )
    def forward(self, z: Tensor) -> Tensor:
        z = self.pos_enc(z)
        return self.net(z)

class DeterministicLatentODE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_size: int,
        hidden_size: int,
        embed_size: int,
        num_slices: int = 64,
    ):
        super().__init__()

        self.latent_size = latent_size

        self.encoder = DeterministicEncoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
        )
        self.p_init_distr = PriorInitDistribution(latent_size)
        self.p_ode = PriorODE(latent_size, hidden_size)

        # AR decoder: 1-layer causal transformer
        self.p_observe = DiscreteObservation(
            latent_size=latent_size,
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            nb_heads=4,
            dropout=0.0,
        )

        univariate_test = FastEppsPulley(t_max=5.0, n_points=17)
        self.latent_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=num_slices,
            reduction="mean",
        )

        self.predictor = Predictor(latent_size, hidden_size)

    def encode(self, tokens: Tensor) -> Tensor:
        return self.encoder(tokens)

    def decode_logits(self, z: Tensor, tokens: Tensor) -> Tensor:
        return self.p_observe.get_logits(z, tokens)

    def ode_matching_loss(self, z: Tensor) -> Tensor:
        # z: (B, L, D); treat discrete latent path as Euler samples of an ODE
        B, L, D = z.shape
        if L < 2:
            return z.new_zeros(())

        dt = 1.0 / (L - 1)

        # (B, L-1, D)
        z_t = z[:, :-1, :]
        z_next = z[:, 1:, :]
        dz_true = (z_next - z_t)

        # time grid t ∈ [0, 1]
        t_grid = torch.linspace(0.0, 1.0, L, device=z.device, dtype=z.dtype)
        t_t = t_grid[:-1].view(1, L - 1, 1).expand(B, L - 1, 1)

        # flatten batch × time
        z_t_flat = z_t.reshape(-1, D)
        t_t_flat = t_t.reshape(-1, 1)
        dz_true_flat = dz_true.reshape(-1, D)

        # predicted increments from ODE
        f = self.p_ode(z_t_flat, t_t_flat)  # drift fθ(z_t, t_t)
        dz_pred_flat = f * dt               # Euler step: f Δt

        resid = dz_pred_flat - dz_true_flat      # (B*(L-1), D)
        ode_loss = resid.abs().mean()            # scalar
        z_pred = (z_t_flat.detach() + dz_pred_flat).reshape_as(z_t)

        return ode_loss, z_pred

    def loss_components(self, tokens: Tensor):
        bs, seq_len = tokens.shape

        # Deterministic latent path z_t
        z = self.encoder(tokens)  # (B, L, latent)
        latent_size = z.shape[-1]



        # Latent normality regulariser via sliced Epps-Pulley 
        z_for_test = z.reshape(1, -1, latent_size)  # (1, N, D)
        latent_stat = self.latent_test(z_for_test)
        latent_reg = latent_stat.mean()

        # ODE regression: match local dynamics z_{t+1} - z_t
        ode_reg_loss, z_pred = self.ode_matching_loss(z)

        z_for_test = z_pred.reshape(1, -1, latent_size)  # (1, N, D)
        latent_stat = self.latent_test(z_for_test)  # FIXED: Use reshaped tensor
        latent_reg = latent_stat.mean() + latent_reg

        p_x = self.p_observe(torch.cat([z[:, :1, :], z_pred], dim=1), tokens)
        recon_loss = -p_x.log_prob(tokens.reshape(-1)).mean()

        return recon_loss, latent_reg, ode_reg_loss

    def forward(
        self,
        tokens: Tensor,
        loss_weights: tuple[float, float, float, float] = (1.0, 0.1, 1.0),
    ):
        recon_loss, latent_reg, ode_reg_loss = self.loss_components(tokens)
        w_recon, w_latent, w_ode = loss_weights

        loss = (
            w_recon * recon_loss
            + w_latent * latent_reg
            + w_ode * ode_reg_loss
        )

        stats = {
            "recon": recon_loss.detach(),
            "latent_ep": latent_reg.detach(),
            "ode_reg": ode_reg_loss.detach(),
        }

        return loss, stats


# ──────────────────────────────────────────────────────────────────────────────
#  Sampling from the learned prior ODE + decoder
# ──────────────────────────────────────────────────────────────────────────────
def sample_sequences_ode(
    model: DeterministicLatentODE,
    seq_len: int,
    n_samples: int,
    device: torch.device,
) -> Tensor:
    p_ode = model.p_ode
    p_observe = model.p_observe
    z0 = torch.randn(1, model.latent_size, device=device).repeat(n_samples, 1)

    with torch.no_grad():

        zs = solve_ode(p_ode, z0, 0.0, 1.0, n_steps=seq_len - 1)  # (L, B, latent)
        zs = zs.permute(1, 0, 2)  # (B, L, latent)

        tokens_fixed = torch.full(
            (n_samples, seq_len),
            fill_value=char2idx["?"],
            device=device,
            dtype=torch.long,
        )

        for t in range(seq_len):
            logits = p_observe.get_logits(zs, tokens_fixed)   # (B, L, V)
            step_logits = logits[:, t, :]               # (B, V)
            probs = torch.softmax(step_logits, dim=-1)
            tokens_fixed[:, t] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        z0 = torch.randn(n_samples, model.latent_size, device=device)


        zs = solve_ode(p_ode, z0, 0.0, 1.0, n_steps=seq_len - 1)  # (L, B, latent)
        zs = zs.permute(1, 0, 2)  # (B, L, latent)

        tokens_random = torch.full(
            (n_samples, seq_len),
            fill_value=char2idx["?"],
            device=device,
            dtype=torch.long,
        )

        for t in range(seq_len):
            logits = p_observe.get_logits(zs, tokens_random)   # (B, L, V)
            step_logits = logits[:, t, :]               # (B, V)
            probs = torch.softmax(step_logits, dim=-1)
            tokens_random[:, t] = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return tokens_fixed, tokens_random



# ──────────────────────────────────────────────────────────────────────────────
#  Training loop (ODE matching)
# ──────────────────────────────────────────────────────────────────────────────


def train_ode(
    model: DeterministicLatentODE,
    dataloader: DataLoader,
    n_iter: int,
    device: torch.device,
    loss_weights: tuple[float, float, float] = (1.0, 0.05, 1.0),
):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

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

        tokens = tokens.to(device)  # (B, L)

        # Warmup for loss_weights[1] (EP term) over the first 10000 steps
        if step < warmup_steps:
            interp = step / warmup_steps
            current_ep = initial_ep + interp * (final_ep - initial_ep)
        else:
            current_ep = final_ep
        weights = (loss_weights[0], current_ep, loss_weights[2])

        model.train()
        loss, loss_dict = model(tokens, loss_weights=weights)

        optim.zero_grad()
        loss.backward()
        # IMPROVED: Gradient clipping for stability (from numerical-stability + training-optimization agents)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        desc = (
            f"{loss.item():.4f} | "
            f"rec {loss_dict['recon']:.3f} "
            f"ep {loss_dict['latent_ep']:.3f} "
            f"ode {loss_dict['ode_reg']:.3f} "
            f"ep {current_ep:.3f}"
        )
        pbar.set_description(desc)

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                n_samples = 8
                seq_len = tokens.shape[1]
                samples_fixed, samples_random = sample_sequences_ode(
                    model,
                    seq_len=seq_len,
                    n_samples=n_samples,
                    device=device,
                )

                print("\nSamples that share a Z")
                for i in range(n_samples):
                    print(decode(samples_fixed[i].cpu()))
                
                print("\nSamples with a random Z")
                for i in range(n_samples):
                    print(decode(samples_random[i].cpu()))


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
#  RACCOON-IN-A-BUNGEECORD IMPLEMENTATION
#  Stochastic SDE-based continuous learning architecture
# ──────────────────────────────────────────────────────────────────────────────

class TimeAwareTransform(nn.Module):
    """
    Multi-scale time embedding using exponentially-spaced frequency bands.
    Provides richer temporal representation than simple sinusoidal encoding.
    """
    def __init__(self, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim

        # Frequency bands from 1 Hz to 1000 Hz (multi-scale temporal resolution)
        freqs = torch.exp(torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            time_dim // 2
        ))
        self.register_buffer('freqs', freqs)

    def embed_time(self, t: Tensor) -> Tensor:
        """
        Convert scalar time to rich multi-frequency features.

        Args:
            t: Time tensor (batch, 1)
        Returns:
            time_embed: (batch, time_dim) with sin/cos features
        """
        # t shape: (batch, 1)
        angles = t * self.freqs[None, :]  # (batch, time_dim//2)

        # Stack sin and cos for rotation-invariant representation
        time_embed = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ], dim=-1)  # (batch, time_dim)

        return time_embed


class RaccoonDynamics(nn.Module):
    """
    SDE dynamics with separate drift and diffusion networks.

    Implements: dz = drift(z,t)*dt + diffusion(z,t)*dW
    where dW is Wiener process (Brownian motion)
    """
    def __init__(self, latent_dim: int, hidden_dim: int,
                 sigma_min: float = 1e-4, sigma_max: float = 1.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Drift network (deterministic component)
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # FIXED: Diffusion network outputs log-variance for better numerical stability
        self.log_diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # IMPROVED: Better initialization for gradient flow (from numerical-stability agent)
        for module in [self.drift_net, self.log_diffusion_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)

    def forward(self, z: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute drift and diffusion at current state and time.

        Args:
            z: Current latent state (batch, latent_dim)
            t: Current time (batch, 1)
        Returns:
            drift: Deterministic velocity (batch, latent_dim)
            diffusion: Stochastic scale (batch, latent_dim)
        """
        zt = torch.cat([z, t], dim=-1)

        drift = self.drift_net(zt)

        # FIXED: Compute diffusion with proper bounds for numerical stability
        log_diffusion = self.log_diffusion_net(zt)
        # Clip log-diffusion to ensure numerical stability
        log_diffusion = torch.clamp(log_diffusion,
                                     math.log(self.sigma_min),
                                     math.log(self.sigma_max))
        diffusion = torch.exp(log_diffusion)

        return drift, diffusion


def solve_sde(
    dynamics: RaccoonDynamics,
    z0: Tensor,
    t_span: Tensor,
) -> Tensor:
    """
    Solve SDE using Euler-Maruyama method.

    Args:
        dynamics: RaccoonDynamics instance
        z0: Initial state (batch, latent_dim)
        t_span: Time points (num_steps,)
    Returns:
        path: Trajectory (batch, num_steps, latent_dim)
    """
    device = z0.device
    batch_size = z0.shape[0]

    path = [z0]
    z = z0

    for i in range(len(t_span) - 1):
        dt = t_span[i+1] - t_span[i]
        # FIXED: Correct tensor broadcasting from 0-d scalar
        t_curr = t_span[i:i+1].expand(batch_size, 1)

        # Get drift and diffusion
        drift, diffusion = dynamics(z, t_curr)

        # Euler-Maruyama step
        dW = torch.randn_like(z) * torch.sqrt(dt)
        z = z + drift * dt + diffusion * dW

        path.append(z)

    return torch.stack(path, dim=1)  # (batch, num_steps, latent)


class CouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows.
    Splits dimensions, uses one half to transform the other.

    IMPROVED: Parameterized time_dim and scale_range for better flexibility.
    """
    def __init__(self, dim: int, hidden: int, mask: Tensor,
                 time_dim: int = 32, scale_range: float = 3.0):
        super().__init__()
        self.register_buffer('mask', mask)
        self.time_dim = time_dim
        self.scale_range = scale_range

        # FIXED: Network with parameterized time_dim (not hardcoded 32)
        self.transform_net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim * 2)  # Output scale and shift
        )

    def forward(self, x: Tensor, time_feat: Tensor,
                reverse: bool = False) -> tuple[Tensor, Tensor]:
        """
        Apply coupling transformation.

        Args:
            x: Input (batch, dim)
            time_feat: Time features (batch, 32)
            reverse: If True, apply inverse transform
        Returns:
            y: Transformed output (batch, dim)
            log_det: Log determinant of Jacobian (batch,)
        """
        # Split using mask
        x_masked = x * self.mask

        # Compute transformation parameters conditioned on masked input and time
        h = torch.cat([x_masked, time_feat], dim=-1)
        params = self.transform_net(h)
        scale, shift = params.chunk(2, dim=-1)

        # FIXED: Configurable scale bounds for more expressiveness
        scale = torch.tanh(scale / self.scale_range) * self.scale_range

        # Apply transformation only to non-masked dimensions
        if not reverse:
            y = x_masked + (1 - self.mask) * (x * torch.exp(scale) + shift)
            log_det = (scale * (1 - self.mask)).sum(dim=-1)
        else:
            y = x_masked + (1 - self.mask) * ((x - shift) * torch.exp(-scale))
            log_det = (-scale * (1 - self.mask)).sum(dim=-1)

        return y, log_det


class RaccoonFlow(nn.Module):
    """
    Normalizing flow with multiple coupling layers.
    Provides invertible transformation for latent variables.

    IMPROVED: Parameterized time_dim throughout for consistency.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int = 4, time_dim: int = 32):
        super().__init__()
        self.time_embed = TimeAwareTransform(time_dim=time_dim)

        # Build coupling layers with alternating masks
        self.flows = nn.ModuleList()
        for i in range(num_layers):
            # Alternate which dimensions we transform
            mask = self._make_mask(latent_dim, i % 2)
            # FIXED: Pass time_dim to CouplingLayer
            self.flows.append(
                CouplingLayer(latent_dim, hidden_dim, mask, time_dim=time_dim)
            )

    def _make_mask(self, dim: int, parity: int) -> Tensor:
        """Create alternating mask for coupling layers."""
        mask = torch.zeros(dim)
        mask[parity::2] = 1  # Every other dimension
        return mask

    def forward(self, z: Tensor, t: Tensor,
                reverse: bool = False) -> tuple[Tensor, Tensor]:
        """
        Apply flow transformation.

        Args:
            z: Input latent (batch, latent_dim)
            t: Time (batch, 1)
            reverse: If True, apply inverse flow (generation)
        Returns:
            z_out: Transformed latent (batch, latent_dim)
            log_det_sum: Total log determinant (batch,)
        """
        time_features = self.time_embed.embed_time(t)
        log_det_sum = torch.zeros(z.shape[0], device=z.device)

        # Apply flows in order (or reverse)
        flows = reversed(self.flows) if reverse else self.flows

        for flow in flows:
            z, log_det = flow(z, time_features, reverse=reverse)
            log_det_sum += log_det

        return z, log_det_sum


class RaccoonMemory:
    """
    Experience replay buffer with priority sampling.
    Stores trajectories and quality scores for continuous learning.
    """
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.scores = []

    def add(self, trajectory, score: float):
        """
        Add experience to memory with quality score.
        If full, remove worst experience.
        """
        # Handle both tensor and dict inputs
        if isinstance(trajectory, dict):
            # Already in dict format (from fixed continuous_update)
            self.buffer.append(trajectory)
        else:
            # Old format - convert to CPU
            self.buffer.append(trajectory.detach().cpu())
        self.scores.append(score)

        # FIXED: Use numpy for efficiency instead of creating tensor each time
        if len(self.buffer) > self.max_size:
            import numpy as np
            scores_array = np.array(self.scores)
            worst_idx = int(scores_array.argmin())
            self.buffer.pop(worst_idx)
            self.scores.pop(worst_idx)

    def sample(self, n: int, device: torch.device) -> list:
        """
        Sample experiences with bias toward higher quality.

        Args:
            n: Number of samples
            device: Device to load tensors to
        Returns:
            List of sampled trajectories
        """
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from empty memory buffer")

        # FIXED: Determine if we need replacement
        available = len(self.buffer)
        replacement = available < n

        # FIXED: Robust score normalization using softmax trick
        import numpy as np
        scores_array = np.array(self.scores)

        # Subtract max for numerical stability (standard softmax trick)
        scores_shifted = scores_array - scores_array.max()

        # Use softmax with temperature
        temperature = 1.0
        exp_scores = np.exp(scores_shifted / temperature)
        probs = exp_scores / exp_scores.sum()

        # Ensure valid probabilities
        probs = np.maximum(probs, 1e-10)
        probs = probs / probs.sum()

        probs_tensor = torch.from_numpy(probs).float()
        indices = torch.multinomial(probs_tensor, n, replacement=replacement)

        # Return sampled items (handle dict format from fixed continuous_update)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def state_dict(self) -> dict:
        """ADDED: Export memory state for checkpointing."""
        return {
            'buffer': self.buffer,  # Already on CPU from add()
            'scores': self.scores,
            'max_size': self.max_size,
        }

    def load_state_dict(self, state: dict):
        """ADDED: Load memory from checkpoint."""
        self.buffer = state['buffer']
        self.scores = state['scores']
        self.max_size = state['max_size']


# ──────────────────────────────────────────────────────────────────────────────
#  LOG CLASSIFICATION TASK
# ──────────────────────────────────────────────────────────────────────────────

# Log categories
LOG_CATEGORIES = ["ERROR", "WARNING", "INFO", "DEBUG"]
NUM_LOG_CLASSES = len(LOG_CATEGORIES)

# Create vocabulary for log messages (reuse character vocab + add digits)
log_chars = chars + [str(i) for i in range(10)]  # Add 0-9
log_char2idx = {ch: i for i, ch in enumerate(log_chars)}
log_idx2char = {i: ch for ch, i in log_char2idx.items()}
log_vocab_size = len(log_chars)


def encode_log(s: str) -> Tensor:
    """Encode log message string to indices."""
    return torch.tensor([log_char2idx.get(c, 0) for c in s], dtype=torch.long)


def decode_log(t: Tensor) -> str:
    """Decode indices to log message string."""
    return "".join(log_idx2char.get(int(i), "_") for i in t)


class LogDataset(Dataset):
    """
    Synthetic log dataset with temporal patterns and concept drift.
    Generates realistic system logs across 4 categories.
    """
    def __init__(self, n_samples: int, seq_len: int = 50, drift_point: int = None):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.drift_point = drift_point  # Sample index where distribution shifts

        # Log message templates
        self.templates = {
            0: ["ERROR", "FAIL", "CRASH", "EXCEPTION", "NULL"],  # ERROR
            1: ["WARN", "DEPRECATED", "SLOW", "RETRY"],  # WARNING
            2: ["INFO", "START", "STOP", "READY"],  # INFO
            3: ["DEBUG", "TRACE", "VERBOSE"]  # DEBUG
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Introduce concept drift if specified
        if self.drift_point and idx >= self.drift_point:
            # After drift: ERROR becomes more common, DEBUG less common
            class_probs = [0.4, 0.3, 0.2, 0.1]
        else:
            # Before drift: balanced distribution
            class_probs = [0.25, 0.25, 0.25, 0.25]

        # Sample log category
        category = torch.multinomial(torch.tensor(class_probs), 1).item()

        # Generate log message from template
        template = self.templates[category][torch.randint(0, len(self.templates[category]), (1,)).item()]

        # Pad or truncate to seq_len
        if len(template) < self.seq_len:
            # Pad with underscores
            message = template + "_" * (self.seq_len - len(template))
        else:
            message = template[:self.seq_len]

        # Add some noise (10% character corruption)
        message_list = list(message)
        for i in range(len(message_list)):
            if torch.rand(1).item() < 0.1:
                message_list[i] = log_chars[torch.randint(0, len(log_chars), (1,)).item()]
        message = "".join(message_list)

        # Encode message
        tokens = encode_log(message)

        # Return tokens and category label
        return tokens, category


class RaccoonLogClassifier(nn.Module):
    """
    Raccoon-style continuous learning model for log classification.
    Combines SDE dynamics, normalizing flows, and experience replay.
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        memory_size: int = 5000,
        adaptation_rate: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate

        # Encoder: tokens → latent distribution
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.enc_mean = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # SDE dynamics (using new sigma_min/sigma_max parameters)
        self.dynamics = RaccoonDynamics(latent_dim, hidden_dim, sigma_min=1e-4, sigma_max=1.0)

        # IMPROVED: More flow layers for better expressiveness (from normalizing-flows agent)
        self.flow = RaccoonFlow(latent_dim, hidden_dim, num_layers=8)

        # Classifier head: latent → class logits
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # Latent regularization
        univariate_test = FastEppsPulley(t_max=5.0, n_points=17)
        self.latent_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=64,
            reduction="mean",
        )

        # Experience replay memory
        self.memory = RaccoonMemory(max_size=memory_size)

        # Learnable prior
        self.z0_mean = nn.Parameter(torch.zeros(latent_dim))
        self.z0_logvar = nn.Parameter(torch.zeros(latent_dim))

    def encode(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode tokens to latent distribution.

        Args:
            tokens: (batch, seq_len)
        Returns:
            mean: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Pool over sequence dimension
        x = self.encoder(tokens)  # (batch, seq_len, hidden)
        x = x.mean(dim=1)  # (batch, hidden)

        mean = self.enc_mean(x)
        logvar = self.enc_logvar(x)

        return mean, logvar

    def sample_latent(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def classify(self, z: Tensor) -> Tensor:
        """Classify from latent representation."""
        return self.classifier(z)

    def forward(
        self,
        tokens: Tensor,
        labels: Tensor,
        loss_weights: tuple[float, float, float] = (1.0, 0.1, 0.01),
    ):
        """
        Full forward pass with loss computation.

        Args:
            tokens: (batch, seq_len)
            labels: (batch,) class labels
            loss_weights: (class_loss, kl_loss, ep_loss)
        Returns:
            loss: Total loss
            stats: Dictionary of loss components
        """
        batch_size = tokens.shape[0]

        # FIXED: Guard against empty batches
        if batch_size == 0:
            return torch.tensor(0.0, device=tokens.device), {
                "class_loss": torch.tensor(0.0, device=tokens.device),
                "kl_loss": torch.tensor(0.0, device=tokens.device),
                "ep_loss": torch.tensor(0.0, device=tokens.device),
                "accuracy": torch.tensor(0.0, device=tokens.device),
            }

        # Encode
        mean, logvar = self.encode(tokens)
        z = self.sample_latent(mean, logvar)

        # Apply SDE dynamics (short trajectory)
        t_span = torch.linspace(0.0, 0.1, 3, device=z.device)  # Short time horizon
        z_traj = solve_sde(self.dynamics, z, t_span)  # (batch, 3, latent)
        z = z_traj[:, -1, :]  # Take final state

        # Apply normalizing flow
        t = torch.ones(batch_size, 1, device=z.device) * 0.5  # Mid-time point
        z_flow, log_det = self.flow(z, t, reverse=False)

        # Classify
        logits = self.classify(z_flow)

        # Classification loss
        class_loss = F.cross_entropy(logits, labels)

        # KL divergence to prior
        # FIXED: Corrected KL formula (was inverted with wrong sign)
        # KL(q||p) = 0.5 * mean[logvar_p - logvar + (var_q + (mu_p - mu_q)^2)/var_p - 1]
        var_q = torch.exp(logvar)
        var_p = torch.exp(self.z0_logvar)
        kl_loss = 0.5 * torch.mean(
            self.z0_logvar - logvar +
            (var_q + (mean - self.z0_mean).pow(2)) / (var_p + 1e-8) -
            1
        )
        # Clamp to ensure non-negative
        kl_loss = torch.clamp(kl_loss, min=0.0)

        # Epps-Pulley regularization
        z_for_test = z_flow.unsqueeze(0)  # (1, batch, latent)
        ep_loss = self.latent_test(z_for_test)

        # Total loss
        w_class, w_kl, w_ep = loss_weights
        loss = w_class * class_loss + w_kl * kl_loss + w_ep * ep_loss

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

        stats = {
            "class_loss": class_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "ep_loss": ep_loss.detach(),
            "accuracy": acc.detach(),
        }

        return loss, stats

    def continuous_update(self, tokens: Tensor, labels: Tensor):
        """
        Perform small online update with memory replay.

        Args:
            tokens: (batch, seq_len) new observations
            labels: (batch,) new labels
        """
        # Encode and score new data
        with torch.no_grad():
            mean, logvar = self.encode(tokens)
            z = self.sample_latent(mean, logvar)
            logits = self.classify(z)

            # IMPROVED: Use uncertainty (entropy) instead of confidence (from continual-learning agent)
            # Higher entropy = more uncertain = more valuable for learning
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            score = entropy.mean().item()

        # FIXED: Store as dict instead of concatenating tensors
        # Add to memory (store each sample separately with proper structure)
        for i in range(tokens.shape[0]):
            memory_item = {
                'tokens': tokens[i:i+1].detach().cpu(),  # Keep batch dim
                'label': labels[i:i+1].detach().cpu()
            }
            self.memory.add(memory_item, score)

        # Perform update if enough memory
        if len(self.memory) >= 32:
            # Sample from memory
            memory_batch = self.memory.sample(16, device=tokens.device)

            if len(memory_batch) > 0:
                # FIXED: Properly extract and concatenate from dict structure
                device = tokens.device  # Get device from input tokens
                memory_tokens_list = [m['tokens'].to(device) for m in memory_batch]
                memory_labels_list = [m['label'].to(device) for m in memory_batch]

                memory_tokens = torch.cat(memory_tokens_list, dim=0)  # (16, seq_len)
                memory_labels = torch.cat(memory_labels_list, dim=0).squeeze()  # (16,)

                # Combine with new data - shapes now match!
                all_tokens = torch.cat([tokens, memory_tokens], dim=0)
                all_labels = torch.cat([labels, memory_labels], dim=0)

                # FIXED: Proper gradient management using optimizer
                if not hasattr(self, '_adaptation_optimizer'):
                    self._adaptation_optimizer = torch.optim.SGD(
                        self.parameters(),
                        lr=self.adaptation_rate,
                        momentum=0.0
                    )

                # Forward pass and gradient update
                loss, _ = self.forward(all_tokens, all_labels)
                self._adaptation_optimizer.zero_grad()
                loss.backward()
                self._adaptation_optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOPS
# ──────────────────────────────────────────────────────────────────────────────

def train_raccoon_classifier(
    model: RaccoonLogClassifier,
    dataloader: DataLoader,
    n_iter: int,
    device: torch.device,
    loss_weights: tuple[float, float, float] = (1.0, 0.1, 0.01),
):
    """
    Phase 1: Initial supervised training.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    pbar = trange(n_iter, desc="Phase 1: Initial Training")
    data_iter = iter(dataloader)

    for step in pbar:
        try:
            tokens, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            tokens, labels = next(data_iter)

        tokens = tokens.to(device)
        labels = labels.to(device)

        # Forward pass
        model.train()
        loss, stats = model(tokens, labels, loss_weights=loss_weights)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # IMPROVED: Gradient clipping for stability (from numerical-stability + training-optimization agents)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{stats['accuracy']:.3f}",
            "class": f"{stats['class_loss']:.3f}",
            "kl": f"{stats['kl_loss']:.3f}",
        })

    print("\n✅ Phase 1 complete!")


def continuous_learning_phase(
    model: RaccoonLogClassifier,
    dataloader: DataLoader,
    n_samples: int,
    device: torch.device,
):
    """
    Phase 2: Continuous learning with online adaptation.
    """
    model.eval()  # Eval mode for base model, updates happen in continuous_update

    pbar = trange(n_samples, desc="Phase 2: Continuous Learning")
    data_iter = iter(dataloader)

    accuracies = []

    for step in pbar:
        try:
            tokens, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            tokens, labels = next(data_iter)

        # Take only one sample for online learning
        tokens = tokens[:1].to(device)
        labels = labels[:1].to(device)

        # Evaluate before update
        with torch.no_grad():
            mean, logvar = model.encode(tokens)
            z = model.sample_latent(mean, logvar)
            logits = model.classify(z)
            pred = logits.argmax(dim=1)
            correct = (pred == labels).float().item()
            accuracies.append(correct)

        # Continuous update
        model.continuous_update(tokens, labels)

        # Logging
        if step % 10 == 0:
            recent_acc = sum(accuracies[-100:]) / min(100, len(accuracies))
            pbar.set_postfix({
                "memory": len(model.memory),
                "acc": f"{recent_acc:.3f}",
            })

    print(f"\n✅ Phase 2 complete! Final memory size: {len(model.memory)}")
    print(f"📊 Final 100-sample accuracy: {sum(accuracies[-100:])/100:.3f}")


if __name__ == "__main__":
    # Force CPU for compatibility
    device = torch.device("cpu")
    print(f"🦝 Raccoon-in-a-Bungeecord Log Classifier")
    print(f"🖥️  Device: {device}")

    # Test original ODE model (optional)
    run_original_ode = False
    if run_original_ode:
        print("\n" + "="*70)
        print("RUNNING ORIGINAL ODE MODEL (Character Task)")
        print("="*70)

        ds = SyntheticTargetDataset(n_samples=100_000)
        x0 = ds[0]
        print("Example synthetic sequence:")
        print(decode(x0))

        batch_size = 128
        seq_len = 64
        latent_size = 64
        hidden_size = 128
        embed_size = 64

        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        model_ode = DeterministicLatentODE(
            vocab_size=vocab_size,
            latent_size=latent_size,
            hidden_size=hidden_size,
            embed_size=embed_size,
            num_slices=1024,
        ).to(device)

        model_ode.apply(weight_init)

        train_steps = 1000  # Reduced for testing
        train_ode(model_ode, dataloader, train_steps, device)

    # Run Raccoon log classifier
    print("\n" + "="*70)
    print("RUNNING RACCOON LOG CLASSIFIER (Continuous Learning)")
    print("="*70)

    # Create log dataset
    print("\n📝 Creating log dataset...")
    train_ds = LogDataset(n_samples=5000, seq_len=50, drift_point=None)
    test_ds = LogDataset(n_samples=1000, seq_len=50, drift_point=None)
    drift_ds = LogDataset(n_samples=1000, seq_len=50, drift_point=500)  # Concept drift

    # Show examples
    print("\n📋 Example logs:")
    for i in range(4):
        tokens, label = train_ds[i]
        message = decode_log(tokens)
        print(f"  [{LOG_CATEGORIES[label]}] {message}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    drift_loader = DataLoader(drift_ds, batch_size=1, shuffle=True)

    # Create Raccoon model
    print("\n🦝 Initializing Raccoon model...")
    model = RaccoonLogClassifier(
        vocab_size=log_vocab_size,
        num_classes=NUM_LOG_CLASSES,
        latent_dim=32,
        hidden_dim=64,
        embed_dim=32,
        memory_size=2000,
        adaptation_rate=1e-4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count:,}")

    # Phase 1: Initial training
    print("\n🏋️ Starting Phase 1: Initial Training...")
    train_raccoon_classifier(
        model=model,
        dataloader=train_loader,
        n_iter=1000,
        device=device,
        loss_weights=(1.0, 0.1, 0.01),
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            mean, logvar = model.encode(tokens)
            z = model.sample_latent(mean, logvar)
            logits = model.classify(z)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"✅ Test Accuracy: {correct/total:.3f} ({correct}/{total})")

    # Phase 2: Continuous learning
    print("\n🔄 Starting Phase 2: Continuous Learning...")
    continuous_learning_phase(
        model=model,
        dataloader=drift_loader,
        n_samples=1000,
        device=device,
    )

    # Final evaluation
    print("\n📊 Final evaluation after continuous learning...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            mean, logvar = model.encode(tokens)
            z = model.sample_latent(mean, logvar)
            logits = model.classify(z)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"✅ Final Test Accuracy: {correct/total:.3f} ({correct}/{total})")
    print(f"📦 Memory Buffer Size: {len(model.memory)}")

    print("\n" + "="*70)
    print("🎉 RACCOON-IN-A-BUNGEECORD TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✨ Successfully implemented:")
    print("  ✅ SDE Dynamics (drift + diffusion)")
    print("  ✅ Normalizing Flows (4 coupling layers)")
    print("  ✅ Experience Replay Memory")
    print("  ✅ Continuous Learning")
    print("  ✅ Multi-scale Time Embedding")
    print("  ✅ Epps-Pulley Regularization")
    print("\n🦝 The Raccoon has learned to bounce continuously!")

