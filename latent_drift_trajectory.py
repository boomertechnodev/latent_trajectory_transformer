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

        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt

        phi = (-0.5 * t.square()).exp()

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * self.phi)

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
    def __init__(self, latent_size: int, hidden_size: int):
        super().__init__()

        layers = []
        input_dim = latent_size + 1
        for i in range(11):
            linear = nn.Linear(input_dim, hidden_size)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            input_dim = hidden_size
        final_linear = nn.Linear(hidden_size, latent_size)
        nn.init.xavier_uniform_(final_linear.weight)
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
        latent_stat = self.latent_test(z_pred)
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


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # quick sanity check on dataset
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

    model = DeterministicLatentODE(
        vocab_size=vocab_size,
        latent_size=latent_size,
        hidden_size=hidden_size,
        embed_size=embed_size,
        num_slices=1024,
    ).to(device)

    model.apply(weight_init)

    train_steps = 100_000
    train_ode(model, dataloader, train_steps, device)

