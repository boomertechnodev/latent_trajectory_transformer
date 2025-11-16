# Variational Path Flows: A Friendly Implementation
# =================================================
# "Let's learn to predict butterfly paths without chasing the butterfly!"

"""
┌─────────────────────────────────────────────────────────────────┐
│                    WHAT THIS CODE DOES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Imagine you want to predict where a butterfly will be at      │
│  ANY point in time, without calculating its path step-by-step. │
│                                                                 │
│  Traditional way: Track every millisecond of flight             │
│  This way: Learn to jump directly to any time!                 │
│                                                                 │
│  It's like learning the "essence" of butterfly motion,         │
│  not just recording one specific flight.                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
from typing import Tuple, Any, Sequence
from abc import ABC, abstractmethod
import math
import torch
from torch import nn, Tensor
from torch import distributions as D
import matplotlib.pyplot as plt
from tqdm import trange

# ┌─────────────────────────────────────────────────────────────────┐
# │                        UTILITIES                                 │
# │           (Helper functions to make our life easier)            │
# └─────────────────────────────────────────────────────────────────┘

def set_seed(seed: int = 1234):
    """
    Makes randomness reproducible - like using the same dice every time!
    
    Why this matters: Science needs reproducibility. If you get cool results,
    others should be able to get the same results with the same "dice".
    """
    import random, os
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def visualise_data(xs: Tensor, filename: str = "figure.jpg"):
    """
    Creates those beautiful 3D butterfly plots you see!
    
    What it does: Takes trajectories (paths through time) and draws them
    in 3D space, creating those mesmerizing patterns.
    
    Think of it as: A cosmic spirograph in 3D!
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw each trajectory as a continuous line
    for xs_i in xs:
        ax.plot(xs_i[:, 0].detach().cpu(), 
                xs_i[:, 1].detach().cpu(), 
                xs_i[:, 2].detach().cpu())
    
    # Clean up the plot - remove clutter, add labels
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    
    plt.savefig(filename, format='jpg', dpi=300)
    plt.close(fig)

# ┌─────────────────────────────────────────────────────────────────┐
# │                    DATA GENERATION                               │
# │        (Creating synthetic butterfly paths to learn from)        │
# └─────────────────────────────────────────────────────────────────┘

class SDE(nn.Module, ABC):
    """
    Abstract base class for Stochastic Differential Equations
    
    Think of it as: A recipe for systems that evolve with both
    deterministic rules AND random jiggling!
    
    Like: A leaf falling (gravity = deterministic, wind = random)
    """
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        """The deterministic part - where the system 'wants' to go"""
        ...
    
    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        """The random part - how much randomness affects the system"""
        ...
    
    def forward(self, z: Tensor, t: Tensor, *args: Any) -> Tuple[Tensor, Tensor]:
        """Combine both parts to describe the full dynamics"""
        return self.drift(z, t, *args), self.vol(z, t, *args)

class StochasticLorenzSDE(SDE):
    """
    The famous Lorenz system - creates butterfly-shaped attractors!
    
    Historical note: Discovered by Edward Lorenz while modeling weather.
    It's the poster child for chaos theory - tiny changes lead to
    drastically different outcomes (the "butterfly effect").
    
    What it models: Originally atmospheric convection, but it shows up
    everywhere in nature where you have rotating fluids.
    """
    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.15, .15, .15)):
        super().__init__()
        # a = parameters that control the shape of the attractor
        # b = how much random noise to add to each dimension
        self.a = a
        self.b = b
    
    def drift(self, x: Tensor, t: Tensor, *args) -> Tensor:
        """
        The deterministic Lorenz equations - the 'rules' of the system
        
        These three equations create the beautiful butterfly pattern:
        - f1: Convection (hot air rising)
        - f2: Temperature gradient
        - f3: Nonlinear feedback
        """
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        a1, a2, a3 = self.a
        
        # The famous Lorenz equations
        f1 = a1 * (x2 - x1)              # Rate of convection
        f2 = a2 * x1 - x2 - x1 * x3      # Horizontal temperature variation
        f3 = x1 * x2 - a3 * x3           # Vertical temperature variation
        
        return torch.cat([f1, f2, f3], dim=1)
    
    def vol(self, x: Tensor, t: Tensor, *args) -> Tensor:
        """
        The random part - adds realistic 'jiggling' to the motion
        
        In real systems, there's always noise: thermal fluctuations,
        measurement errors, external disturbances, etc.
        """
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        b1, b2, b3 = self.b
        
        # Noise proportional to the state (multiplicative noise)
        # This is more realistic than constant noise
        return torch.cat([x1*b1, x2*b2, x3*b3], dim=1)

@torch.no_grad()
def solve_sde(
    sde: SDE,
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Numerically integrates an SDE forward in time (Euler-Maruyama method)
    
    This is the "old-school" way - stepping through time bit by bit.
    Our new method will learn to avoid this!
    
    How it works:
    1. Take current position
    2. Add deterministic change (drift × small time)
    3. Add random change (volatility × random × √time)
    4. Repeat many times
    
    Why √time? That's how randomness accumulates in continuous time!
    """
    # Create time grid from start (ts) to finish (tf)
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)[:-1]
    dt = (tf - ts) / n_steps          # Time step size
    dt_2 = abs(dt) ** 0.5             # Square root for the randomness scaling
    
    path = [z]  # Store the trajectory
    
    for t in tt:
        f, g = sde(z, t)               # Get drift and volatility
        w = torch.randn_like(z)        # Random noise
        
        # Euler-Maruyama update: new = old + drift×dt + noise×√dt
        z = z + f * dt + g * w * dt_2
        path.append(z)
    
    return torch.stack(path)  # (n_steps+1, batch_size, dimensions)

def gen_data(
    batch_size: int,
    ts: float,
    tf: float,
    n_steps: int,
    noise_std: float,
    n_inner_steps: int = 100
) -> Tuple[Tensor, Tensor]:
    """
    Generates training data: noisy observations of Lorenz trajectories
    
    The process:
    1. Simulate high-resolution Lorenz paths (n_inner_steps between observations)
    2. Subsample to get the observation times
    3. Normalize the data (zero mean, unit variance)
    4. Add observation noise
    
    This mimics real-world scenarios where:
    - The true dynamics are continuous
    - We only observe at discrete times
    - Observations are noisy
    """
    sde = StochasticLorenzSDE()
    
    # Random starting positions
    z0 = torch.randn(batch_size, 3)
    
    # Simulate at high resolution (for accuracy)
    zs = solve_sde(sde, z0, ts, tf, n_steps=n_steps * n_inner_steps)
    
    # Subsample to observation times and reshape
    # (num_times, batch, dims) → (batch, num_times, dims)
    zs = zs[::n_inner_steps].permute(1, 0, 2)
    
    # Normalize the data (important for neural network training!)
    mean, std = zs.mean(dim=(0, 1)), zs.std(dim=(0, 1))
    xs = (zs - mean) / std + noise_std * torch.randn_like(zs)
    
    # Create time stamps for each observation
    ts_grid = torch.linspace(ts, tf, n_steps + 1, device=xs.device)
    ts_grid = ts_grid[None, :, None].repeat(batch_size, 1, 1)
    
    return xs, ts_grid  # (batch, time, dims), (batch, time, 1)

# ┌─────────────────────────────────────────────────────────────────┐
# │                 POSTERIOR INFERENCE NETWORK                      │
# │        (The detective: inferring hidden states from data)       │
# └─────────────────────────────────────────────────────────────────┘

"""
┌────────────────────────────────────────────────────────────────┐
│              WHAT IS A POSTERIOR?                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Imagine you see footprints in the sand (observations).       │
│  The posterior asks: "What path did the person take?"         │
│                                                                │
│  It's working backwards from effects to causes!               │
│                                                                │
│  In math notation: q(z|x) = "probability of hidden state z    │
│                              given observation x"              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""

class PosteriorEncoder(nn.Module):
    """
    Encodes the entire observed sequence into a useful representation
    
    Uses a GRU (Gated Recurrent Unit) - think of it as a neural network
    with memory. It reads through the sequence and builds up an understanding
    of the patterns.
    
    Like reading a book: Each word updates your understanding of the story.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # GRU: A type of RNN that's good at remembering important things
        # and forgetting irrelevant details
        self.gru = nn.GRU(input_size=input_size, 
                         hidden_size=hidden_size, 
                         batch_first=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Process the sequence and return both:
        - The final hidden state (summary of everything)
        - All intermediate states (detailed history)
        """
        out, h = self.gru(x)
        # Concatenate final state with all states for maximum flexibility
        return torch.cat([h[0, :, None], out], dim=1)  # (batch, 1+time, hidden)

class PosteriorAffine(nn.Module):
    """
    Takes the encoded sequence and produces a probability distribution
    for the hidden state at ANY requested time.
    
    The clever part: It uses a time-adaptive attention mechanism!
    
    Think of it as: "Given everything I know about this trajectory,
    what's probably happening at time t?"
    """
    def __init__(self, latent_size: int, hidden_size: int, init_logstd: float = -0.5):
        super().__init__()
        self.latent_size = latent_size
        
        # Neural network that takes context + time → mean and std of Gaussian
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size), nn.SiLU(),  # SiLU = smooth ReLU
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 2 * latent_size),  # Output mean and log(std)
        )
        
        # Softmax for creating smooth attention weights
        self.sm = nn.Softmax(dim=-1)
        
        # Initialize the network sensibly
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Good for deep networks
                nn.init.zeros_(m.bias)
        
        # Initialize with small-ish standard deviation
        last = self.net[-1]
        with torch.no_grad():
            last.bias[self.latent_size:] = init_logstd
    
    def get_coeffs(self, ctx: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        The time-adaptive magic happens here!
        
        How it works:
        1. Look at all encoded time steps
        2. Pay more attention to times near our query time t
        3. Blend the information smoothly
        4. Output mean and standard deviation for the Gaussian
        """
        l = ctx.shape[1] - 1  # Number of time steps
        h, out = ctx[:, 0], ctx[:, 1:]  # Split global and per-step features
        
        # Create attention weights based on temporal distance
        ts = torch.linspace(0, 1, l, device=ctx.device, dtype=ctx.dtype)[None, :]
        # Gaussian-like attention: exp(-(distance)²)
        c = self.sm(-(l * (ts - t)) ** 2)
        
        # Weighted combination of contexts
        out = (out * c[:, :, None]).sum(dim=1)
        
        # Combine global context + weighted local context + query time
        ctx_t = torch.cat([h + out, t], dim=1)
        
        # Get mean and log(std) from network
        m, log_s = self.net(ctx_t).chunk(2, dim=1)
        s = torch.exp(log_s)  # Convert log(std) to std
        
        return m, s
    
    def forward(self, ctx: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Just a wrapper for get_coeffs"""
        return self.get_coeffs(ctx, t)

# ┌─────────────────────────────────────────────────────────────────┐
# │                    FLOW UTILITIES                                │
# │         (Mathematical tools for probability and flows)           │
# └─────────────────────────────────────────────────────────────────┘

def diag_gauss_logprob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    """
    Log probability of x under a diagonal Gaussian distribution
    
    What's a diagonal Gaussian? Each dimension is independent with
    its own mean and variance. Like multiple 1D bell curves.
    
    Why log? Probabilities multiply, but logs add - much more
    numerically stable for computers!
    """
    var = var.clamp_min(1e-10)  # Avoid division by zero
    log2pi = math.log(2.0 * math.pi)
    
    # The famous Gaussian formula, in log form
    return -0.5 * (torch.log(var) + log2pi + (x - mean) ** 2 / var).sum(dim=-1)

class TimeEmbed(nn.Module):
    """
    Converts scalar time values into rich feature vectors
    
    Why? Neural networks work better with multi-dimensional inputs.
    This is like converting a single number into a unique "fingerprint".
    
    Uses sinusoidal embeddings (like in Transformers) - different
    frequencies capture different time scales.
    """
    def __init__(self, dim: int = 64, max_freq: float = 16.0):
        super().__init__()
        # Create frequencies: [1, 2, 4, 8, 16, ...] up to max_freq
        freqs = torch.exp(torch.linspace(0., math.log(max_freq), dim // 2))
        self.register_buffer("freqs", freqs)
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Convert time to features using sin and cos at multiple frequencies
        
        Like a musical chord - multiple frequencies combine to create
        a unique "sound" for each time.
        """
        ang = t.to(self.freqs.dtype) * self.freqs.view(*([1]*(t.dim()-1)), -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

class MLP(nn.Module):
    """
    A simple but effective neural network architecture
    
    MLP = Multi-Layer Perceptron (fancy name for "regular neural network")
    
    Structure: input → hidden → activation → hidden → activation → output
    The activations (SiLU) add non-linearity, letting it learn complex patterns.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        
        # Xavier initialization: Keeps signals from exploding or vanishing
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# ┌─────────────────────────────────────────────────────────────────┐
# │                TIME-CONDITIONAL LATENT FLOW                      │
# │    (The main innovation: learning time-dependent transforms)     │
# └─────────────────────────────────────────────────────────────────┘

"""
┌────────────────────────────────────────────────────────────────┐
│                    NORMALIZING FLOWS 101                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Imagine you have a lump of clay (simple distribution).       │
│  You want to sculpt it into a complex shape (data).          │
│                                                                │
│  Rules:                                                        │
│  1. You can only use reversible deformations                 │
│  2. You must track how much you stretch/compress each part   │
│                                                                │
│  That's a normalizing flow!                                   │
│                                                                │
│  Here, we make the deformation depend on TIME, so the        │
│  sculpture can smoothly morph as time passes.                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""

class AffineCoupling(nn.Module):
    """
    One layer of the RealNVP flow - a clever reversible transformation
    
    The trick: Split variables in half. Use one half to transform the other.
    Since half stays unchanged, you can always reverse the operation!
    
    Think of it like a combination lock: You can always undo it if you
    know which dials you turned and by how much.
    """
    def __init__(self, D: int, hidden: int, tdim: int, mask: Tensor, clamp: float = 1.5):
        super().__init__()
        # Mask determines which variables stay fixed (1) vs transform (0)
        self.register_buffer("mask", mask.view(1, -1).float())
        
        # Time embedding network
        self.temb = TimeEmbed(tdim)
        
        # Network that computes the transformation parameters
        self.net = MLP(in_dim=D + tdim, hidden=hidden, out_dim=2 * D)
        
        # Clamp prevents extreme transformations (numerical stability)
        self.clamp = clamp
        
        # Initialize as identity transform (no change at start)
        last = None
        for m in self.net.net:
            if isinstance(m, nn.Linear):
                last = m
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    
    def forward(self, x: Tensor, t: Tensor, inverse: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Apply the coupling transformation (or its inverse)
        
        Forward:  y = x * exp(s(x,t)) + b(x,t)  [only on masked dimensions]
        Inverse:  x = (y - b(x,t)) * exp(-s(x,t))
        
        The exp(s) ensures positivity and the log-determinant is just sum(s)!
        """
        m = self.mask
        xa = x * m  # Variables that stay fixed
        
        # Compute transformation parameters from fixed variables + time
        h = torch.cat([xa, self.temb(t)], dim=-1)
        s, b = self.net(h).chunk(2, dim=-1)
        
        # Bound the scale to prevent numerical issues
        s = torch.tanh(s) * self.clamp
        
        # Only transform the non-fixed variables
        comp = (1. - m)
        s = s * comp
        b = b * comp
        
        if not inverse:
            # Forward transform
            y = xa + comp * (x * torch.exp(s) + b)
            logdet = s.sum(dim=-1)  # Log of Jacobian determinant
        else:
            # Inverse transform
            y = xa + comp * ((x - b) * torch.exp(-s))
            logdet = (-s).sum(dim=-1)
        
        return y, logdet

class TimeCondRealNVP(nn.Module):
    """
    Stack multiple coupling layers to create a powerful flow
    
    Like a master sculptor using multiple tools - each layer adds
    more flexibility to the transformation.
    
    The alternating masks ensure all dimensions get transformed.
    """
    def __init__(self, D: int, hidden: int = 160, n_layers: int = 6, tdim: int = 64):
        super().__init__()
        
        # Create alternating masks: [1,0,1,0,...] and [0,1,0,1,...]
        base_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(D)])
        masks = [(base_mask if k % 2 == 0 else 1 - base_mask) for k in range(n_layers)]
        
        # Stack coupling layers with alternating masks
        self.layers = nn.ModuleList([
            AffineCoupling(D, hidden, tdim, m) for m in masks
        ])
    
    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        """Transform from simple space (y) to complex space (z)"""
        x = y
        for layer in self.layers:
            x, _ = layer(x, t, inverse=False)
        return x
    
    def inverse(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform from complex space (z) to simple space (y) + log det"""
        y = x
        logdet = x.new_zeros(x.size(0))
        
        # Apply layers in reverse order
        for layer in reversed(self.layers):
            y, ld = layer(y, t, inverse=True)
            logdet = logdet + ld
        
        return y, logdet

# ┌─────────────────────────────────────────────────────────────────┐
# │              CONDITIONAL OBSERVATION FLOW                        │
# │    (How hidden states generate what we actually observe)        │
# └─────────────────────────────────────────────────────────────────┘

"""
┌────────────────────────────────────────────────────────────────┐
│                 OBSERVATION MODEL                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Hidden state z (wind patterns) → Observations x (leaf motion)│
│                                                                │
│  This flow learns: p(x|z) = "What we see given hidden state" │
│                                                                │
│  It's conditional because the transformation depends on z!    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""

class CondAffineCoupling(nn.Module):
    """
    Like AffineCoupling, but the transformation depends on external condition z
    
    Real-world analogy: How you walk (x) depends on the terrain (z).
    The same walking motion looks different on sand vs concrete.
    """
    def __init__(self, D: int, hidden: int, cond_dim: int, mask: Tensor, clamp: float = 1.5):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1).float())
        
        # Network takes masked x AND condition z
        self.net = MLP(in_dim=D + cond_dim, hidden=hidden, out_dim=2 * D)
        self.clamp = clamp
        
        # Identity initialization
        last = None
        for m in self.net.net:
            if isinstance(m, nn.Linear):
                last = m
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    
    def forward(self, x: Tensor, z: Tensor, inverse: bool = False) -> Tuple[Tensor, Tensor]:
        """Same as AffineCoupling but conditioned on z"""
        m = self.mask
        xa = x * m
        
        # Key difference: concatenate with condition z
        h = torch.cat([xa, z], dim=-1)
        s, b = self.net(h).chunk(2, dim=-1)
        s = torch.tanh(s) * self.clamp
        
        comp = (1. - m)
        s = s * comp
        b = b * comp
        
        if not inverse:
            y = xa + comp * (x * torch.exp(s) + b)
            logdet = s.sum(dim=-1)
        else:
            y = xa + comp * ((x - b) * torch.exp(-s))
            logdet = (-s).sum(dim=-1)
        
        return y, logdet

class CondRealNVP(nn.Module):
    """Stack of conditional coupling layers"""
    def __init__(self, D: int, cond_dim: int, hidden: int = 128, n_layers: int = 6):
        super().__init__()
        
        # Alternating masks as before
        base_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(D)])
        masks = [(base_mask if k % 2 == 0 else 1 - base_mask) for k in range(n_layers)]
        
        self.layers = nn.ModuleList([
            CondAffineCoupling(D, hidden, cond_dim, m) for m in masks
        ])
    
    def forward(self, u: Tensor, z: Tensor) -> Tensor:
        """Generate observation x from noise u, conditioned on state z"""
        x = u
        for layer in self.layers:
            x, _ = layer(x, z, inverse=False)
        return x
    
    def inverse(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract noise u from observation x, conditioned on state z"""
        u = x
        logdet = x.new_zeros(x.size(0))
        for layer in reversed(self.layers):
            u, ld = layer(u, z, inverse=True)
            logdet = logdet + ld
        return u, logdet

class ObservationFlow(nn.Module):
    """
    The complete observation model: p(x|z)
    
    Key insight: We model x = g(u; z) where u ~ N(0,I)
    This gives us: p(x|z) = p(u) × |det(dg/du)|
    
    It's saying: "Observations are just Gaussian noise pushed through
    a complex (but reversible) transformation that depends on the hidden state."
    """
    def __init__(self, data_size: int, cond_dim: int, hidden: int = 128, n_layers: int = 6):
        super().__init__()
        self.flow = CondRealNVP(D=data_size, cond_dim=cond_dim, 
                               hidden=hidden, n_layers=n_layers)
    
    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Compute log p(x|z) using change of variables formula
        
        1. Transform x back to noise u (and get Jacobian determinant)
        2. Evaluate probability of u under standard Gaussian
        3. Add log determinant to account for the transformation
        """
        u, logdet = self.flow.inverse(x, z)
        
        # Standard Gaussian log probability
        mean0 = torch.zeros_like(u)
        var1 = torch.ones_like(u)
        log_pu = diag_gauss_logprob(u, mean0, var1)
        
        return log_pu + logdet
    
    @torch.no_grad()
    def sample(self, z: Tensor, n: int = None) -> Tensor:
        """
        Generate observations given hidden states
        
        1. Sample Gaussian noise u ~ N(0,I)
        2. Transform through flow: x = g(u; z)
        """
        B, Dz = z.shape
        Dx = self.flow.layers[0].mask.numel()
        
        if n is None:
            # One sample per z
            u = torch.randn(B, Dx, device=z.device, dtype=z.dtype)
            return self.flow.forward(u, z)
        else:
            # Multiple samples per z
            zt = z.repeat_interleave(n, dim=0)
            u = torch.randn(B * n, Dx, device=z.device, dtype=z.dtype)
            return self.flow.forward(u, zt)

# ┌─────────────────────────────────────────────────────────────────┐
# │             ORNSTEIN-UHLENBECK BASE DYNAMICS                     │
# │        (The simple dynamics that we transform into complex ones) │
# └─────────────────────────────────────────────────────────────────┘

"""
┌────────────────────────────────────────────────────────────────┐
│              ORNSTEIN-UHLENBECK (OU) PROCESS                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Think of a particle in honey attached to a spring:           │
│                                                                │
│  • Spring pulls toward center (mean reversion)                │
│  • Random thermal kicks (Brownian motion)                     │
│  • Honey provides damping (decay rate)                        │
│                                                                │
│  It's the continuous-time version of:                         │
│  tomorrow = today + pull_to_center + random_shock            │
│                                                                │
│  Why use it? It has nice mathematical properties and         │
│  realistic behavior (doesn't explode to infinity).           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""

class DiagOUSDE(nn.Module):
    """
    Diagonal Ornstein-Uhlenbeck SDE - each dimension evolves independently
    
    Parameters:
    - mu: long-term mean (where the spring pulls to)
    - kappa: mean reversion rate (spring stiffness)
    - sigma: volatility (strength of random kicks)
    """
    def __init__(self, D: int, init_mu: float = 0.0, 
                 init_logk: float = -0.7, init_logs: float = -1.0):
        super().__init__()
        
        # Learnable parameters (in log space for positivity)
        self.mu = nn.Parameter(torch.full((D,), init_mu))
        self.log_kappa = nn.Parameter(torch.full((D,), init_logk))
        self.log_sigma = nn.Parameter(torch.full((D,), init_logs))
    
    def _params(self):
        """Convert log parameters to actual values (ensures positivity)"""
        kappa = torch.nn.functional.softplus(self.log_kappa) + 1e-6
        sigma = torch.nn.functional.softplus(self.log_sigma) + 1e-6
        mu = self.mu
        return mu, kappa, sigma
    
    @torch.no_grad()
    def sample_path_cond(self, ts_grid: Tensor, y0: Tensor) -> Tensor:
        """
        Sample a path from the OU process given initial condition
        
        Uses the exact solution (not numerical integration!):
        For OU process, the transition distribution is known in closed form.
        
        This is much more accurate than Euler-Maruyama for this special case.
        """
        device = ts_grid.device
        n, D = y0.shape
        mu, kappa, sigma = self._params()
        mu = mu.to(device)
        kappa = kappa.to(device)
        sigma = sigma.to(device)
        
        T = ts_grid.size(0)
        y = torch.zeros(n, T, D, device=device)
        y[:, 0, :] = y0
        
        for k in range(T - 1):
            dt = (ts_grid[k+1] - ts_grid[k]).clamp(min=1e-6)
            
            # Exact transition formula for OU process
            Ad = torch.exp(-kappa * dt)  # Decay factor
            mean = mu + Ad * (y[:, k, :] - mu)  # Mean at next time
            
            # Variance of the transition (derived from OU theory)
            q = (sigma**2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
            
            # Sample next state
            y[:, k+1, :] = mean + torch.randn_like(mean) * q.sqrt()
        
        return y
    
    def path_log_prob_cond(self, y: Tensor, ts_batch: Tensor, y0_given: Tensor) -> Tensor:
        """
        Compute log probability of a path under the OU process
        
        Since we know the transition distributions exactly, we can compute
        the exact path likelihood as a product of transition probabilities.
        
        log p(path) = sum of log p(y_{t+1} | y_t)
        """
        B, T, D = y.shape
        
        # Handle time dimension
        if ts_batch.dim() == 2:
            ts_batch = ts_batch[None, :, :].expand(B, -1, -1)
        
        # Get parameters and expand dimensions
        mu, kappa, sigma = self._params()
        mu = mu[None, None, :].to(y.device)
        kappa = kappa[None, None, :].to(y.device)
        sigma = sigma[None, None, :].to(y.device)
        
        # Time intervals
        t0, t1 = ts_batch[:, :-1, :], ts_batch[:, 1:, :]
        dt = (t1 - t0).clamp(min=1e-6)
        
        # Transition parameters
        Ad = torch.exp(-kappa * dt)
        
        # Previous states (including given initial condition)
        prev = torch.cat([y0_given[:, None, :], y[:, 1:T-1, :]], dim=1)
        
        # Mean and variance of transitions
        mean = mu + Ad * (prev - mu)
        q = (sigma**2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        
        # Next states
        y_next = y[:, 1:, :]
        
        # Log probability of all transitions
        lp_trans = diag_gauss_logprob(y_next, mean, q).sum(dim=1)
        
        return lp_trans

class NF_SDE_Model(nn.Module):
    """
    The complete prior model: OU process + time-conditional flow
    
    This combines:
    1. Simple base dynamics (OU process in y-space)
    2. Complex transformation (normalizing flow z = f_t(y))
    
    Result: Complex dynamics in z-space with tractable likelihoods!
    """
    def __init__(self, D: int, hidden: int = 160, n_layers: int = 6, 
                 tdim: int = 64, t_min: float = 0.0, t_max: float = 1.0):
        super().__init__()
        
        # Time-conditional flow: y → z
        self.flow = TimeCondRealNVP(D, hidden=hidden, n_layers=n_layers, tdim=tdim)
        
        # Base dynamics: OU process in y-space
        self.ou = DiagOUSDE(D)
        
        # Time normalization bounds
        self.register_buffer("t_min", torch.tensor(float(t_min)))
        self.register_buffer("t_max", torch.tensor(float(t_max)))
    
    def _cond_t(self, t: Tensor) -> Tensor:
        """Normalize time to [0, 1] for the flow"""
        denom = (self.t_max - self.t_min).clamp(min=1e-8)
        tau = (t - self.t_min) / denom
        return tau.clamp(0., 1.)
    
    def log_prob_paths_cond(self, z_path: Tensor, ts_batch: Tensor, z0: Tensor) -> Tensor:
        """
        Compute log probability of a z-space path
        
        Strategy:
        1. Transform z-path back to y-path (and track Jacobian)
        2. Compute OU log probability in y-space
        3. Add Jacobian correction for change of variables
        """
        B, T, D = z_path.shape
        
        # Flatten for batch processing
        if ts_batch.dim() == 2:
            ts_batch = ts_batch[None, :, :].expand(B, -1, -1)
        
        zf = z_path.reshape(B * T, D)
        tf = ts_batch.reshape(B * T, 1)
        tf_cond = self._cond_t(tf)
        
        # Transform z → y and get Jacobians
        yf, logdetf = self.flow.inverse(zf, tf_cond)
        y = yf.reshape(B, T, D)
        logdet_seq = logdetf.reshape(B, T)[:, 1:].sum(dim=1)
        
        # Transform initial condition
        t0 = ts_batch[:, 0, :]
        t0_cond = self._cond_t(t0)
        y0, _ = self.flow.inverse(z0, t0_cond)
        
        # OU log probability + Jacobian correction
        lp_trans = self.ou.path_log_prob_cond(y, ts_batch, y0)
        
        return lp_trans + logdet_seq
    
    @torch.no_grad()
    def sample_paths_cond(self, ts_grid: Tensor, z0: Tensor) -> Tensor:
        """
        Sample z-space paths given initial condition
        
        Strategy:
        1. Transform z0 → y0
        2. Sample OU path in y-space
        3. Transform y-path → z-path
        """
        n, D = z0.shape
        
        # Transform initial condition
        t0 = ts_grid[0:1, :].expand(n, -1)
        t0_cond = self._cond_t(t0)
        y0, _ = self.flow.inverse(z0, t0_cond)
        
        # Sample in y-space
        y = self.ou.sample_path_cond(ts_grid, y0)
        
        # Transform to z-space
        yf = y.reshape(-1, D)
        tf = ts_grid[None, :, :].expand(n, -1, -1).reshape(-1, 1)
        tf_cond = self._cond_t(tf)
        z = self.flow.forward(yf, tf_cond).reshape(n, ts_grid.size(0), D)
        
        return z

# ┌─────────────────────────────────────────────────────────────────┐
# │                    INITIAL DISTRIBUTION                          │
# │              (Where our trajectories begin)                      │
# └─────────────────────────────────────────────────────────────────┘

class PriorInitDistribution(nn.Module):
    """
    Learned prior over initial latent states z0
    
    Simple diagonal Gaussian with learnable mean and standard deviation.
    This learns "typical starting points" for the dynamics.
    """
    def __init__(self, latent_size: int, init_log_s: float = -0.2):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1, latent_size))
        self.log_s = nn.Parameter(torch.full((1, latent_size), init_log_s))
    
    def forward(self) -> D.Distribution:
        """Return a PyTorch distribution object"""
        m = self.m
        s = torch.exp(self.log_s)
        return D.Independent(D.Normal(m, s), 1)

# ┌─────────────────────────────────────────────────────────────────┐
# │                    THE COMPLETE MODEL                            │
# │            (Putting all the pieces together)                     │
# └─────────────────────────────────────────────────────────────────┘

class FlowPriorMatchingCond(nn.Module):
    """
    The full variational model combining everything
    
    Components:
    1. Prior: How latent trajectories evolve (prior_flow + z0_prior)
    2. Observation model: How latents generate observations (p_obs_flow)
    3. Inference: How to infer latents from observations (q_enc + q_affine)
    
    Training objective (ELBO):
    maximize E_q[ log p(x|z) + log p(z) - log q(z|x) ]
    
    This balances:
    - Reconstruction: observations should match data
    - Prior matching: latents should follow learned dynamics
    - Inference accuracy: posterior should be well-calibrated
    """
    def __init__(
        self,
        prior_flow: NF_SDE_Model,
        p_observe_flow: ObservationFlow,
        q_enc: PosteriorEncoder,
        q_affine: PosteriorAffine,
        z0_prior: PriorInitDistribution,
    ):
        super().__init__()
        self.prior = prior_flow
        self.p_obs_flow = p_observe_flow
        self.q_enc = q_enc
        self.q_affine = q_affine
        self.z0_prior = z0_prior
    
    def _posterior_coeffs_all(self, ctx: Tensor, ts: Tensor) -> Tuple[Tensor, Tensor]:
        """Get posterior parameters for all time points at once"""
        B, T, _ = ts.shape
        
        # Flatten for batch processing
        t_flat = ts.reshape(B*T, 1)
        ctx_rep = ctx.repeat_interleave(T, dim=0)
        
        # Get means and stds
        m_flat, s_flat = self.q_affine(ctx_rep, t_flat)
        
        # Reshape back
        Dz = m_flat.size(-1)
        m = m_flat.view(B, T, Dz)
        s = s_flat.view(B, T, Dz)
        
        return m, s
    
    def forward(self, xs: Tensor, ts: Tensor) -> Tuple[Tensor, dict]:
        """
        Compute the training loss (negative ELBO)
        
        Steps:
        1. Encode observations to get posterior parameters
        2. Sample latents from posterior
        3. Compute observation likelihood p(x|z)
        4. Compute prior likelihood p(z)
        5. Compute posterior entropy (via -log q(z|x))
        6. Combine into ELBO
        """
        B, T, Dx = xs.shape
        
        # 1. Encode observations
        ctx = self.q_enc(xs)  # (B, 1+T, H)
        
        # 2. Get posterior and sample latents
        m, s = self._posterior_coeffs_all(ctx, ts)
        eps = torch.randn_like(m)
        z = m + s * eps  # Reparameterization trick
        
        Dz = z.size(-1)
        
        # 3. Observation likelihood log p(x|z)
        x_flat = xs.reshape(B*T, Dx)
        z_flat = z.reshape(B*T, Dz)
        log_px_flat = self.p_obs_flow.log_prob(x_flat, z_flat)
        log_px = log_px_flat.view(B, T).sum(dim=1)  # Sum over time
        
        # 4. Prior likelihood: log p(z0) + log p(z_{1:T}|z0)
        z0 = z[:, 0, :]
        log_pz0 = self.z0_prior().log_prob(z0)
        log_pz_cond = self.prior.log_prob_paths_cond(z, ts, z0)
        
        # 5. Posterior entropy: -E_q[log q]
        log_q = diag_gauss_logprob(
            z.reshape(B*T, Dz), 
            m.reshape(B*T, Dz), 
            (s.reshape(B*T, Dz) ** 2)
        ).view(B, T).sum(dim=1)
        
        # 6. ELBO = E_q[log p(x,z) - log q(z|x)]
        elbo = log_px + log_pz0 + log_pz_cond - log_q
        
        # Return negative ELBO as loss (we minimize loss)
        loss = -elbo.mean()
        
        return loss

# ┌─────────────────────────────────────────────────────────────────┐
# │                      MAIN TRAINING LOOP                          │
# │                 (Where the magic happens!)                       │
# └─────────────────────────────────────────────────────────────────┘

"""
┌────────────────────────────────────────────────────────────────┐
│                    TRAINING SUMMARY                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Generate synthetic data (noisy Lorenz trajectories)       │
│  2. Build all model components                                │
│  3. Train using gradient descent on the ELBO                  │
│  4. Periodically generate samples to check progress           │
│                                                                │
│  The model learns to:                                          │
│  • Compress observations into meaningful latents              │
│  • Model smooth dynamics in latent space                      │
│  • Generate new trajectories that look like the data          │
│                                                                │
│  All without ever simulating the dynamics step-by-step!       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ═══════════════════════════════════════════════════════════════
    #                     Generate Training Data
    # ═══════════════════════════════════════════════════════════════
    
    batch_size = 2 ** 10  # 1024 trajectories
    ts0, tf = 0., 3.      # Time span: 0 to 3
    n_steps = 120         # Number of observation points
    noise_std = .01       # Observation noise level
    
    # Generate noisy observations of Lorenz trajectories
    xs, ts = gen_data(batch_size, ts0, tf, n_steps, noise_std)
    xs, ts = xs.to(device), ts.to(device)
    
    # Visualize some examples
    visualise_data(xs[:6], 'data.jpg')
    print("Generated data visualization saved to 'data.jpg'")
    
    # ═══════════════════════════════════════════════════════════════
    #                     Build Model Components
    # ═══════════════════════════════════════════════════════════════
    
    # Dimensions
    data_size = xs.size(-1)     # 3 (x, y, z coordinates)
    latent_size = 4             # Hidden state dimension
    hidden_size = 128           # Neural network width
    
    # Prior components
    prior_flow = NF_SDE_Model(
        D=latent_size, 
        hidden=hidden_size, 
        n_layers=6, 
        tdim=64,
        t_min=ts0, 
        t_max=tf
    ).to(device)
    
    z0_prior = PriorInitDistribution(
        latent_size, 
        init_log_s=-0.2
    ).to(device)
    
    # Observation model
    p_obs_flow = ObservationFlow(
        data_size, 
        cond_dim=latent_size, 
        hidden=hidden_size, 
        n_layers=6
    ).to(device)
    
    # Inference components
    q_enc = PosteriorEncoder(
        data_size, 
        hidden_size
    ).to(device)
    
    q_affine = PosteriorAffine(
        latent_size, 
        hidden_size, 
        init_logstd=-0.5
    ).to(device)
    
    # Complete model
    model = FlowPriorMatchingCond(
        prior_flow, 
        p_obs_flow, 
        q_enc, 
        q_affine, 
        z0_prior
    ).to(device)
    
    print(f"Model built with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ═══════════════════════════════════════════════════════════════
    #                        Training Loop
    # ═══════════════════════════════════════════════════════════════
    
    # Optimizer (Adam works well for this)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training iterations
    iters = 10000
    pbar = trange(iters, desc="Training")
    
    for i in pbar:
        # Forward pass
        optim.zero_grad(set_to_none=True)
        loss = model(xs, ts)
        
        # Backward pass
        loss.backward()
        optim.step()
        
        # Update progress bar
        pbar.set_description(f"Loss: {loss.item():.6f}")
        
        # Periodically generate samples to check progress
        if i % 100 == 0:
            with torch.no_grad():
                # Generate new trajectories from the learned model
                T_vis = 1000  # High resolution for smooth visualization
                ts_vis = torch.linspace(0., 3., T_vis + 1, device=device).unsqueeze(1)
                
                # Sample initial states
                n_paths = 6
                z0_samples = z0_prior().rsample([n_paths])[:, 0, :]
                
                # Generate latent paths (no ODE solving!)
                z_paths = prior_flow.sample_paths_cond(ts_vis, z0_samples)
                
                # Generate observations
                Dx = data_size
                u0 = torch.zeros(n_paths*(T_vis+1), Dx, device=device)
                z_flat = z_paths.reshape(-1, latent_size)
                x_flat = p_obs_flow.flow.forward(u0, z_flat)
                x_paths = x_flat.reshape(n_paths, T_vis + 1, Dx)
                
                # Save visualization
                visualise_data(x_paths.detach().cpu(), 'samples.jpg')
    
    print("\nTraining complete! Check 'samples.jpg' for generated trajectories.")
    print("The model can now generate butterfly-like paths at any time resolution")
    print("without solving differential equations!")

"""
┌────────────────────────────────────────────────────────────────┐
│                     FINAL THOUGHTS                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  What we've accomplished:                                      │
│                                                                │
│  • Learned continuous dynamics without integration            │
│  • Can query any time point instantly                        │
│  • Exact likelihoods for optimization/search                 │
│  • Stable long-term generation                               │
│                                                                │
│  This approach opens doors for:                              │
│  • Fast simulation of complex systems                        │
│  • Gradient-based trajectory optimization                    │
│  • Real-time applications                                    │
│  • Learning from partial observations                        │
│                                                                │
│  Remember: Sometimes the best way to solve a hard problem    │
│  is to ask if we're solving the right problem!              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
"""
