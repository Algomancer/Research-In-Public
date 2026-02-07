"""
Discrete Bipartite Flows from Scratch
======================================

Reference: Tran et al., "Discrete Flows: Invertible Generative Models
of Discrete Data" (NeurIPS 2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path

from data import FullRankDiscrete

Tensor = torch.Tensor
SCRIPT_DIR = Path(__file__).parent.resolve()
torch.set_float32_matmul_precision('high')

# =============================================================================
# Device
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FlowConfig:
    hidden_dim: int = 64
    depth: int = 2
    heads: int = 2
    mlp_mult: int = 4
    n_flows: int = 4
    tau: float = 0.1

    @property
    def kv_heads(self) -> int:
        return self.heads

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.heads


# =============================================================================
# Normalization
# =============================================================================

def norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


# =============================================================================
# Initialization
# =============================================================================

def xavier_init(m: nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


# =============================================================================
# RoPE (1D for sequence positions)
# =============================================================================

def build_rope_cache(seq_len: int, head_dim: int, device: torch.device,
                     base: float = 10000.0) -> Tuple[Tensor, Tensor]:
    dim_per_axis = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(dim_per_axis, device=device).float() / dim_per_axis))
    positions = torch.arange(seq_len, device=device).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (seq_len, dim_per_axis)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, S, H, D), cos/sin: (S, rope_dim) -> (B, S, H, D)"""
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().view(1, -1, 1, cos.shape[-1])   # (1, S, 1, rope_dim)
    sin = sin.float().view(1, -1, 1, sin.shape[-1])
    x_re, x_im = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out = torch.stack([x_re * cos - x_im * sin, x_re * sin + x_im * cos], -1)
    return out.flatten(-2).to(orig_dtype)


# =============================================================================
# AttnContext
# =============================================================================

@dataclass
class AttnContext:
    rope_cos: Tensor  # (seq_len, rope_dim)
    rope_sin: Tensor  # (seq_len, rope_dim)


def make_ctx(seq_len: int, rope_cos: Tensor, rope_sin: Tensor) -> AttnContext:
    return AttnContext(rope_cos=rope_cos[:seq_len], rope_sin=rope_sin[:seq_len])


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(dim, dim * mult, bias=False)
        self.down = nn.Linear(dim * mult, dim, bias=False)
        xavier_init(self.up)
        xavier_init(self.down)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.relu(self.up(x)).square())


# =============================================================================
# Attention
# =============================================================================

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.qkv = nn.Linear(dim, (heads + 2 * kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        xavier_init(self.qkv)
        xavier_init(self.out)

    def forward(self, x: Tensor, ctx: AttnContext) -> Tensor:
        """x: (B, S, dim) -> (B, S, dim)"""
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split(
            [self.heads * self.head_dim, self.kv_heads * self.head_dim,
             self.kv_heads * self.head_dim], -1
        )
        q = norm(q.view(B, S, self.heads, self.head_dim))
        k = norm(k.view(B, S, self.kv_heads, self.head_dim))
        v = v.view(B, S, self.kv_heads, self.head_dim)

        q = apply_rope(q, ctx.rope_cos, ctx.rope_sin)
        k = apply_rope(k, ctx.rope_cos, ctx.rope_sin)

        # (B, S, H, D) -> (B, H, S, D) for SDPA
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=self.scale,
        )
        return self.out(out.transpose(1, 2).reshape(B, S, -1))


# =============================================================================
# Transformer Blocks
# =============================================================================

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.attn = Attention(dim, heads, kv_heads)
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: Tensor, ctx: AttnContext) -> Tensor:
        x = x + self.attn(norm(x), ctx)
        return x + self.mlp(norm(x))


class Backbone(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, kv_heads: int,
                 mlp_mult: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, heads, kv_heads, mlp_mult)
            for _ in range(depth)
        ])
        self.depth = depth
        self.skip_alpha = nn.Parameter(torch.zeros(depth // 2))

    def forward(self, x: Tensor, ctx: AttnContext) -> Tensor:
        skips = []
        mid = len(self.blocks) // 2
        for idx, block in enumerate(self.blocks):
            if mid > 0 and idx >= mid:
                x = x + torch.sigmoid(self.skip_alpha[idx - mid]) * skips.pop()
            x = block(x, ctx)
            if mid > 0 and idx < mid:
                skips.append(x)
        return x


# =============================================================================
# Coupling Conditioner
# =============================================================================

class CouplingNet(nn.Module):
    """
    Transformer conditioner for a bipartite coupling layer.

    Takes identity-partition values x_id (batch, n_id) as LongTensor,
    embeds them alongside n_tr learnable query tokens, runs the full
    sequence through the transformer backbone, and reads off per-query
    logits (batch, n_tr, K) for the transform-partition shift parameters.
    """

    def __init__(self, n_id: int, n_tr: int, K: int, cfg: FlowConfig):
        super().__init__()
        self.n_id = n_id
        self.n_tr = n_tr
        self.K = K
        dim = cfg.hidden_dim

        # Input embeddings for identity tokens
        self.embed = nn.Embedding(K, dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.pos_embed_id = nn.Parameter(torch.randn(n_id, dim) * 0.02)

        # Learnable query tokens for transform dimensions
        self.query_tokens = nn.Parameter(torch.randn(n_tr, dim) * 0.02)

        self.backbone = Backbone(
            dim=dim, depth=cfg.depth, heads=cfg.heads,
            kv_heads=cfg.kv_heads, mlp_mult=cfg.mlp_mult,
        )

        # Per-token head: each query token projects to K logits
        self.head = nn.Linear(dim, K, bias=True)
        xavier_init(self.head)

    def forward(self, x_id: Tensor, ctx: AttnContext) -> Tensor:
        """
        x_id: (batch, n_id) LongTensor in {0..K-1}
        Returns: (batch, n_tr, K) logits
        """
        B = x_id.shape[0]
        # Embed identity tokens: (B, n_id, dim)
        h_id = self.embed(x_id) + self.pos_embed_id.unsqueeze(0)
        # Expand query tokens: (B, n_tr, dim)
        h_query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        # Concatenate: (B, n_id + n_tr, dim)
        h = torch.cat([h_id, h_query], dim=1)
        # Transformer processes full sequence
        h = self.backbone(h, ctx)
        # Read off the query positions: (B, n_tr, dim)
        h_out = h[:, self.n_id:, :]
        # Per-token projection to K logits: (B, n_tr, K)
        return self.head(norm(h_out))


# =============================================================================
# Straight-Through Estimator
# =============================================================================

def straight_through_one_hot(logits: Tensor, K: int, tau: float = 0.1) -> Tensor:
    """
    ST one-hot: forward uses argmax, backward uses softmax(logits/tau).

    logits: (..., K)
    Returns: (..., K) one-hot with ST gradients
    """
    soft = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(logits.argmax(-1), K).float()
    return hard - soft.detach() + soft


# =============================================================================
# Bipartite Coupling Layer
# =============================================================================

class BipartiteCoupling(nn.Module):
    """
    Single bipartite coupling layer with modular location shift.

    Splits dimensions into identity (id_dims) and transform (tr_dims).
    x_id passes through unchanged; x_tr is shifted by mu(x_id) mod K.

    sigma = 1 throughout (location-only, as in paper experiments).
    """

    def __init__(self, D: int, K: int, id_dims: list, tr_dims: list,
                 cfg: FlowConfig):
        super().__init__()
        self.D = D
        self.K = K
        self.id_dims = id_dims
        self.tr_dims = tr_dims
        self.n_id = len(id_dims)
        self.n_tr = len(tr_dims)
        self.tau = cfg.tau

        self.net = CouplingNet(self.n_id, self.n_tr, K, cfg)

    def get_mu_st(self, x_id: Tensor, ctx: AttnContext) -> Tensor:
        """Get ST one-hot shift parameters. Returns (batch, n_tr, K) in float32."""
        logits = self.net(x_id, ctx).float()
        return straight_through_one_hot(logits, self.K, self.tau)

    def forward(self, x: Tensor, ctx: AttnContext) -> Tensor:
        """Forward: y_tr = (x_tr + argmax(mu)) % K. x: (batch, D)"""
        x_id = x[:, self.id_dims]
        x_tr = x[:, self.tr_dims]
        mu_st = self.get_mu_st(x_id, ctx)  # (batch, n_tr, K)
        mu_hard = mu_st.argmax(-1)  # (batch, n_tr)
        y_tr = (x_tr + mu_hard) % self.K
        y = x.clone()
        y[:, self.tr_dims] = y_tr
        return y

    def inverse(self, y: Tensor, ctx: AttnContext) -> Tensor:
        """Inverse: x_tr = (y_tr - argmax(mu)) % K. y: (batch, D)"""
        y_id = y[:, self.id_dims]  # identity dims are same in x and y
        y_tr = y[:, self.tr_dims]
        mu_st = self.get_mu_st(y_id, ctx)
        mu_hard = mu_st.argmax(-1)
        x_tr = (y_tr - mu_hard) % self.K
        x = y.clone()
        x[:, self.tr_dims] = x_tr
        return x


# =============================================================================
# Discrete Flow
# =============================================================================

class DiscreteFlow(nn.Module):
    """
    Composed bipartite discrete flow.

    Base distribution: factorized Categorical with learnable logits.
    Flow: L bipartite coupling layers with alternating even/odd masks.

    log_prob uses circular convolution of ST shift distributions to
    propagate gradients through the composed discrete transforms.
    """

    def __init__(self, D: int, K: int, cfg: FlowConfig):
        super().__init__()
        self.D = D
        self.K = K
        self.cfg = cfg

        # Learnable factorized base distribution
        self.base_logits = nn.Parameter(torch.zeros(D, K))

        # Build alternating coupling layers
        even_dims = list(range(0, D, 2))
        odd_dims = list(range(1, D, 2))

        self.layers = nn.ModuleList()
        for i in range(cfg.n_flows):
            if i % 2 == 0:
                id_dims, tr_dims = even_dims, odd_dims
            else:
                id_dims, tr_dims = odd_dims, even_dims
            self.layers.append(
                BipartiteCoupling(D, K, id_dims, tr_dims, cfg)
            )

        # RoPE cache -- seq_len is n_id + n_tr = D for every layer
        max_seq = D
        head_dim = cfg.head_dim
        rope_cos, rope_sin = build_rope_cache(max_seq, head_dim, DEVICE)
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)

        # Precompute circular convolution index: idx[k, j] = (k - j) % K
        idx = torch.arange(K).unsqueeze(1) - torch.arange(K).unsqueeze(0)
        self.register_buffer('circ_idx', idx % K)  # (K, K)

    def _make_ctx(self, seq_len: int) -> AttnContext:
        return make_ctx(seq_len, self.rope_cos, self.rope_sin)

    def _circ_conv(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Circular convolution of two distributions over Z_K.

        a: (..., K) -- distribution of shift A
        b: (..., K) -- distribution of shift B
        result[k] = sum_j a[j] * b[(k-j) % K]

        This gives the distribution of (A + B) mod K.
        """
        # b_shifted[..., k, j] = b[..., (k-j) % K]
        b_shifted = b[..., self.circ_idx]  # (..., K, K)
        return (a.unsqueeze(-2) * b_shifted).sum(-1)  # (..., K)

    @torch.compile(fullgraph=True)
    def log_prob(self, y: Tensor) -> Tensor:
        """
        Differentiable log-probability of data y under the flow.

        Uses ST shift distributions and circular convolution to propagate
        gradients through the composed modular-arithmetic transforms.

        y: (batch, D) LongTensor in {0..K-1}
        Returns: (batch,) log-probabilities
        """
        B, D, K = y.shape[0], self.D, self.K
        device = y.device

        # Per-dimension shift distribution, initialized as delta at 0
        # shift_dist[b, d, k] = P(total shift = k) for dimension d
        shift_dist = torch.zeros(B, D, K, device=device)
        shift_dist[:, :, 0] = 1.0

        # Track hard-inverted values for conditioner inputs
        z = y.clone()

        # Process layers in reverse (inverse direction)
        for layer in reversed(self.layers):
            tr_dims = layer.tr_dims
            id_dims = layer.id_dims

            # Get ST one-hot shift from conditioner
            # seq_len = n_id + n_tr for the query-token architecture
            ctx = self._make_ctx(layer.n_id + layer.n_tr)
            mu_st = layer.get_mu_st(z[:, id_dims], ctx)  # (B, n_tr, K)

            # Circular-convolve the shift distribution for transform dims
            # The inverse shift is -mu, so we need to convolve with the
            # distribution of -mu: if mu has mass at k, -mu has mass at (-k)%K
            neg_idx = (-torch.arange(K, device=device)) % K
            neg_mu_st = mu_st[:, :, neg_idx]  # (B, n_tr, K)

            cur_shift = shift_dist[:, tr_dims, :]  # (B, n_tr, K)
            shift_dist = shift_dist.clone()
            shift_dist[:, tr_dims, :] = self._circ_conv(cur_shift, neg_mu_st)

            # Hard inverse for next layer's conditioner input
            mu_hard = mu_st.argmax(-1)  # (B, n_tr)
            z = z.clone()
            z[:, tr_dims] = (z[:, tr_dims] - mu_hard) % K

        # Compute log prob under base distribution
        log_base = F.log_softmax(self.base_logits.float(), dim=-1)  # (D, K)

        # For each dim d and shift k: base_prob at (y_d - k) % K
        shift_values = torch.arange(K, device=device)
        base_indices = (y.unsqueeze(-1) - shift_values.view(1, 1, K)) % K  # (B, D, K)
        # Gather base log probs
        log_base_expanded = log_base.unsqueeze(0).expand(B, -1, -1)  # (B, D, K)
        log_base_at_idx = log_base_expanded.gather(2, base_indices)  # (B, D, K)

        # Weighted sum: sum_k shift_dist[d,k] * exp(log_base[d, (y_d-k)%K])
        prob_per_dim = (shift_dist * log_base_at_idx.exp()).sum(-1)  # (B, D)
        # Clamp for log stability
        log_prob_per_dim = prob_per_dim.clamp(min=1e-30).log()

        return log_prob_per_dim.sum(-1)  # (B,)

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        """Sample from the flow by drawing from base and applying forward layers."""
        # Sample from factorized base
        base_probs = F.softmax(self.base_logits.float(), dim=-1)  # (D, K)
        z = torch.multinomial(base_probs.expand(n, -1, -1).reshape(n * self.D, self.K),
                              1).view(n, self.D)  # (n, D)

        # Apply layers in forward order
        for layer in self.layers:
            ctx = self._make_ctx(layer.n_id + layer.n_tr)
            z = layer.forward(z, ctx)

        return z


# =============================================================================
# Training
# =============================================================================

def _get_learned_joint(model: DiscreteFlow, D: int, K: int,
                       n_samples: int = 100_000) -> torch.Tensor:
    """Sample from model and estimate joint distribution. Returns (K^D,) on CPU."""
    samples = model.sample(n_samples)
    strides = K ** torch.arange(D - 1, -1, -1, device=samples.device)
    flat_idx = (samples * strides.unsqueeze(0)).sum(1)
    counts = torch.zeros(K ** D, device=samples.device)
    counts.scatter_add_(0, flat_idx, torch.ones(n_samples, device=samples.device))
    return (counts / n_samples).cpu()


def _compute_factorized(probs_flat: torch.Tensor, D: int, K: int) -> torch.Tensor:
    """Compute product of marginals from a flat joint. Returns (K^D,) on CPU."""
    import numpy as np
    p = probs_flat.cpu().numpy().reshape([K] * D)
    marginals = []
    for d in range(D):
        axes = tuple(i for i in range(D) if i != d)
        marginals.append(p.sum(axis=axes))
    factorized = marginals[0]
    for m in marginals[1:]:
        factorized = np.multiply.outer(factorized, m)
    return torch.from_numpy(factorized.ravel().copy()).float()


def _pairwise_marginal(probs_flat, D, K, di, dj):
    """K*K marginal for dims (di, dj) from flat joint. Returns numpy array."""
    import numpy as np
    p = probs_flat.cpu().numpy().reshape([K] * D)
    axes = tuple(d for d in range(D) if d != di and d != dj)
    return p.sum(axis=axes) if axes else p.copy()


def _pairwise_from_samples(samples, K, di, dj):
    """K*K marginal for dims (di, dj) from model samples. Returns numpy array."""
    n = samples.shape[0]
    idx = samples[:, di] * K + samples[:, dj]
    counts = torch.zeros(K * K, device=samples.device)
    counts.scatter_add_(0, idx, torch.ones(n, device=samples.device))
    return (counts / n).cpu().numpy().reshape(K, K)


def _top_pairs_by_mi(probs_flat, D, K, max_pairs=4):
    """Select dimension pairs with highest mutual information."""
    import numpy as np
    pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            marg = _pairwise_marginal(probs_flat, D, K, i, j)
            mi_i, mi_j = marg.sum(axis=1), marg.sum(axis=0)
            indep = np.outer(mi_i, mi_j)
            mask = (marg > 0) & (indep > 0)
            mi = np.sum(marg[mask] * np.log(marg[mask] / indep[mask]))
            pairs.append((mi, i, j))
    pairs.sort(reverse=True)
    return [(i, j) for _, i, j in pairs[:max_pairs]]


def _bubble_panel(ax, mat, title, subtitle, cmap, bg, fg, grid_color,
                  vmin=None, vmax=None, diverging=False,
                  show_cbar=False, cbar_ax=None,
                  xlabel="$x_1$", ylabel="$x_0$"):
    """Bubble chart on a discrete grid (from figure_twitter.py style)."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle

    K_ = mat.shape[0]
    if vmin is None:
        vmin = mat.min()
    if vmax is None:
        vmax = mat.max()

    if diverging:
        abs_max = max(abs(vmin), abs(vmax))
        norm_ = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm_ = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_)
    max_r = 0.44
    ref = (max(abs(vmin), abs(vmax)) if diverging else vmax) or 1.0

    for i in range(K_):
        for j in range(K_):
            val = mat[i, j]
            r = max_r * (abs(val) / ref) ** 0.5
            ax.add_patch(Circle((j, i), r, facecolor=sm.to_rgba(val),
                                edgecolor="none"))

    ax.set_xlim(-0.6, K_ - 0.4)
    ax.set_ylim(-0.6, K_ - 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(K_))
    ax.set_yticks(range(K_))
    ax.set_xticklabels(range(K_), fontsize=8)
    ax.set_yticklabels(range(K_), fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10, color=fg)
    ax.set_ylabel(ylabel, fontsize=10, color=fg)
    for i_ in range(K_ + 1):
        ax.axhline(i_ - 0.5, color=grid_color, linewidth=0.5, zorder=0)
        ax.axvline(i_ - 0.5, color=grid_color, linewidth=0.5, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color=fg)
    ax.text(0.5, -0.17, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="#9999aa",
            fontfamily="monospace")
    if show_cbar and cbar_ax is not None:
        cb = plt.colorbar(sm, cax=cbar_ax)
        cb.ax.tick_params(labelsize=7, colors=fg)
        cb.outline.set_edgecolor(grid_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(0.8)


def plot_training(losses: list, model: DiscreteFlow, data: FullRankDiscrete,
                  step: int, n_iter: int, tag: str = "training"):
    """Save {tag}.jpg: 16:9 — loss on top, bubble rows below."""
    import numpy as np

    bg, fg, accent = "#0f1117", "#e8e8ec", "#6c9cfc"
    grid_color = "#1e2030"
    prev = plt.rcParams.copy()
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg,
        "text.color": fg, "axes.labelcolor": fg,
        "xtick.color": fg, "ytick.color": fg,
        "font.family": "monospace", "font.size": 10,
    })

    D, K = data.D, data.K

    # Top-2 MI pairs (1 if D=2 since there's only 1 pair)
    pairs = _top_pairs_by_mi(data.probs, D, K, max_pairs=2)
    n_pair_rows = len(pairs)

    # Learned samples for marginal estimation
    samples = model.sample(100_000)

    # 16:9 — row 0 = loss (full width), rows 1+ = bubble triples
    fig = plt.figure(figsize=(16, 9))
    n_grid_rows = 1 + n_pair_rows
    height_ratios = [1.0] + [1.2] * n_pair_rows
    gs = fig.add_gridspec(n_grid_rows, 4,
                          width_ratios=[1, 1, 1, 0.05],
                          height_ratios=height_ratios,
                          wspace=0.25, hspace=0.50,
                          left=0.06, right=0.94,
                          top=0.88, bottom=0.06)

    fig.text(0.5, 0.95,
             f"Discrete Bipartite Flow  D={D}  K={K}  "
             f"step {step}/{n_iter}", ha="center", fontsize=15,
             fontweight="bold", color=fg)

    # --- Loss curve (top row, spans bubble columns) ---
    ax_loss = fig.add_subplot(gs[0, :3])
    ax_loss.set_facecolor(bg)
    steps_arr = np.arange(1, len(losses) + 1)
    ax_loss.plot(steps_arr, losses, color=accent, linewidth=1.0, alpha=0.6)
    if len(losses) > 50:
        w = max(len(losses) // 100, 10)
        smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax_loss.plot(np.arange(w, len(losses) + 1), smooth, color=accent,
                     linewidth=2.0)
    ax_loss.axhline(data.entropy, color="#ff6b6b", linestyle="--", linewidth=1.5,
                    label=f"H(p*) = {data.entropy:.3f}")
    factorized_probs = _compute_factorized(data.probs, D, K)
    mask_f = factorized_probs > 0
    H_fact = -(factorized_probs[mask_f] * factorized_probs[mask_f].log()).sum().item()
    ax_loss.axhline(H_fact, color="#666688", linestyle=":", linewidth=1.0,
                    label=f"H(marginals) = {H_fact:.3f}")
    ax_loss.set_xlabel("step", color=fg)
    ax_loss.set_ylabel("NLL (nats)", color=fg)
    ax_loss.set_title("Training Loss", fontsize=11, fontweight="bold", color=fg)
    ax_loss.legend(fontsize=8, facecolor=bg, edgecolor=grid_color,
                   labelcolor=fg, loc="upper right")
    ax_loss.set_xlim(0, n_iter)
    if losses:
        ylo = max(data.entropy - 0.2, 0)
        yhi = max(losses[0], H_fact) + 0.3
        ax_loss.set_ylim(ylo, yhi)
    ax_loss.tick_params(colors=fg)
    for spine in ax_loss.spines.values():
        spine.set_color(grid_color)

    # --- Pairwise marginal bubble charts (below loss) ---
    for pidx, (di, dj) in enumerate(pairs):
        grow = 1 + pidx  # grid row (0 is loss)

        true_marg = _pairwise_marginal(data.probs, D, K, di, dj)
        learned_marg = _pairwise_from_samples(samples, K, di, dj)
        marg_i, marg_j = true_marg.sum(axis=1), true_marg.sum(axis=0)
        factorized_ij = np.outer(marg_i, marg_j)
        residual = true_marg - factorized_ij

        mask_mi = (true_marg > 0) & (factorized_ij > 0)
        mi = float(np.sum(true_marg[mask_mi] * np.log(
            true_marg[mask_mi] / factorized_ij[mask_mi])))

        vmax_pair = max(float(true_marg.max()), float(learned_marg.max()))
        xl, yl = f"$x_{{{dj}}}$", f"$x_{{{di}}}$"

        ax_t = fig.add_subplot(gs[grow, 0])
        _bubble_panel(ax_t, true_marg, f"True  dims ({di},{dj})",
                      f"MI = {mi:.3f} nats",
                      "magma", bg, fg, grid_color,
                      vmin=0, vmax=vmax_pair, xlabel=xl, ylabel=yl)

        ax_l = fig.add_subplot(gs[grow, 1])
        l1_err = float(np.abs(learned_marg - true_marg).sum())
        _bubble_panel(ax_l, learned_marg, f"Learned  dims ({di},{dj})",
                      f"L1 = {l1_err:.3f}",
                      "magma", bg, fg, grid_color,
                      vmin=0, vmax=vmax_pair, xlabel=xl, ylabel=yl)

        ax_r = fig.add_subplot(gs[grow, 2])
        cbar_ax = fig.add_subplot(gs[grow, 3])
        _bubble_panel(ax_r, residual, f"Correlations ({di},{dj})",
                      f"MI = {mi:.3f}",
                      "RdBu_r", bg, fg, grid_color,
                      diverging=True, show_cbar=True, cbar_ax=cbar_ax,
                      xlabel=xl, ylabel=yl)

    plt.savefig(SCRIPT_DIR / f"{tag}.jpg", dpi=150, facecolor=bg)
    plt.close()
    plt.rcParams.update(prev)


def train(model: DiscreteFlow, data: FullRankDiscrete,
          n_iter: int = 20000, batch_size: int = 512, lr: float = 1e-3,
          warmup_frac: float = 0.05, max_grad_norm: float = 1.0,
          log_every: int = 500, plot_every: int = 500,
          tag: str = "training"):
    """Train the discrete flow with maximum likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    warmup = int(n_iter * warmup_frac)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(n_iter - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    pbar = trange(n_iter, desc=f"D={data.D} K={data.K}")
    best_nll = float('inf')
    losses = []

    for i in pbar:
        y = data.sample(batch_size)
        log_p = model.log_prob(y)
        loss = -log_p.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        nll = loss.item()
        losses.append(nll)

        if (i + 1) % log_every == 0:
            best_nll = min(best_nll, nll)
            pbar.set_postfix(nll=f"{nll:.3f}", H=f"{data.entropy:.3f}",
                             gap=f"{nll - data.entropy:.3f}")

        if (i + 1) % plot_every == 0 or (i + 1) == n_iter:
            model.eval()
            with torch.no_grad():
                plot_training(losses, model, data, i + 1, n_iter, tag=tag)
            model.train()

    return best_nll


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Paper Table 1 settings
    settings = [
        (2,  2,  5_000),
        (5,  5,  40_000),
        (5,  10, 40_000),
        (10, 5,  60_000),
    ]

    results = {}

    for D, K, n_iter in settings:
        tag = f"D{D}_K{K}"
        print(f"\n--- {tag} ---")
        cfg = FlowConfig(
            n_flows=4,
            tau=0.1,
        )

        data = FullRankDiscrete(D, K, seed=118, device=DEVICE)
        print(f"  Entropy H(p*) = {data.entropy:.3f}")

        model = DiscreteFlow(D, K, cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}  flows={cfg.n_flows}  tau={cfg.tau}")

        best_nll = train(model, data, n_iter=n_iter, batch_size=512, lr=1e-3,
                         tag=tag)
        results[(D, K)] = best_nll

        # Final evaluation on large batch
        model.eval()
        with torch.no_grad():
            y_eval = data.sample(4096)
            final_nll = -model.log_prob(y_eval).mean().item()
        model.train()

        print(f"  Final NLL: {final_nll:.3f} | H(p*): {data.entropy:.3f} | "
              f"Gap: {final_nll - data.entropy:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary (nats)")
    print("=" * 60)
    print(f"{'Setting':<12} {'H(p*)':<8} {'NLL':<8} {'Gap':<8}")
    print("-" * 36)
    for D, K, _ in settings:
        data = FullRankDiscrete(D, K, seed=118, device=DEVICE)
        ours = results[(D, K)]
        print(f"D={D:2d} K={K:2d}  {data.entropy:<8.3f} {ours:<8.3f} "
              f"{ours - data.entropy:<8.3f}")
