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
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().view(1, -1, 1, cos.shape[-1])
    sin = sin.float().view(1, -1, 1, sin.shape[-1])
    x_re, x_im = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out = torch.stack([x_re * cos - x_im * sin, x_re * sin + x_im * cos], -1)
    return out.flatten(-2).to(orig_dtype)


# =============================================================================
# AttnContext
# =============================================================================

@dataclass
class AttnContext:
    rope_cos: Tensor
    rope_sin: Tensor


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
    Transformer conditioner. Takes masked one-hot (B, D, K) — identity
    dims have values, transform dims are zeroed — runs D-position
    transformer, outputs (B, D, K) logits at every position.
    Only transform-position outputs are used (via masking in the layer).
    """

    def __init__(self, D: int, K: int, cfg: FlowConfig):
        super().__init__()
        dim = cfg.hidden_dim

        self.embed = nn.Linear(K, dim, bias=False)
        xavier_init(self.embed)
        self.pos_embed = nn.Parameter(torch.randn(D, dim) * 0.02)

        self.backbone = Backbone(
            dim=dim, depth=cfg.depth, heads=cfg.heads,
            kv_heads=cfg.kv_heads, mlp_mult=cfg.mlp_mult,
        )

        self.head = nn.Linear(dim, K, bias=True)
        xavier_init(self.head)

    def forward(self, x: Tensor, ctx: AttnContext) -> Tensor:
        """x: (B, D, K) masked one-hot -> (B, D, K) logits."""
        h = self.embed(x) + self.pos_embed.unsqueeze(0)
        h = self.backbone(h, ctx)
        return self.head(norm(h))


# =============================================================================
# Straight-Through Estimator
# =============================================================================

def straight_through_one_hot(logits: Tensor, K: int, tau: float = 0.1) -> Tensor:
    soft = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(logits.argmax(-1), K).float()
    return hard - soft.detach() + soft


# =============================================================================
# Modular Arithmetic in One-Hot Space
# =============================================================================

def one_hot_add(a: Tensor, b: Tensor) -> Tensor:
    """(a + b) % K in one-hot space via circular convolution."""
    return torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).real


def one_hot_minus(inputs: Tensor, shift: Tensor) -> Tensor:
    """(inputs - shift) % K in one-hot space.
    Reference: edward2 one_hot_minus — roll-based shift matrix + einsum.
    """
    K = inputs.shape[-1]
    shift_matrix = torch.stack(
        [shift.roll(i, dims=-1) for i in range(K)], dim=-2
    )
    return torch.einsum('...v,...uv->...u', inputs, shift_matrix)


# =============================================================================
# Bipartite Coupling Layer
# =============================================================================

class BipartiteCoupling(nn.Module):
    """
    Single bipartite coupling layer following the reference exactly.

    mask: (D, 1) — 1 for identity dims, 0 for transform dims.

    forward:  output = mask*x + (1-mask) * one_hot_add(loc, x)
    inverse:  output = mask*x + (1-mask) * one_hot_minus(x, loc)

    loc is conditioned on mask * x (identity dims only).
    """

    def __init__(self, D: int, K: int, mask: Tensor, cfg: FlowConfig):
        super().__init__()
        self.K = K
        self.tau = cfg.tau
        self.register_buffer('mask', mask.view(D, 1))
        self.net = CouplingNet(D, K, cfg)

    def _get_loc(self, x: Tensor, ctx: AttnContext) -> Tensor:
        """Compute ST one-hot location shift from masked input.
        Detach input so each layer's gradient is independent —
        avoids chaining STE gradients across layers.
        """
        logits = self.net((self.mask * x), ctx).float()
        return straight_through_one_hot(logits, self.K, self.tau)

    def forward(self, x: Tensor, ctx: AttnContext) -> Tensor:
        """Forward (generation): y = mask*x + (1-mask)*add(loc, x)."""
        loc = self._get_loc(x, ctx)
        return self.mask * x + (1.0 - self.mask) * one_hot_add(loc, x)

    def inverse(self, x: Tensor, ctx: AttnContext) -> Tensor:
        """Inverse (density): z = mask*x + (1-mask)*minus(x, loc)."""
        loc = self._get_loc(x, ctx)
        return self.mask * x + (1.0 - self.mask) * one_hot_minus(x, loc)


# =============================================================================
# Discrete Flow
# =============================================================================

class DiscreteFlow(nn.Module):
    """
    Composed bipartite discrete flow. Everything in one-hot space.
    """

    def __init__(self, D: int, K: int, cfg: FlowConfig):
        super().__init__()
        self.D = D
        self.K = K
        self.cfg = cfg

        self.base_logits = nn.Parameter(torch.randn(D, K) * 0.02)

        even_mask = torch.zeros(D)
        even_mask[::2] = 1.0
        odd_mask = 1.0 - even_mask

        self.layers = nn.ModuleList()
        for i in range(cfg.n_flows):
            mask = even_mask if i % 2 == 0 else odd_mask
            self.layers.append(BipartiteCoupling(D, K, mask, cfg))

        head_dim = cfg.head_dim
        rope_cos, rope_sin = build_rope_cache(D, head_dim, DEVICE)
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)

    def _ctx(self) -> AttnContext:
        return AttnContext(rope_cos=self.rope_cos, rope_sin=self.rope_sin)

    @torch.compile(fullgraph=True)
    def log_prob(self, y: Tensor) -> Tensor:
        """
        Differentiable log p(y).

        y: (batch, D) LongTensor -> (batch,) log-probs
        """
        x = F.one_hot(y, self.K).float()
        ctx = self._ctx()

        for layer in reversed(self.layers):
            x = layer.inverse(x, ctx)

        log_base = F.log_softmax(self.base_logits.float(), dim=-1)  # (D, K)
        return (log_base.unsqueeze(0) * x).sum(-1).sum(-1)       # (B,)
    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        """Sample n points. Returns (n, D) LongTensor."""
        base_probs = F.softmax(self.base_logits.float(), dim=-1)
        z = torch.multinomial(
            base_probs.expand(n, -1, -1).reshape(n * self.D, self.K), 1
        ).view(n, self.D)

        x = F.one_hot(z, self.K).float()
        ctx = self._ctx()
        for layer in self.layers:
            x = layer.forward(x, ctx)
        return x.argmax(-1)

    @torch.no_grad()
    def inverse(self, y: Tensor) -> Tensor:
        """Map observations to base. Returns (n, D) LongTensor."""
        x = F.one_hot(y, self.K).float()
        ctx = self._ctx()
        for layer in reversed(self.layers):
            x = layer.inverse(x, ctx)
        return x.argmax(-1)


# =============================================================================
# Training Utilities
# =============================================================================

def _compute_factorized(probs_flat: torch.Tensor, D: int, K: int) -> torch.Tensor:
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
    import numpy as np
    p = probs_flat.cpu().numpy().reshape([K] * D)
    axes = tuple(d for d in range(D) if d != di and d != dj)
    return p.sum(axis=axes) if axes else p.copy()


def _pairwise_from_samples(samples, K, di, dj):
    n = samples.shape[0]
    idx = samples[:, di] * K + samples[:, dj]
    counts = torch.zeros(K * K, device=samples.device)
    counts.scatter_add_(0, idx, torch.ones(n, device=samples.device))
    return (counts / n).cpu().numpy().reshape(K, K)


def _top_pairs_by_mi(probs_flat, D, K, max_pairs=4):
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
    pairs = _top_pairs_by_mi(data.probs, D, K, max_pairs=2)
    n_pair_rows = len(pairs)
    samples = model.sample(100_000)

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

    for pidx, (di, dj) in enumerate(pairs):
        grow = 1 + pidx

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


# =============================================================================
# Training
# =============================================================================

def train(model: DiscreteFlow, data: FullRankDiscrete,
          n_iter: int = 20000, batch_size: int = 512, lr: float = 1e-3,
          warmup_frac: float = 0.05, max_grad_norm: float = 1.0,
          log_every: int = 100, plot_every: int = 500,
          tag: str = "training"):
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

    D, K = data.D, data.K
    strides = K ** torch.arange(D - 1, -1, -1, device=data.device)

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
            model.eval()
            with torch.no_grad():
                # Round-trip test
                y_test = data.sample(2048)
                z_inv = model.inverse(y_test)
                ctx = model._ctx()
                y_rt = F.one_hot(z_inv, K).float()
                for layer in model.layers:
                    y_rt = layer.forward(y_rt, ctx)
                rt_acc = (y_rt.argmax(-1) == y_test).all(dim=1).float().mean().item()

                # TV distance
                n_tv = 50_000
                samples = model.sample(n_tv)
                flat_idx = (samples * strides.unsqueeze(0)).sum(1)
                counts = torch.zeros(K ** D, device=samples.device)
                counts.scatter_add_(0, flat_idx,
                                    torch.ones(n_tv, device=samples.device))
                p_model = counts / n_tv
                tv = 0.5 * (data.probs - p_model).abs().sum().item()
            model.train()

            pbar.set_postfix(nll=f"{nll:.3f}", H=f"{data.entropy:.3f}",
                             gap=f"{nll - data.entropy:.3f}",
                             rt=f"{rt_acc:.3f}", tv=f"{tv:.3f}")

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
    settings = [
        #(2,  2,  5_000),
        (5,  5,  20_000),
        #(5,  10, 40_000),
        #(10, 5,  60_000),
    ]

    results = {}

    for D, K, n_iter in settings:
        tag = f"D{D}_K{K}"
        print(f"\n--- {tag} ---")
        cfg = FlowConfig(
            n_flows=4,
            tau=1.0,
        )

        data = FullRankDiscrete(D, K, seed=118, device=DEVICE)
        print(f"  Entropy H(p*) = {data.entropy:.3f}")

        model = DiscreteFlow(D, K, cfg).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}  flows={cfg.n_flows}  tau={cfg.tau}")

        best_nll = train(model, data, n_iter=n_iter, batch_size=512, lr=1e-4,
                         tag=tag)
        results[(D, K)] = best_nll

        model.eval()
        with torch.no_grad():
            y_eval = data.sample(4096)
            final_nll = -model.log_prob(y_eval).mean().item()
        model.train()

        print(f"  Final NLL: {final_nll:.3f} | H(p*): {data.entropy:.3f} | "
              f"Gap: {final_nll - data.entropy:.3f}")

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
