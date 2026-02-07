"""
Full-Rank Discrete Distribution

Draws a random joint distribution over K^D outcomes from Dirichlet(1),
then samples from it. The distribution has arbitrary correlations that
a factorized base cannot capture.
"""

import torch
from torch import Tensor


class FullRankDiscrete:
    """
    Holds a fixed ground-truth joint distribution and samples from it.

    Usage:
        data = FullRankDiscrete(D=5, K=5, seed=42, device=device)
        x = data.sample(1024)          # (1024, D) LongTensor in {0..K-1}
        nll = data.nll(x)              # scalar, ground-truth NLL
        print(data.entropy)            # H(p*), lower bound on any model's NLL
    """

    def __init__(self, D: int, K: int, seed: int = 42, device: torch.device = torch.device("cpu")):
        self.D = D
        self.K = K
        self.device = device

        # Draw p ~ Dirichlet(1, ..., 1) over K^D outcomes
        rng = torch.Generator().manual_seed(seed)
        # Gamma(1,1) draws -> normalize = Dirichlet(1)
        gamma = torch.zeros(K ** D).exponential_(generator=rng)
        self.probs = (gamma / gamma.sum()).to(device)  # (K^D,)

        # Precompute entropy (best achievable NLL per sample)
        mask = self.probs > 0
        self.entropy = -(self.probs[mask] * self.probs[mask].log()).sum().item()

    def sample(self, n: int) -> Tensor:
        """Sample (n, D) LongTensor from the ground-truth distribution."""
        flat = torch.multinomial(self.probs, n, replacement=True)  # (n,)
        # Decode flat index -> multi-dim index via repeated div/mod
        x = torch.zeros(n, self.D, dtype=torch.long, device=self.device)
        remainder = flat
        for d in range(self.D - 1, -1, -1):
            x[:, d] = remainder % self.K
            remainder = remainder // self.K
        return x

    def nll(self, x: Tensor) -> Tensor:
        """Ground-truth NLL per sample, averaged. x: (n, D) LongTensor."""
        flat = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        for d in range(self.D):
            flat = flat * self.K + x[:, d]
        return -self.probs[flat].log().mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for D, K in [(2, 2), (5, 5), (5, 10), (10, 5)]:
        data = FullRankDiscrete(D, K, device=device)
        x = data.sample(4096)
        nll = data.nll(x)
        print(f"D={D:2d} K={K:2d} | H(p*)={data.entropy:.3f} nats | "
              f"empirical NLL={nll.item():.3f} | sample shape={tuple(x.shape)} "
              f"range=[{x.min().item()}, {x.max().item()}]")
