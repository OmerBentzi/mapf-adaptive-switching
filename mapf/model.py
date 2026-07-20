"""Actor-Critic network for the PPO agent.

The convolutional trunk ends with ``AdaptiveAvgPool2d``, which makes the
network *independent of the observation radius*: a model trained with
``obs_radius=5`` (11x11 windows) can be evaluated zero-shot with radius 3 or 7
windows.  This supports the paper's second research axis ("how much the agents
see") without training a separate network per radius.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RadiusAgnosticPool(nn.Module):
    """Drop-in replacement for ``nn.AdaptiveAvgPool2d`` computing the exact
    same output (identical bin boundaries), expressed as two matmuls.

    Needed because the native MPS kernel rejects input sizes that are not
    divisible by the output size (pytorch#96056), e.g. 11x11 -> 5x5.
    Parameter-free, so checkpoints stay interchangeable with models built
    with ``nn.AdaptiveAvgPool2d``.
    """

    def __init__(self, output_size):
        super().__init__()
        self.out_h, self.out_w = output_size
        self._cache = {}

    @staticmethod
    def _weights(in_size, out_size, device, dtype):
        w = torch.zeros(out_size, in_size, device=device, dtype=dtype)
        for i in range(out_size):
            lo = (i * in_size) // out_size
            hi = -((-(i + 1) * in_size) // out_size)
            w[i, lo:hi] = 1.0 / (hi - lo)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        key = (h, w, x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = (
                self._weights(h, self.out_h, x.device, x.dtype),
                self._weights(w, self.out_w, x.device, x.dtype))
        rows, cols = self._cache[key]
        return rows @ x @ cols.T


class ActorCritic(nn.Module):
    """Shared conv trunk with separate policy (actor) and value (critic) heads.

    Parameters
    ----------
    in_channels:
        Number of observation channels (4: obstacles / target / agents /
        visited -- see :mod:`mapf.env`).
    n_actions:
        Size of the discrete action space (5 in POGEMA).
    """

    def __init__(self, in_channels: int = 4, n_actions: int = 5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            RadiusAgnosticPool((5, 5)),          # radius-agnostic
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """Returns ``(logits [B, n_actions], value [B])``."""
        features = self.trunk(x)
        return self.actor(features), self.critic(features).squeeze(-1)

    @torch.no_grad()
    def act_greedy(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Deterministic (argmax) actions for evaluation."""
        logits, _ = self(obs_batch)
        return logits.argmax(dim=-1)
