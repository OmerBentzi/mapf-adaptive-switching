"""PPO training utilities.

Fixes over the first version of the project:

* **Truncation bootstrap** -- when an episode is cut by the step limit the
  value of the final state is used to bootstrap the advantage estimate
  (previously truncation was treated like termination, biasing the critic).
* **Multi-episode buffer + minibatches** -- updates run on at least
  ``min_batch`` transitions with shuffled minibatches instead of one
  full-batch update per episode, which stabilises learning.
* Trajectories are stored per agent per episode, so GAE never has to mask
  intermediate ``done`` flags.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_gae(rewards, values, gamma: float, lam: float,
                bootstrap_value: float = 0.0):
    """Generalised Advantage Estimation for one trajectory.

    ``bootstrap_value`` is ``V(s_T)`` when the trajectory was *truncated*
    and ``0`` when it properly *terminated*.
    Returns ``(returns, advantages)`` as float32 numpy arrays.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else bootstrap_value
        delta = rewards[t] + gamma * next_value - values[t]
        g = delta + gamma * lam * g
        advantages[t] = g
    returns = advantages + np.asarray(values, dtype=np.float32)
    return returns, advantages


class PPOTrainer:
    """Clipped-surrogate PPO with a simple trajectory buffer."""

    def __init__(self, model, device, lr: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, clip: float = 0.2, epochs: int = 4,
                 minibatch: int = 512, min_batch: int = 2048,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=300, gamma=0.7)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch, self.min_batch = epochs, minibatch, min_batch
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.max_grad_norm = max_grad_norm
        self._buffer = []
        self._size = 0

    # ------------------------------------------------------------------ #
    def add_trajectory(self, obs, actions, log_probs, values, rewards,
                       truncated: bool, bootstrap_value: float):
        """Store one agent's episode trajectory."""
        if not rewards:
            return
        returns, advantages = compute_gae(
            rewards, values, self.gamma, self.lam,
            bootstrap_value if truncated else 0.0)
        self._buffer.append((np.asarray(obs, dtype=np.float32),
                             np.asarray(actions, dtype=np.int64),
                             np.asarray(log_probs, dtype=np.float32),
                             returns, advantages))
        self._size += len(rewards)

    def ready(self) -> bool:
        return self._size >= self.min_batch

    # ------------------------------------------------------------------ #
    def update(self):
        """Run PPO epochs over the buffered transitions; returns stats."""
        obs = torch.as_tensor(np.concatenate([b[0] for b in self._buffer]),
                              device=self.device)
        actions = torch.as_tensor(np.concatenate([b[1] for b in self._buffer]),
                                  device=self.device)
        old_logp = torch.as_tensor(np.concatenate([b[2] for b in self._buffer]),
                                   device=self.device)
        returns = torch.as_tensor(np.concatenate([b[3] for b in self._buffer]),
                                  device=self.device)
        adv = torch.as_tensor(np.concatenate([b[4] for b in self._buffer]),
                              device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        # accumulate loss stats on-device; .item() per minibatch forces an
        # MPS pipeline sync between optimizer steps
        totals = torch.zeros(3, device=self.device)
        batches = 0
        for _ in range(self.epochs):
            for start in range(0, n, self.minibatch):
                idx = torch.randperm(n, device=self.device)[start:start + self.minibatch]
                logits, values = self.model(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                totals += torch.stack([policy_loss.detach(),
                                       value_loss.detach(),
                                       entropy.detach()])
                batches += 1
        self.scheduler.step()
        self._buffer, self._size = [], 0
        policy_loss, value_loss, entropy = (totals / max(batches, 1)).cpu().tolist()
        return {"policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": entropy}

    # ------------------------------------------------------------------ #
    def state_dict(self):
        return {"model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
