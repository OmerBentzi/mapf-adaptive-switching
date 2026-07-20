"""Planning, deadlock escape, and the Deadlock-Aware Adaptive Switching policy.

Five evaluated configurations, all sharing the same ``select_actions``
interface:

* :class:`AStarPolicy`            -- planning only,
* :class:`EscapePolicy` (A*)      -- planning + smart escape,
* :class:`RLPolicy`               -- learned policy only,
* :class:`EscapePolicy` (RL)      -- learned policy + smart escape,
* :class:`SwitchingPolicy`        -- **our contribution**: deadlock-aware
  adaptive switching between A*, RL and escape.

Partial observability is respected everywhere: the planner of agent *i* uses
only (a) agent *i*'s accumulated obstacle memory, (b) agent *i*'s own goal
coordinate (agents always know their own goal in POGEMA), and (c) other agents
*currently visible* in agent *i*'s field of view.  No global information is
shared between agents.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .env import DELTA_TO_ACTION, MOVES, manhattan

# --------------------------------------------------------------------------- #
# A* on the agent's accumulated memory map
# --------------------------------------------------------------------------- #


def a_star(known_obstacles: np.ndarray, start, goal,
           blocked=frozenset(), max_expansions: int = 20_000):
    """A* over a partially known grid.

    Unobserved cells are treated as free (optimistic assumption, standard for
    navigation under partial observability); ``blocked`` adds temporary
    obstacles (currently visible agents).  The goal cell is never treated as
    blocked.  Returns the path ``[start, ..., goal]`` or ``None``.
    """
    if start == goal:
        return [start]
    h, w = known_obstacles.shape
    open_heap = [(manhattan(start, goal), 0, start)]
    g_cost = {start: 0}
    parent = {}
    expansions = 0
    while open_heap and expansions < max_expansions:
        _, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            path = [cell]
            while cell in parent:
                cell = parent[cell]
                path.append(cell)
            return path[::-1]
        if g > g_cost.get(cell, float("inf")):
            continue                       # stale heap entry
        expansions += 1
        for dx, dy in MOVES[1:]:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if known_obstacles[nxt] == 1 or (nxt in blocked and nxt != goal):
                continue
            ng = g + 1
            if ng < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = ng
                parent[nxt] = cell
                heapq.heappush(open_heap, (ng + manhattan(nxt, goal), ng, nxt))
    return None


# --------------------------------------------------------------------------- #
# deadlock detection + escape
# --------------------------------------------------------------------------- #


class DeadlockMonitor:
    """Detects agents that stopped making progress.

    An agent is considered deadlocked when its last ``window`` positions
    contain at most two distinct cells -- this catches both frozen agents and
    two-cell oscillations (livelocks).  When triggered, the agent enters
    escape mode for ``escape_steps`` steps and its history is cleared so the
    trigger does not re-fire immediately.
    """

    def __init__(self, num_agents: int, window: int = 4, escape_steps: int = 3):
        self.window = window
        self.escape_steps = escape_steps
        self.history = [deque(maxlen=window) for _ in range(num_agents)]
        self.countdown = [0] * num_agents

    def in_escape(self, i: int, position) -> bool:
        """Update agent *i* with its current position; True while escaping."""
        if self.countdown[i] > 0:
            self.countdown[i] -= 1
            return True
        self.history[i].append(position)
        if (len(self.history[i]) == self.window
                and len(set(self.history[i])) <= 2):
            self.countdown[i] = self.escape_steps - 1
            self.history[i].clear()
            return True
        return False


def escape_action(obs: np.ndarray, rng: np.random.Generator,
                  previous_position=None, current_position=None) -> int:
    """Pick a random *safe* direction to break a deadlock.

    A direction is safe when the neighbouring cell is neither an obstacle
    (channel 0) nor occupied by a visible agent (channel 2).  Stepping back to
    ``previous_position`` is avoided when an alternative exists, so escape
    actually leaves the conflict area instead of bouncing.
    """
    c = obs.shape[-1] // 2
    options = []
    for action, (dx, dy) in enumerate(MOVES):
        if action == 0:
            continue
        if obs[0][c + dx, c + dy] == 0 and obs[2][c + dx, c + dy] == 0:
            options.append(action)
    if previous_position is not None and current_position is not None and len(options) > 1:
        back = (previous_position[0] - current_position[0],
                previous_position[1] - current_position[1])
        back_action = DELTA_TO_ACTION.get(back)
        options = [a for a in options if a != back_action] or options
    return int(rng.choice(options)) if options else 0


# --------------------------------------------------------------------------- #
# policies (shared interface: select_actions(env, obs) -> list[int])
# --------------------------------------------------------------------------- #


class AStarPolicy:
    """Plan with A* on the agent's memory map; replan lazily.

    A cached path is reused until (a) the agent deviated from it, (b) the next
    cell is blocked by a known obstacle or a visible agent, or (c)
    ``replan_every`` steps elapsed (new observations may have revealed a
    better route).
    """

    name = "A* Only"

    def __init__(self, env, replan_every: int = 8, seed: int = 0):
        self.env = env
        self.replan_every = replan_every
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.paths = {}
        self.age = {}

    def _action_for(self, i: int, obs_i: np.ndarray) -> int:
        env = self.env
        pos, goal = env.agent_position(i), env.agent_goal(i)
        visible = env.visible_agent_cells(i)
        path = self.paths.get(i)
        self.age[i] = self.age.get(i, 0) + 1
        stale = (not path or pos not in path
                 or self.age[i] % self.replan_every == 0)
        if not stale:
            nxt = path[path.index(pos) + 1] if path.index(pos) + 1 < len(path) else None
            stale = (nxt is None or nxt in visible
                     or env.known_obstacles(i)[nxt] == 1)
        if stale:
            path = a_star(env.known_obstacles(i), pos, goal, blocked=visible)
            self.paths[i] = path
        if not path or pos not in path or path.index(pos) + 1 >= len(path):
            # fully enclosed by known obstacles / agents: nudge randomly
            return escape_action(obs_i, self.rng, current_position=pos)
        nxt = path[path.index(pos) + 1]
        return DELTA_TO_ACTION[(nxt[0] - pos[0], nxt[1] - pos[1])]

    def select_actions(self, env, obs):
        return [self._action_for(i, obs[i]) if env.active(i) else 0
                for i in range(env.num_agents)]


class RLPolicy:
    """Actions from a trained :class:`~mapf.model.ActorCritic`.

    ``stochastic=False`` (default) takes the argmax action.  With
    ``stochastic=True`` actions are *sampled* from the policy distribution --
    the regime the network was actually trained in; greedy evaluation of a
    stochastically-trained policy can lock into oscillations that sampling
    breaks.  Sampling uses a seeded CPU generator so paired-seed evaluation
    stays reproducible on any device.
    """

    name = "RL Only"

    def __init__(self, model, device, stochastic: bool = False, seed: int = 0):
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.stochastic = stochastic
        if stochastic:
            self.name = "RL Only (sampled)"
            self.gen = torch.Generator().manual_seed(seed)

    def reset(self):
        pass

    def actions_for(self, obs_subset):
        """Batched actions for a list of observations."""
        if not obs_subset:
            return []
        batch = self.torch.as_tensor(np.stack(obs_subset),
                                     dtype=self.torch.float32,
                                     device=self.device)
        if not self.stochastic:
            return self.model.act_greedy(batch).cpu().tolist()
        with self.torch.no_grad():
            logits, _ = self.model(batch)
        probs = self.torch.softmax(logits, dim=-1).cpu()
        return self.torch.multinomial(probs, 1,
                                      generator=self.gen).squeeze(-1).tolist()

    def select_actions(self, env, obs):
        idx = [i for i in range(env.num_agents) if env.active(i)]
        acts = self.actions_for([obs[i] for i in idx])
        out = [0] * env.num_agents
        for i, a in zip(idx, acts):
            out[i] = a
        return out


class RandomPolicy:
    """Uniform-random movement policy.

    Used as an ablation drop-in for the RL slot of
    :class:`SwitchingPolicy`, answering "is the learned policy load-bearing,
    or does the A*/escape scaffold do all the work?".
    """

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def actions_for(self, obs_subset):
        return [int(a) for a in self.rng.integers(0, len(MOVES),
                                                  size=len(obs_subset))]

    def select_actions(self, env, obs):
        idx = [i for i in range(env.num_agents) if env.active(i)]
        acts = self.actions_for([obs[i] for i in idx])
        out = [0] * env.num_agents
        for i, a in zip(idx, acts):
            out[i] = a
        return out


class EscapePolicy:
    """Wrap any base policy with deadlock detection + smart escape."""

    def __init__(self, base, num_agents: int, window: int = 4,
                 escape_steps: int = 3, seed: int = 0):
        self.base = base
        self.name = base.name.replace(" Only", "") + " + Escape"
        self.monitor = DeadlockMonitor(num_agents, window, escape_steps)
        self.rng = np.random.default_rng(seed)
        self.prev_pos = [None] * num_agents

    def reset(self):
        self.base.reset()
        self.monitor = DeadlockMonitor(len(self.prev_pos),
                                       self.monitor.window,
                                       self.monitor.escape_steps)
        self.prev_pos = [None] * len(self.prev_pos)

    def select_actions(self, env, obs):
        base_actions = self.base.select_actions(env, obs)
        out = []
        for i in range(env.num_agents):
            if not env.active(i):
                out.append(0)
                continue
            pos = env.agent_position(i)
            if self.monitor.in_escape(i, pos):
                out.append(escape_action(obs[i], self.rng,
                                         self.prev_pos[i], pos))
            else:
                out.append(base_actions[i])
            self.prev_pos[i] = pos
        return out


class SwitchingPolicy:
    """Deadlock-Aware Adaptive Switching (our contribution).

    Per agent, per step:

    * **escape** when the :class:`DeadlockMonitor` fires -- neither planning
      nor learning resolves persistent deadlocks reliably;
    * **RL** when other agents are visible in the FOV -- the learned policy
      was trained specifically on multi-agent interference;
    * **A*** otherwise -- optimal and cheap when navigating alone on the
      accumulated map.

    Mode usage is counted in :attr:`mode_counts` and reported by the
    evaluation script.
    """

    name = "Switching (A*/RL) + Escape"

    def __init__(self, astar: AStarPolicy, rl: RLPolicy, num_agents: int,
                 window: int = 4, escape_steps: int = 3, seed: int = 0):
        self.astar = astar
        self.rl = rl
        self.monitor = DeadlockMonitor(num_agents, window, escape_steps)
        self.rng = np.random.default_rng(seed)
        self.prev_pos = [None] * num_agents
        self.mode_counts = {"astar": 0, "rl": 0, "escape": 0}

    def reset(self):
        self.astar.reset()
        self.monitor = DeadlockMonitor(len(self.prev_pos),
                                       self.monitor.window,
                                       self.monitor.escape_steps)
        self.prev_pos = [None] * len(self.prev_pos)

    def select_actions(self, env, obs):
        out = [0] * env.num_agents
        rl_agents = []
        for i in range(env.num_agents):
            if not env.active(i):
                continue
            pos = env.agent_position(i)
            if self.monitor.in_escape(i, pos):
                out[i] = escape_action(obs[i], self.rng, self.prev_pos[i], pos)
                self.mode_counts["escape"] += 1
            elif obs[i][2].sum() > 0:          # other agents in FOV
                rl_agents.append(i)
                self.mode_counts["rl"] += 1
            else:
                out[i] = self.astar._action_for(i, obs[i])
                self.mode_counts["astar"] += 1
            self.prev_pos[i] = pos
        for i, a in zip(rl_agents, self.rl.actions_for([obs[i] for i in rl_agents])):
            out[i] = a
        return out
