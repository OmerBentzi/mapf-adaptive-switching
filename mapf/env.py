"""POGEMA-based Multi-Agent Pathfinding environment wrapper.

This wrapper builds on ``pogema.grid.Grid`` (POGEMA 1.4.0) and fixes several
issues found in the first version of this project:

1. **Reward** -- the old ``+0.05 if target visible`` bonus fired on *every*
   step because POGEMA projects (clamps) the target onto the border of the
   observation window when it is out of range.  The net per-step reward was
   therefore positive (+0.04), which paid agents for wandering instead of
   finishing.  It is replaced with distance-progress shaping (see
   :meth:`MAPFEnv._reward`).

2. **Agents channel** -- the old code built the "nearby agents" channel with
   swapped row/column indices (transposed w.r.t. the obstacle and target
   channels).  We now use ``Grid.get_positions`` which is natively aligned
   with the POGEMA ``[dx, dy]`` convention.

3. **Goal semantics** -- agents that reach their goal are now *removed* from
   the map via ``Grid.hide_agent`` (POGEMA ``on_target='finish'`` semantics).
   Previously they stayed on their goal cells forever and acted as walls,
   artificially inflating deadlocks.

4. **Move order** -- agents move in a *randomized* order every step instead of
   always index 0 first, removing a systematic priority bias of the
   sequential ``Grid.move`` API.

5. **Memory** -- the old per-agent memory was a single local window that was
   element-wise ``max``-ed with observations taken at *different* world
   positions, i.e. meaningless superposition.  We now maintain, per agent:

   * a **global obstacle-memory map** (everything the agent has ever seen,
     written at world coordinates) -- used by the A* planner, *not* fed to
     the network;
   * a **global visited map** (cells the agent has stood on) whose *local
     slice* is fed to the network as channel 3, giving the policy an
     anti-looping signal that is meaningful inside the local window.

Observation per agent: ``float32 (4, 2r+1, 2r+1)`` with POGEMA convention
``[delta_row(=dx), delta_col(=dy)]``:

* ``0`` obstacles in field of view (1 = obstacle, includes border padding),
* ``1`` projected target (single cell, clamped to the window border when the
  goal is out of range),
* ``2`` other visible agents (self excluded),
* ``3`` visited cells inside the field of view.
"""

from __future__ import annotations

import numpy as np
from pogema import GridConfig
from pogema.grid import Grid

#: POGEMA action semantics: 0 stay, 1 up (-row), 2 down (+row), 3 left, 4 right.
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
DELTA_TO_ACTION = {d: a for a, d in enumerate(MOVES)}


def manhattan(a, b):
    """Manhattan (L1) distance between two ``(row, col)`` cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class MAPFEnv:
    """Partially-observable multi-agent grid environment.

    Parameters
    ----------
    grid_kwargs:
        Keyword arguments forwarded to :class:`pogema.GridConfig`
        (``num_agents``, ``size``, ``density``, ``obs_radius``,
        ``max_episode_steps``, or ``map``/``agents_xy``/``targets_xy`` for
        deterministic instances).
    """

    #: reward constants (documented in the module docstring / paper)
    R_GOAL = 5.0          # terminal reward for reaching the goal
    R_STEP = -0.01        # per-step time cost
    R_PROGRESS = 0.05     # x (previous distance - new distance) to *true* goal
    R_BLOCKED = -0.05     # movement action attempted but the agent did not move

    def __init__(self, **grid_kwargs):
        self.grid_kwargs = dict(grid_kwargs)
        self.max_steps = int(self.grid_kwargs.get("max_episode_steps", 256))
        self.grid: Grid | None = None
        self.steps = 0
        self.rng = np.random.default_rng()
        # per-agent global maps, created in reset()
        self.memory: list[np.ndarray] = []    # known obstacles (planner)
        self.visited: list[np.ndarray] = []   # visited cells (net channel 3)
        self._done: list[bool] = []

    # ------------------------------------------------------------------ #
    # episode control
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        """Start a new episode; returns ``(observations, info)``."""
        if seed is None:
            seed = int(self.rng.integers(0, 1_000_000))
        # reseed the dynamics RNG (move order) from the episode seed so that
        # paired-seed evaluations share identical dynamics and reruns are
        # reproducible, not just map-paired
        self.rng = np.random.default_rng(seed)
        cfg = GridConfig(seed=seed, **self.grid_kwargs)
        self.grid = Grid(cfg)
        self.config = cfg
        self.steps = 0
        na = cfg.num_agents
        full_shape = self.grid.obstacles.shape  # padded (size+2r, size+2r)
        self.memory = [np.zeros(full_shape, dtype=np.float32) for _ in range(na)]
        self.visited = [np.zeros(full_shape, dtype=np.float32) for _ in range(na)]
        self._done = [False] * na
        for i in range(na):
            self._sense(i)
        return [self._obs(i) for i in range(na)], {"seed": seed}

    @property
    def num_agents(self) -> int:
        return self.config.num_agents

    def active(self, i: int) -> bool:
        return not self._done[i]

    # ------------------------------------------------------------------ #
    # observation helpers (also used by the planners)
    # ------------------------------------------------------------------ #
    def agent_position(self, i):
        """Current ``(row, col)`` of agent *i* in padded map coordinates."""
        return tuple(self.grid.positions_xy[i])

    def agent_goal(self, i):
        """Goal ``(row, col)`` of agent *i* (agents always know their own goal)."""
        return tuple(self.grid.finishes_xy[i])

    def known_obstacles(self, i) -> np.ndarray:
        """Agent *i*'s accumulated obstacle map (1 = seen obstacle)."""
        return self.memory[i]

    def visible_agent_cells(self, i):
        """World coordinates of *other* agents currently inside *i*'s FOV."""
        r = self.config.obs_radius
        window = self.grid.get_positions(i)
        x, y = self.agent_position(i)
        cells = set()
        for wx, wy in np.argwhere(window > 0):
            if wx == r and wy == r:      # skip self at the window center
                continue
            cells.add((x - r + int(wx), y - r + int(wy)))
        return cells

    def _sense(self, i):
        """Write the current ground-truth FOV into agent *i*'s global maps."""
        r = self.config.obs_radius
        x, y = self.agent_position(i)
        window = self.grid.get_obstacles_for_agent(i)
        # observations are ground truth for the window -> plain assignment
        self.memory[i][x - r:x + r + 1, y - r:y + r + 1] = window
        self.visited[i][x, y] = 1.0

    def _obs(self, i) -> np.ndarray:
        """4-channel observation for agent *i* (zeros once the agent is done)."""
        r = self.config.obs_radius
        s = 2 * r + 1
        if self._done[i]:
            return np.zeros((4, s, s), dtype=np.float32)
        obstacles = self.grid.get_obstacles_for_agent(i)
        target = self.grid.get_square_target(i)
        agents = self.grid.get_positions(i).copy()
        agents[r, r] = 0.0                       # exclude self
        x, y = self.agent_position(i)
        visited = self.visited[i][x - r:x + r + 1, y - r:y + r + 1]
        return np.stack([obstacles, target, agents, visited]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def step(self, actions):
        """Advance one time step.

        Returns ``(obs, rewards, terminated, truncated, info)`` where the
        ``terminated[i]`` flag is True only on the step agent *i* reaches its
        goal, and ``truncated[i]`` is True for agents still active when the
        step limit is hit.
        """
        assert self.grid is not None, "call reset() first"
        self.steps += 1
        na = self.num_agents
        rewards = [0.0] * na
        terminated = [False] * na
        truncated = [False] * na
        blocked = [False] * na

        order = self.rng.permutation(na)         # fix #4: no index priority
        for i in order:
            i = int(i)
            if self._done[i]:
                continue
            pos_before = self.agent_position(i)
            goal = self.agent_goal(i)
            d_before = manhattan(pos_before, goal)
            action = int(actions[i])
            self.grid.move(i, action)
            pos_after = self.agent_position(i)
            blocked[i] = action != 0 and pos_after == pos_before

            if pos_after == goal:                # fix #3: finish-and-vanish
                rewards[i] = self.R_GOAL
                terminated[i] = True
                self._done[i] = True
                self.grid.hide_agent(i)
                continue

            d_after = manhattan(pos_after, goal)
            rewards[i] = (self.R_STEP
                          + self.R_PROGRESS * (d_before - d_after)
                          + (self.R_BLOCKED if blocked[i] else 0.0))

        hit_limit = self.steps >= self.max_steps
        for i in range(na):
            if self._done[i]:
                continue
            self._sense(i)
            if hit_limit:
                truncated[i] = True
                self._done[i] = True

        obs = [self._obs(i) for i in range(na)]
        info = {"blocked": blocked,
                "positions": [self.agent_position(i) for i in range(na)]}
        return obs, rewards, terminated, truncated, info

    def all_done(self) -> bool:
        return all(self._done)
