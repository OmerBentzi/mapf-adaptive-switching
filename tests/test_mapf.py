"""Unit tests for the MAPF project (run with ``python -m pytest tests -q``).

The suite uses small deterministic POGEMA instances (custom maps with
explicit agent/target positions) so every assertion checks exact,
hand-computed values.  ``test_observation_channels_are_aligned`` is the test
that would have caught the transposed-agents-channel bug in the first version
of this project.
"""

import numpy as np
import pytest
import torch

from mapf.env import MAPFEnv, manhattan
from mapf.model import ActorCritic
from mapf.planners import (AStarPolicy, DeadlockMonitor, RLPolicy,
                           SwitchingPolicy, a_star, escape_action)
from mapf.ppo import PPOTrainer, compute_gae

FIVE_BY_FIVE = """
.....
.#...
...#.
.....
.....
"""

OPEN_9 = "\n".join(["." * 9] * 9)


def make_env(**kwargs):
    defaults = dict(obs_radius=2, max_episode_steps=32)
    defaults.update(kwargs)
    return MAPFEnv(**defaults)


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #

def test_observation_channels_are_aligned():
    """Obstacles / agents / target channels must share the [dx, dy] layout."""
    env = make_env(map=FIVE_BY_FIVE, num_agents=2,
                   agents_xy=[[0, 0], [2, 2]], targets_xy=[[4, 4], [0, 4]])
    obs, _ = env.reset(seed=1)
    c = 2  # window center for obs_radius=2

    # map obstacle at (1,1): delta (+1,+1) from agent0 at (0,0)
    assert obs[0][0][c + 1, c + 1] == 1
    # other agent at (2,2): delta (+2,+2) -- the old code transposed this
    assert obs[0][2][c + 2, c + 2] == 1
    # self must be excluded from the agents channel
    assert obs[0][2][c, c] == 0
    # target (4,4) is 4 cells away -> clamped to the window border (+2,+2)
    target = obs[0][1]
    assert target.sum() == 1 and target[c + 2, c + 2] == 1


def test_visited_channel_tracks_own_path():
    env = make_env(map=OPEN_9, num_agents=1,
                   agents_xy=[[4, 4]], targets_xy=[[8, 8]])
    obs, _ = env.reset(seed=1)
    c = 2
    assert obs[0][3][c, c] == 1                  # start cell visited
    obs, *_ = env.step([2])                      # move down (+row)
    assert obs[0][3][c, c] == 1                  # current cell
    assert obs[0][3][c - 1, c] == 1              # previous cell, one row up


def test_planner_memory_accumulates_at_world_coordinates():
    """Memory must be a global map, not a superposition of local windows."""
    env = make_env(map=FIVE_BY_FIVE, num_agents=1,
                   agents_xy=[[0, 0]], targets_xy=[[4, 4]])
    env.reset(seed=1)
    r = env.config.obs_radius
    wall = (1 + r, 1 + r)                        # map (1,1) in padded coords
    assert env.known_obstacles(0)[wall] == 1
    for action in (2, 2, 4, 4):                  # walk away from the wall
        env.step([action])
    assert env.known_obstacles(0)[wall] == 1     # still remembered


# --------------------------------------------------------------------------- #
# rewards and goal semantics
# --------------------------------------------------------------------------- #

def test_reward_progress_and_blocked():
    env = make_env(map=OPEN_9, num_agents=1,
                   agents_xy=[[0, 0]], targets_xy=[[0, 4]])
    env.reset(seed=1)
    _, rewards, *_ = env.step([4])               # right: 1 cell closer
    assert rewards[0] == pytest.approx(env.R_STEP + env.R_PROGRESS)
    _, rewards, *_ = env.step([1])               # up into border padding
    assert rewards[0] == pytest.approx(env.R_STEP + env.R_BLOCKED)


def test_goal_gives_terminal_reward_and_agent_vanishes():
    env = make_env(map=OPEN_9, num_agents=2,
                   agents_xy=[[0, 0], [0, 2]], targets_xy=[[0, 1], [8, 8]])
    obs, _ = env.reset(seed=1)
    c = 2
    assert obs[1][2][c, c - 2] == 1              # agent1 sees agent0
    obs, rewards, terminated, _, _ = env.step([4, 0])
    assert rewards[0] == env.R_GOAL and terminated[0]
    assert not env.active(0)
    assert obs[1][2].sum() == 0                  # finished agent removed


def test_truncation_flags_at_step_limit():
    env = make_env(map=OPEN_9, num_agents=1, max_episode_steps=3,
                   agents_xy=[[0, 0]], targets_xy=[[8, 8]])
    env.reset(seed=1)
    truncated = [False]
    for _ in range(3):
        _, _, _, truncated, _ = env.step([0])
    assert truncated[0] and env.all_done()


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #

def test_astar_straight_line():
    grid = np.zeros((7, 7))
    path = a_star(grid, (0, 0), (0, 4))
    assert path is not None and len(path) == 5
    assert path[0] == (0, 0) and path[-1] == (0, 4)


def test_astar_detours_around_wall():
    grid = np.zeros((5, 7))
    grid[1, 1:6] = 1                             # horizontal wall
    path = a_star(grid, (0, 3), (2, 3))
    assert path is not None
    assert len(path) > manhattan((0, 3), (2, 3)) + 1
    assert all(grid[cell] == 0 for cell in path)


def test_astar_respects_blocked_but_not_blocked_goal():
    grid = np.zeros((3, 3))
    grid[0, 1] = grid[2, 1] = 1                  # only corridor: (1,1)
    assert a_star(grid, (1, 0), (1, 2), blocked={(1, 1)}) is None
    assert a_star(grid, (1, 0), (1, 1), blocked={(1, 1)}) is not None


def test_astar_policy_reaches_goal_alone():
    env = make_env(map=FIVE_BY_FIVE, num_agents=1,
                   agents_xy=[[0, 0]], targets_xy=[[4, 4]])
    obs, _ = env.reset(seed=1)
    policy = AStarPolicy(env)
    for _ in range(env.max_steps):
        obs, _, terminated, _, _ = env.step(policy.select_actions(env, obs))
        if terminated[0]:
            break
    assert terminated[0]


# --------------------------------------------------------------------------- #
# deadlock detection / escape
# --------------------------------------------------------------------------- #

def test_deadlock_monitor_frozen_and_oscillation():
    m = DeadlockMonitor(num_agents=1, window=4, escape_steps=3)
    assert [m.in_escape(0, (1, 1)) for _ in range(3)] == [False] * 3
    assert m.in_escape(0, (1, 1))                # 4 identical -> trigger
    assert m.in_escape(0, (1, 1)) and m.in_escape(0, (1, 1))  # countdown
    assert not m.in_escape(0, (1, 1))            # history was cleared

    m2 = DeadlockMonitor(num_agents=1, window=4, escape_steps=3)
    for pos in [(0, 0), (0, 1), (0, 0)]:
        assert not m2.in_escape(0, pos)
    assert m2.in_escape(0, (0, 1))               # A-B-A-B livelock


def test_escape_action_avoids_obstacles_and_agents():
    obs = np.zeros((4, 5, 5), dtype=np.float32)
    obs[0][1, 2] = 1                             # obstacle above
    obs[2][2, 3] = 1                             # agent to the right
    rng = np.random.default_rng(0)
    actions = {escape_action(obs, rng) for _ in range(50)}
    assert actions <= {2, 3}                     # only down / left are safe


# --------------------------------------------------------------------------- #
# switching controller
# --------------------------------------------------------------------------- #

def _switching_setup(agents_xy, targets_xy):
    env = make_env(map=OPEN_9, num_agents=2,
                   agents_xy=agents_xy, targets_xy=targets_xy)
    obs, _ = env.reset(seed=1)
    rl = RLPolicy(ActorCritic(), torch.device("cpu"))
    policy = SwitchingPolicy(AStarPolicy(env), rl, num_agents=2)
    return env, obs, policy


def test_switching_uses_astar_when_alone_and_rl_when_crowded():
    env, obs, policy = _switching_setup([[0, 0], [8, 8]], [[0, 8], [8, 0]])
    policy.select_actions(env, obs)              # nobody visible
    assert policy.mode_counts == {"astar": 2, "rl": 0, "escape": 0}

    env, obs, policy = _switching_setup([[0, 0], [0, 1]], [[8, 8], [8, 0]])
    policy.select_actions(env, obs)              # mutually visible
    assert policy.mode_counts["rl"] == 2


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #

def test_gae_matches_hand_computed_values():
    returns, adv = compute_gae([0.0, 1.0], [1.0, 2.0], gamma=0.5, lam=0.5)
    assert adv == pytest.approx([-0.25, -1.0])
    assert returns == pytest.approx([0.75, 1.0])


def test_gae_truncation_bootstraps_final_value():
    returns, adv = compute_gae([0.0], [1.0], gamma=0.5, lam=0.5,
                               bootstrap_value=4.0)
    assert adv == pytest.approx([1.0]) and returns == pytest.approx([2.0])


def test_ppo_update_runs_and_is_finite():
    torch.manual_seed(0)
    model = ActorCritic()
    trainer = PPOTrainer(model, torch.device("cpu"), min_batch=8,
                         minibatch=8, epochs=1)
    obs = np.random.rand(10, 4, 5, 5).astype(np.float32)
    trainer.add_trajectory(obs, [0] * 10, [-1.6] * 10, [0.0] * 10,
                           [0.1] * 10, truncated=False, bootstrap_value=0.0)
    assert trainer.ready()
    stats = trainer.update()
    assert all(np.isfinite(v) for v in stats.values())


def test_env_reset_is_deterministic_per_seed():
    a = MAPFEnv(num_agents=4, size=12, density=0.2, obs_radius=2,
                max_episode_steps=16)
    b = MAPFEnv(num_agents=4, size=12, density=0.2, obs_radius=2,
                max_episode_steps=16)
    obs_a, _ = a.reset(seed=7)
    obs_b, _ = b.reset(seed=7)
    for x, y in zip(obs_a, obs_b):
        assert np.array_equal(x, y)
