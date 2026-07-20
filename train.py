"""Train the PPO agent (fully resumable -- safe against Colab disconnects).

All settings are environment variables so the script can be re-run across
several Colab sessions and simply continue where it stopped:

======================  =========================================  =========
variable                meaning                                    default
======================  =========================================  =========
``CKPT_DIR``            where checkpoints/logs are written          auto [*]
``MAX_HOURS``           wall-clock budget for *this* session        3.5
``TARGET_EPISODES``     stop after this many episodes (total)       8000
``AGENT_MIX``           agent counts sampled per episode            4,8,16
``MAP_SIZE``            grid size                                   32
``DENSITY``             obstacle density                            0.3
``OBS_RADIUS``          observation radius (window = 2r+1)          5
``MAX_STEPS``           episode step limit                          256
``SEED``                reproducible run: seeds weight init and
                        derives per-episode instance seeds from
                        (SEED, episode), stable across resumes       unset
======================  =========================================  =========

[*] ``/content/drive/MyDrive/RL_Project_Final`` when Google Drive is mounted,
otherwise ``./checkpoints``.

Checkpointing (fixes the old ``save()`` bug that overwrote the best weights
with the current ones every 30 minutes):

* ``checkpoint.pt``  -- full training state (model / optimizer / scheduler /
  episode counter / history / best score), written every 15 minutes and on
  exit; used to resume.
* ``best_model.pt``  -- model weights only, written **only** when the
  250-episode moving success rate improves.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from mapf.env import MAPFEnv
from mapf.model import ActorCritic
from mapf.ppo import PPOTrainer


def default_ckpt_dir() -> str:
    drive = "/content/drive/MyDrive/RL_Project_Final"
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(drive, exist_ok=True)
        return drive
    os.makedirs("checkpoints", exist_ok=True)
    return "checkpoints"


CKPT_DIR = os.environ.get("CKPT_DIR", default_ckpt_dir())
os.makedirs(CKPT_DIR, exist_ok=True)
MAX_HOURS = float(os.environ.get("MAX_HOURS", 3.5))
TARGET_EPISODES = int(os.environ.get("TARGET_EPISODES", 8000))
AGENT_MIX = [int(x) for x in os.environ.get("AGENT_MIX", "4,8,16").split(",")]
MAP_SIZE = int(os.environ.get("MAP_SIZE", 32))
DENSITY = float(os.environ.get("DENSITY", 0.3))
OBS_RADIUS = int(os.environ.get("OBS_RADIUS", 5))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 256))
SAVE_EVERY_SEC = 15 * 60
LOG_EVERY = 100
MOVING_WINDOW = 250

CKPT_PATH = os.path.join(CKPT_DIR, "checkpoint.pt")
BEST_PATH = os.path.join(CKPT_DIR, "best_model.pt")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.json")


def rollout_episode(env: MAPFEnv, trainer: PPOTrainer, rng: np.random.Generator):
    """Collect one episode with the current stochastic policy.

    Per-agent trajectories are pushed into the trainer's buffer; truncated
    trajectories bootstrap from the value of their final observation.
    Returns the episode's individual success rate (ISR).
    """
    model, device = trainer.model, trainer.device
    num_agents = int(rng.choice(AGENT_MIX))
    env.grid_kwargs["num_agents"] = num_agents
    # sample training seeds above the evaluation range (evaluate.py uses
    # SEED_BASE=90000..90999) so no eval instance is ever seen in training
    obs, _ = env.reset(seed=int(rng.integers(100_000, 1_000_000)))

    traj = [{"obs": [], "act": [], "logp": [], "val": [], "rew": []}
            for _ in range(num_agents)]
    success = [False] * num_agents
    was_truncated = [False] * num_agents
    final_obs = [None] * num_agents

    while not env.all_done():
        active = [i for i in range(num_agents) if env.active(i)]
        batch = torch.as_tensor(np.stack([obs[i] for i in active]),
                                dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, values = model(batch)
            dist = torch.distributions.Categorical(logits=logits)
            sampled = dist.sample()
            log_probs = dist.log_prob(sampled)
        # one device->host transfer per step; per-element int()/float() forces
        # a full GPU sync each call on MPS (measured 4x rollout slowdown)
        sampled = sampled.cpu().tolist()
        log_probs = log_probs.cpu().tolist()
        values = values.cpu().tolist()

        actions = [0] * num_agents
        for j, i in enumerate(active):
            actions[i] = sampled[j]
            traj[i]["obs"].append(obs[i])
            traj[i]["act"].append(actions[i])
            traj[i]["logp"].append(log_probs[j])
            traj[i]["val"].append(values[j])

        obs, rewards, terminated, truncated, _ = env.step(actions)
        for i in active:
            traj[i]["rew"].append(rewards[i])
            if terminated[i]:
                success[i] = True
            if truncated[i]:
                was_truncated[i] = True
                final_obs[i] = obs[i]

    # bootstrap values for truncated agents (fix: was previously treated as 0)
    trunc_idx = [i for i in range(num_agents) if was_truncated[i]]
    boot = {}
    if trunc_idx:
        batch = torch.as_tensor(np.stack([final_obs[i] for i in trunc_idx]),
                                dtype=torch.float32, device=device)
        with torch.no_grad():
            _, values = model(batch)
        boot = {i: float(v) for i, v in zip(trunc_idx, values)}

    for i in range(num_agents):
        trainer.add_trajectory(traj[i]["obs"], traj[i]["act"], traj[i]["logp"],
                               traj[i]["val"], traj[i]["rew"],
                               truncated=was_truncated[i],
                               bootstrap_value=boot.get(i, 0.0))
    return sum(success) / num_agents


def main():
    # seed BEFORE building the model so weight init is reproducible too;
    # per-episode instance randomness is derived from (SEED, episode) in the
    # loop below, so resumed sessions continue the same instance sequence
    # instead of replaying it from the start
    base_seed = (int(os.environ["SEED"])
                 if os.environ.get("SEED") is not None else None)
    if base_seed is not None:
        torch.manual_seed(base_seed)

    device = torch.device(os.environ.get("DEVICE") or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={device} | ckpt_dir={CKPT_DIR}")
    print(f"agents={AGENT_MIX} map={MAP_SIZE}x{MAP_SIZE} density={DENSITY} "
          f"r={OBS_RADIUS} | session budget {MAX_HOURS}h, "
          f"target {TARGET_EPISODES} episodes"
          + (f" | SEED={base_seed}" if base_seed is not None else ""))

    model = ActorCritic()
    trainer = PPOTrainer(model, device)
    episode, best, history = 0, 0.0, []

    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        trainer.load_state_dict(state["trainer"])
        episode = state["episode"]
        best = state["best"]
        history = state["history"]
        print(f"resumed from episode {episode} (best={best:.1%})")

    env = MAPFEnv(num_agents=AGENT_MIX[0], size=MAP_SIZE, density=DENSITY,
                  obs_radius=OBS_RADIUS, max_episode_steps=MAX_STEPS)
    rng = np.random.default_rng()            # used only when SEED is unset

    def save_checkpoint():
        # write-then-rename so a kill mid-write can't corrupt the checkpoint
        torch.save({"trainer": trainer.state_dict(), "episode": episode,
                    "best": best, "history": history[-20_000:]},
                   CKPT_PATH + ".tmp")
        os.replace(CKPT_PATH + ".tmp", CKPT_PATH)
        with open(LOG_PATH + ".tmp", "w") as f:
            json.dump({"episode": episode, "best": best,
                       "history": history[-20_000:]}, f)
        os.replace(LOG_PATH + ".tmp", LOG_PATH)

    start = time.time()
    last_save = start
    last_log = start
    stats = {}
    try:
        while episode < TARGET_EPISODES:
            if (time.time() - start) / 3600 >= MAX_HOURS:
                print("session time budget reached -- run this cell again "
                      "to continue training")
                break
            ep_rng = (np.random.default_rng([base_seed, episode])
                      if base_seed is not None else rng)
            history.append(rollout_episode(env, trainer, ep_rng))
            episode += 1
            if trainer.ready():
                stats = trainer.update()

            if episode % LOG_EVERY == 0:
                window = history[-MOVING_WINDOW:]
                moving = float(np.mean(window))
                eps_per_min = LOG_EVERY / max((time.time() - last_log) / 60, 1e-9)
                last_log = time.time()
                lr = trainer.optimizer.param_groups[0]["lr"]
                print(f"ep {episode:6d} | ISR(ma{MOVING_WINDOW}) {moving:5.1%}"
                      f" | {eps_per_min:5.1f} ep/min | lr {lr:.1e}"
                      f" | pi {stats.get('policy_loss', 0):+.3f}"
                      f" | V {stats.get('value_loss', 0):.3f}"
                      f" | H {stats.get('entropy', 0):.2f}")
                if len(history) >= MOVING_WINDOW and moving > best:
                    best = moving
                    torch.save(model.state_dict(), BEST_PATH + ".tmp")
                    os.replace(BEST_PATH + ".tmp", BEST_PATH)
                    print(f"  -> new best {best:.1%} (saved best_model.pt)")

            if time.time() - last_save > SAVE_EVERY_SEC:
                save_checkpoint()
                last_save = time.time()
                print(f"  [checkpoint saved @ ep {episode}]")
    except KeyboardInterrupt:
        print("interrupted -- saving checkpoint")

    save_checkpoint()
    hours = (time.time() - start) / 3600
    print(f"done: {episode} episodes total, best ISR {best:.1%}, "
          f"this session {hours:.2f}h")


if __name__ == "__main__":
    main()
