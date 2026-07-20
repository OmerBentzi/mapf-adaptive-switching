"""Evaluate all configurations on paired instances and produce the paper's
result tables, statistical tests, and figure.

Runs independently of training: it loads ``best_model.pt`` from ``CKPT_DIR``
(falling back to the latest ``checkpoint.pt``), so results can be produced
even if a training session was cut short.

Configurations (see :mod:`mapf.planners`):
    A* Only / A* + Escape / RL Only (greedy) / RL Only (sampled) /
    RL + Escape / Switching (A*/RL) + Escape / Switching (A*/random) + Escape

``RL Only (sampled)`` evaluates the PPO policy in the stochastic regime it
was trained in (greedy argmax of a stochastically-trained policy locks into
oscillations).  ``Switching (A*/random)`` is the ablation answering whether
the learned policy inside the switcher is load-bearing.

Experiments:
* **Main table** -- ISR + CSR + average makespan for agents in ``AGENTS``
  at ``OBS_RADIUS`` (default 5).  All methods see the *same* instances
  (paired seeds, and -- since the env dynamics RNG is derived from the
  episode seed -- identical move-order dynamics).
* **Statistics** -- per agent count, for the headline comparisons: exact
  McNemar tests on per-episode CSR, sign-flip permutation tests + bootstrap
  CIs on per-episode ISR, and makespan compared on *jointly-solved* episodes
  only (per-method makespan columns condition on that method's successes and
  must not be compared directly).
* **Radius sweep** -- zero-shot transfer to radii ``RADII`` with 8 agents.

Metrics:
* **ISR** -- individual success rate: fraction of agents that reached their
  goal within the step limit (averaged over episodes).
* **CSR** -- cooperative success rate: fraction of episodes where *all*
  agents reached their goals.
* **makespan** -- steps until the last agent finished (solved episodes only).

Outputs in ``CKPT_DIR``: ``results.json`` (aggregates + statistics),
``episodes.csv`` (per-episode records, incl. the radius sweep),
``results.png`` (ISR figure with 95% CI error bars).

Environment variables: ``CKPT_DIR``, ``EPISODES`` (default 100),
``AGENTS`` (default "4,8,16,24,32"), ``OBS_RADIUS`` (5), ``RADII`` ("3,5,7"),
``SWEEP_EPISODES`` (50), ``MAP_SIZE`` (32), ``DENSITY`` (0.3),
``MAX_STEPS`` (256), ``SEED_BASE`` (90000), ``SKIP_SWEEP`` ("0").
"""

from __future__ import annotations

import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mapf.env import MAPFEnv
from mapf.model import ActorCritic
from mapf.planners import (AStarPolicy, EscapePolicy, RandomPolicy, RLPolicy,
                           SwitchingPolicy)
from train import default_ckpt_dir

CKPT_DIR = os.environ.get("CKPT_DIR", default_ckpt_dir())
EPISODES = int(os.environ.get("EPISODES", 100))
AGENTS = [int(x) for x in os.environ.get("AGENTS", "4,8,16,24,32").split(",")]
OBS_RADIUS = int(os.environ.get("OBS_RADIUS", 5))
RADII = [int(x) for x in os.environ.get("RADII", "3,5,7").split(",")]
SWEEP_EPISODES = int(os.environ.get("SWEEP_EPISODES", 50))
MAP_SIZE = int(os.environ.get("MAP_SIZE", 32))
DENSITY = float(os.environ.get("DENSITY", 0.3))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 256))
SEED_BASE = int(os.environ.get("SEED_BASE", 90_000))
SKIP_SWEEP = os.environ.get("SKIP_SWEEP", "0") == "1"

SWITCHING = "Switching (A*/RL) + Escape"
SWITCHING_RANDOM = "Switching (A*/random) + Escape"
METHOD_ORDER = ["A* Only", "A* + Escape", "RL Only", "RL Only (sampled)",
                "RL + Escape", SWITCHING, SWITCHING_RANDOM]
SWEEP_METHODS = ["RL Only", "RL Only (sampled)", SWITCHING]

#: (method_a, method_b) pairs tested in the statistics section
COMPARISONS = [
    (SWITCHING, "A* + Escape"),
    ("A* + Escape", "A* Only"),
    ("RL + Escape", "RL Only (sampled)"),
    (SWITCHING, "RL + Escape"),
    (SWITCHING, SWITCHING_RANDOM),
]

N_RESAMPLES = 10_000


def load_model(device):
    best = os.path.join(CKPT_DIR, "best_model.pt")
    ckpt = os.path.join(CKPT_DIR, "checkpoint.pt")
    model = ActorCritic()
    if os.path.exists(best):
        model.load_state_dict(torch.load(best, map_location=device,
                                         weights_only=True))
        print(f"loaded {best}")
    elif os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["trainer"]["model"])
        print(f"best_model.pt not found -- loaded model from {ckpt}")
    else:
        raise SystemExit(f"no checkpoint found in {CKPT_DIR}; run train.py first")
    return model


def make_policies(model, device, num_agents, seed):
    """Fresh policy instances for one episode (fair paired comparison)."""
    rl = RLPolicy(model, device)
    switching_random = SwitchingPolicy(AStarPolicy(None, seed=seed),
                                       RandomPolicy(seed=seed),
                                       num_agents, seed=seed)
    switching_random.name = SWITCHING_RANDOM
    return {
        "A* Only": AStarPolicy(None, seed=seed),
        "A* + Escape": EscapePolicy(AStarPolicy(None, seed=seed),
                                    num_agents, seed=seed),
        "RL Only": rl,
        "RL Only (sampled)": RLPolicy(model, device, stochastic=True,
                                      seed=seed),
        "RL + Escape": EscapePolicy(rl, num_agents, seed=seed),
        SWITCHING: SwitchingPolicy(AStarPolicy(None, seed=seed), rl,
                                   num_agents, seed=seed),
        SWITCHING_RANDOM: switching_random,
    }


def run_episode(policy, env: MAPFEnv, seed: int):
    obs, _ = env.reset(seed=seed)
    # late-bind env into A*-based policies (they were built before the env)
    for p in [policy, getattr(policy, "base", None), getattr(policy, "astar", None)]:
        if p is not None and hasattr(p, "paths"):
            p.env = env
    na = env.num_agents
    success = [False] * na
    makespan = 0
    for step in range(env.max_steps):
        actions = policy.select_actions(env, obs)
        obs, _, terminated, _, _ = env.step(actions)
        for i in range(na):
            if terminated[i]:
                success[i] = True
                makespan = step + 1
        if env.all_done():
            break
    isr = sum(success) / na
    return {"seed": seed, "isr": isr, "csr": float(all(success)),
            "makespan": makespan if all(success) else None}


# --------------------------------------------------------------------------- #
# statistics (paired seeds -> paired tests; episode = unit of analysis)
# --------------------------------------------------------------------------- #


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(values):
    """Percentile bootstrap 95% CI for the mean (fixed seed: reproducible)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(N_RESAMPLES, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant pair counts.

    ``b`` = episodes solved by A but not B, ``c`` = solved by B but not A.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_permutation(diffs: np.ndarray, rng: np.random.Generator):
    """Sign-flip permutation test + percentile bootstrap CI for mean(diffs)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = len(diffs)
    observed = diffs.mean()
    if n == 0 or np.allclose(diffs, 0):
        return observed, (0.0, 0.0), 1.0
    signs = rng.choice([-1.0, 1.0], size=(N_RESAMPLES, n))
    null = (signs * diffs).mean(axis=1)
    # Phipson & Smyth (2010): count the identity permutation so p >= 1/(M+1)
    exceed = int((np.abs(null) >= abs(observed) - 1e-12).sum())
    p = (exceed + 1) / (N_RESAMPLES + 1)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(observed), (float(lo), float(hi)), p


def compare_methods(rec_a, rec_b, rng):
    """Paired statistics for two methods sharing the same episode seeds."""
    assert [e["seed"] for e in rec_a] == [e["seed"] for e in rec_b]
    a_solved = np.array([e["csr"] for e in rec_a], dtype=bool)
    b_solved = np.array([e["csr"] for e in rec_b], dtype=bool)
    disc_a = int((a_solved & ~b_solved).sum())   # A solved, B failed
    disc_b = int((~a_solved & b_solved).sum())
    isr_diff = np.array([ea["isr"] - eb["isr"]
                         for ea, eb in zip(rec_a, rec_b)])
    isr_mean, isr_ci, isr_p = paired_permutation(isr_diff, rng)
    joint = [(ea["makespan"], eb["makespan"])
             for ea, eb in zip(rec_a, rec_b)
             if ea["makespan"] is not None and eb["makespan"] is not None]
    if joint:
        span_diff = np.array([a - b for a, b in joint])
        span_mean, span_ci, span_p = paired_permutation(span_diff, rng)
    else:
        span_mean, span_ci, span_p = None, (None, None), None
    return {
        "csr_discordant": [disc_a, disc_b],
        "csr_mcnemar_p": mcnemar_exact(disc_a, disc_b),
        "isr_mean_diff": isr_mean,
        "isr_ci95": list(isr_ci),
        "isr_perm_p": isr_p,
        "makespan_joint_n": len(joint),
        "makespan_mean_diff": span_mean,
        "makespan_ci95": list(span_ci),
        "makespan_perm_p": span_p,
    }


def stats_for(records, methods, rng):
    """All pairwise COMPARISONS available among ``methods``, per agent count."""
    out = {}
    for a, b in COMPARISONS:
        if a not in methods or b not in methods:
            continue
        for na in records[a]:
            out.setdefault(str(na), {})[f"{a} vs {b}"] = \
                compare_methods(records[a][na], records[b][na], rng)
    return out


def print_stats(stats):
    print("\n########## PAIRED STATISTICS (per agent count) ##########")
    print("CSR: exact McNemar on discordant episodes; ISR: sign-flip "
          "permutation test\nover per-episode ISR (episode = unit, robust to "
          "within-episode correlation);\nmakespan: jointly-solved episodes "
          "only (negative diff = first method faster).")
    for na, comps in stats.items():
        print(f"\n--- {na} agents ---")
        for name, s in comps.items():
            d = s["csr_discordant"]
            span = ("n/a" if s["makespan_mean_diff"] is None else
                    f"{s['makespan_mean_diff']:+.1f} steps "
                    f"[{s['makespan_ci95'][0]:+.1f},{s['makespan_ci95'][1]:+.1f}] "
                    f"p={s['makespan_perm_p']:.3f} "
                    f"(n={s['makespan_joint_n']})")
            print(f"{name}\n"
                  f"    CSR discordant {d[0]}:{d[1]}  McNemar p={s['csr_mcnemar_p']:.4f}\n"
                  f"    ISR diff {s['isr_mean_diff']:+.3f} "
                  f"[{s['isr_ci95'][0]:+.3f},{s['isr_ci95'][1]:+.3f}] "
                  f"p={s['isr_perm_p']:.4f}\n"
                  f"    makespan (joint) {span}")


# --------------------------------------------------------------------------- #
# experiment driver
# --------------------------------------------------------------------------- #


def evaluate_grid(model, device, agent_counts, obs_radius, episodes,
                  methods=METHOD_ORDER):
    results = {m: {} for m in methods}
    records = {m: {} for m in methods}
    mode_usage = {}
    for na in agent_counts:
        env = MAPFEnv(num_agents=na, size=MAP_SIZE, density=DENSITY,
                      obs_radius=obs_radius, max_episode_steps=MAX_STEPS)
        per_method = {m: [] for m in methods}
        mode_usage[na] = {"astar": 0, "rl": 0, "escape": 0}
        for ep in range(episodes):
            seed = SEED_BASE + ep
            policies = make_policies(model, device, na, seed=seed)
            for m in methods:
                per_method[m].append(run_episode(policies[m], env, seed))
                if m == SWITCHING:   # mode usage of the real switcher only
                    for k, v in policies[m].mode_counts.items():
                        mode_usage[na][k] += v
        for m in methods:
            recs = per_method[m]
            records[m][na] = recs
            isr = float(np.mean([e["isr"] for e in recs]))
            csr_k = int(sum(e["csr"] for e in recs))
            spans = [e["makespan"] for e in recs if e["makespan"]]
            results[m][na] = {
                "isr": isr,
                "isr_ci95": list(bootstrap_mean_ci([e["isr"] for e in recs])),
                "csr": csr_k / len(recs),
                "csr_ci95": list(wilson_ci(csr_k, len(recs))),
                "makespan": float(np.mean(spans)) if spans else None,
                "solved_n": csr_k,
                "episodes": len(recs),
            }
        print(f"\n{'=' * 62}\n{na} agents | {MAP_SIZE}x{MAP_SIZE} | "
              f"r={obs_radius} | {episodes} paired episodes\n{'=' * 62}")
        print(f"{'method':<32}{'ISR':>8}{'CSR':>10}{'makespan*':>11}")
        for m in methods:
            r = results[m][na]
            span = f"{r['makespan']:.0f}" if r["makespan"] else "-"
            csr = f"{r['solved_n']}/{r['episodes']}"
            print(f"{m:<32}{r['isr']:>8.1%}{csr:>10}{span:>11}")
        print("* mean over that method's solved episodes only -- not "
              "comparable across methods\n  (see paired statistics)")
    return results, mode_usage, records


def plot_main(results, path):
    methods = [m for m in METHOD_ORDER if m in results and results[m]]
    agent_counts = sorted(next(iter(results.values())).keys())
    x = np.arange(len(agent_counts))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, m in enumerate(methods):
        vals = [results[m][na]["isr"] for na in agent_counts]
        errs = np.array([
            [results[m][na]["isr"] - results[m][na]["isr_ci95"][0]
             for na in agent_counts],
            [results[m][na]["isr_ci95"][1] - results[m][na]["isr"]
             for na in agent_counts]])
        offset = (k - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=m,
               yerr=errs, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x, [f"{na} agents" for na in agent_counts])
    ax.set_ylabel("Individual Success Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"MAPF success by method ({MAP_SIZE}x{MAP_SIZE}, "
                 f"density {DENSITY}, r={OBS_RADIUS}; "
                 "bars: bootstrap 95% CI)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nfigure saved to {path}")


def save_episode_csv(blocks, path):
    """``blocks`` is a list of ``(records, obs_radius, block_label)`` sets.

    The ``block`` column disambiguates main-experiment rows from sweep rows
    that share the same (method, num_agents, obs_radius) keys.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "method", "num_agents", "obs_radius", "seed",
                    "isr", "csr", "makespan"])
        for records, obs_radius, label in blocks:
            for m, by_na in records.items():
                for na, recs in by_na.items():
                    for e in recs:
                        w.writerow([label, m, na, obs_radius, e["seed"],
                                    e["isr"], e["csr"], e["makespan"]])
    print(f"per-episode records saved to {path}")


def main():
    device = torch.device(os.environ.get("DEVICE") or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"))
    model = load_model(device)
    rng = np.random.default_rng(0)           # resampling RNG (fixed: reproducible)

    print("\n########## MAIN EXPERIMENT ##########")
    results, mode_usage, records = evaluate_grid(model, device, AGENTS,
                                                 OBS_RADIUS, EPISODES)
    totals = {k: sum(mode_usage[na][k] for na in mode_usage)
              for k in ("astar", "rl", "escape")}
    grand = max(sum(totals.values()), 1)
    print("\nSwitching-policy mode usage (all agent counts): "
          + ", ".join(f"{k} {v / grand:.1%}" for k, v in totals.items()))
    for na in sorted(mode_usage):
        t = max(sum(mode_usage[na].values()), 1)
        print(f"  {na:>3} agents: " + ", ".join(
            f"{k} {v / t:.1%}" for k, v in mode_usage[na].items()))

    stats = stats_for(records, METHOD_ORDER, rng)
    print_stats(stats)

    sweep, csv_blocks = {}, [(records, OBS_RADIUS, "main")]
    if not SKIP_SWEEP:
        print("\n########## OBSERVATION-RADIUS SWEEP (zero-shot) ##########")
        for r in RADII:
            res, _, recs = evaluate_grid(model, device, [8], r,
                                         SWEEP_EPISODES,
                                         methods=SWEEP_METHODS)
            sweep[r] = {m: res[m][8] for m in res}
            csv_blocks.append((recs, r, "sweep"))

    out = {"main": results, "mode_usage": totals,
           "mode_usage_by_agents": mode_usage, "stats": stats,
           "radius_sweep": sweep,
           "config": {"map_size": MAP_SIZE, "density": DENSITY,
                      "episodes": EPISODES, "obs_radius": OBS_RADIUS,
                      "seed_base": SEED_BASE, "max_steps": MAX_STEPS,
                      "sweep_episodes": SWEEP_EPISODES}}
    json_path = os.path.join(CKPT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults saved to {json_path}")
    save_episode_csv(csv_blocks, os.path.join(CKPT_DIR, "episodes.csv"))
    plot_main(results, os.path.join(CKPT_DIR, "results.png"))


if __name__ == "__main__":
    main()
