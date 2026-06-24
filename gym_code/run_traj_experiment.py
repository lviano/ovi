"""
run_traj_experiment.py
----------------------
Compare IL algorithms in the very limited data regime.

Sweeps over N_TRAJS expert trajectories. For each N the total step budget
is tau_E = N × MEAN_TRAJ_LEN.

Result keys encode (alg, mode, L, size):
  alg4_ma_L{L}_{size}   – Alg 4, mirror ascent, inner loop L
  alg4_sm_L{L}_{size}   – Alg 4, softmax, inner loop L
  alg5_ma_{size}         – Alg 5, mirror ascent
  alg5_sm_{size}         – Alg 5, softmax
  alg6_{size}            – Alg 6, BC
  alg7_{size}            – Alg 7, DAgger
"""

import os
import json
import argparse
import random
from typing import Optional
import numpy as np
import torch
import re as _re
import gymnasium as gym

def _safe_make(env_name: str, **kwargs):
    """gym.make wrapper that auto-upgrades deprecated gymnasium env versions."""
    try:
        return gym.make(env_name, **kwargs)
    except gym.error.DeprecatedEnv:
        m = _re.match(r'^(.*?)-v(\d+)$', env_name)
        if m:
            new_name = f"{m.group(1)}-v{int(m.group(2)) + 1}"
            return gym.make(new_name, **kwargs)
        raise

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_configs import get_env_config
from expert import (
    load_expert, train_expert, expert_exists, evaluate_policy_fn,
    collect_expert_data, save_offline_dataset, load_offline_dataset,
    offline_dataset_exists,
)
from algorithms import (
    run_alg4_interactive_il,
    run_alg5_spoil,
    run_alg6_bc,
    run_alg7_dagger,
)

# ===========================================================================
# Default configuration
# ===========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERT_SAVE_PATH = "expert_model"
N_EVAL_EPISODES  = 10

N_TRAJS_VALUES = [10, 30, 50, 100]

K_ALG5   = 100   # Alg 5 outer iterations (offline, data is fixed)

ETA      = 0.1
Q_LR     = 1e-3
PI_LR    = 1e-3
Q_STEPS  = 10
PI_STEPS = 10

SEEDS = range(50)

LARGE_HIDDEN           = [64, 64]
SMALL_HIDDEN           = [16, 16]
VERY_SMALL_HIDDEN      = [8, 8]
VERY_VERY_SMALL_HIDDEN = [4, 4]
TINY_HIDDEN            = [2, 2]

SIZE_HIDDEN = {
    "large":          LARGE_HIDDEN,
    "small":          SMALL_HIDDEN,
    "verysmall":      VERY_SMALL_HIDDEN,
    "veryverysmall":  VERY_VERY_SMALL_HIDDEN,
    "tiny":           TINY_HIDDEN,
}

SIZE_TITLES = {
    "large":         "Large network [64, 64]",
    "small":         "Small network [16, 16]",
    "verysmall":     "Very small network [8, 8]",
    "veryverysmall": "Very very small network [4, 4]",
    "tiny":          "Tiny network [2, 2]",
}

RESULTS_DIR = "results"

MODE_TAG = {"mirror ascent": "ma", "softmax": "sm"}
MODE_FULL = {"ma": "mirror ascent", "sm": "softmax"}


# ===========================================================================
# Utilities
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed: int, env_name: str) -> gym.Env:
    e = _safe_make(env_name)
    e.reset(seed=seed)
    return e


def measure_mean_traj_len(expert_model, env_name: str, n_episodes: int = 200,
                          seed: int = 0) -> float:
    env = _safe_make(env_name)
    env.reset(seed=seed)   # fixed seed → same mean_traj_len every run
    lengths = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        length = 0; done = False
        while not done:
            action, _ = expert_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            length += 1; done = terminated or truncated
        lengths.append(length)
    env.close()
    mean_len = float(np.mean(lengths))
    print(f"[measure_mean_traj_len] Mean expert episode length on {env_name}: {mean_len:.1f}")
    return mean_len


def make_config_key(alg: str, mode: str = None, L: int = None, size: str = "large") -> str:
    """Build the in-memory result-dict key for a given (alg, mode, L, size) configuration."""
    if alg == "alg4":
        return f"alg4_{MODE_TAG[mode]}_L{L}_{size}"
    elif alg == "alg5":
        return f"alg5_{MODE_TAG[mode]}_{size}"
    else:
        return f"{alg}_{size}"


def make_sa_key(alg: str, mode: str = None, L: int = None) -> str:
    """Build the size-agnostic file key (one JSON per alg/mode/L, both sizes inside)."""
    if alg == "alg4":
        return f"alg4_{MODE_TAG[mode]}_L{L}"
    elif alg == "alg5":
        return f"alg5_{MODE_TAG[mode]}"
    else:
        return alg


def build_configs(algs, modes, l_alg4, sizes):
    """Return list of config dicts to iterate over."""
    configs = []
    for alg in algs:
        if alg == "alg4":
            for mode in modes:
                for L in l_alg4:
                    for size in sizes:
                        configs.append({"alg": alg, "mode": mode, "L": L, "size": size})
        elif alg == "alg5":
            for mode in modes:
                for size in sizes:
                    configs.append({"alg": alg, "mode": mode, "L": None, "size": size})
        else:
            for size in sizes:
                configs.append({"alg": alg, "mode": None, "L": None, "size": size})
    return configs


# ===========================================================================
# Run one cell
# ===========================================================================

def run_one(cfg: dict, N: int, seed: int, expert_model, device,
            env_name: str, state_dim: int, action_dim: int,
            mean_traj_len: float, offline_states, offline_expert_actions,
            subsampling_freq: int, env_cfg: dict,
            deterministic: bool = False) -> float:
    alg  = cfg["alg"]
    mode = cfg["mode"]
    L    = cfg["L"]
    hidden = SIZE_HIDDEN[cfg["size"]]
    tau_E  = int(N * mean_traj_len)

    q_lr    = env_cfg["q_lr"]
    q_steps = env_cfg["q_steps"]

    if alg == "alg4":
        env = make_env(seed, env_name)
        K   = max(1, N)
        pi  = run_alg4_interactive_il(
            expert_model=expert_model, env=env, hidden_sizes=hidden,
            K=K, tau_E=tau_E, eta=ETA, q_lr=q_lr, pi_lr=PI_LR,
            q_steps=q_steps, pi_steps=PI_STEPS, device=device,
            offline_states=offline_states,
            offline_expert_actions=offline_expert_actions,
            mode=mode, L=L, state_dim=state_dim, action_dim=action_dim,
            subsampling_freq=subsampling_freq, deterministic=deterministic,
        )
        env.close()
    elif alg == "alg5":
        pi = run_alg5_spoil(
            offline_states=offline_states,
            offline_expert_actions=offline_expert_actions,
            hidden_sizes=hidden, K=env_cfg["k_alg5"], eta=ETA, q_lr=q_lr, pi_lr=PI_LR,
            q_steps=q_steps, pi_steps=PI_STEPS, device=device,
            mode=mode, state_dim=state_dim, action_dim=action_dim,
        )
    elif alg == "alg6":
        pi = run_alg6_bc(
            offline_states=offline_states,
            offline_expert_actions=offline_expert_actions,
            hidden_sizes=hidden, pi_lr=PI_LR, pi_steps=env_cfg["bc_steps"], device=device,
            state_dim=state_dim, action_dim=action_dim,
        )
    elif alg == "alg7":
        env = make_env(seed, env_name)
        K   = max(1, N)
        pi  = run_alg7_dagger(
            expert_model=expert_model, env=env, hidden_sizes=hidden,
            K=K, tau_E=tau_E, pi_lr=PI_LR, pi_steps=env_cfg["dagger_pi_steps"], device=device,
            state_dim=state_dim, action_dim=action_dim,
            subsampling_freq=subsampling_freq, deterministic=deterministic,
        )
        env.close()
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    eval_env = _safe_make(env_name)
    eval_env.reset(seed=seed)
    ret = evaluate_policy_fn(lambda s, p=pi: p.act(s), eval_env, N_EVAL_EPISODES)
    eval_env.close()
    return ret


# ===========================================================================
# Plotting
# ===========================================================================

def plot_results(results: dict, expert_return: float, save_dir: str,
                 env_name: str, n_trajs_values: list, configs: list,
                 freq_tag: str) -> None:
    # Assign a colour per unique (alg, mode, L) group
    groups = []
    seen = set()
    for cfg in configs:
        g = (cfg["alg"], cfg.get("mode"), cfg.get("L"))
        if g not in seen:
            groups.append(g)
            seen.add(g)

    cmap = plt.cm.tab10
    group_colour = {g: cmap(i % 10) for i, g in enumerate(groups)}

    def group_label(alg, mode, L):
        base = {"alg4": "Alg4", "alg5": "Alg5", "alg6": "BC", "alg7": "DAgger"}[alg]
        if mode is not None:
            base += f" {MODE_TAG[mode].upper()}"
        if L is not None:
            base += f" L={L}"
        return base

    unique_sizes = list(dict.fromkeys(cfg["size"] for cfg in configs))
    fig, axes = plt.subplots(1, len(unique_sizes),
                             figsize=(8 * len(unique_sizes), 6), sharey=True)
    if len(unique_sizes) == 1:
        axes = [axes]

    for ax, target_size in zip(axes, unique_sizes):
        for cfg in configs:
            alg, mode, L, size = cfg["alg"], cfg["mode"], cfg["L"], cfg["size"]
            if size != target_size:
                continue
            key = make_config_key(alg, mode, L, size)
            if key not in results:
                continue
            means, stds = [], []
            for N in n_trajs_values:
                vals = results[key].get(str(N), [])
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals)   if vals else 0.0)
            means = np.array(means); stds = np.array(stds)
            g = (alg, mode, L)
            ax.plot(n_trajs_values, means,
                    color=group_colour[g], linestyle="-",
                    marker="o", label=group_label(alg, mode, L), linewidth=1.8)
            ax.fill_between(n_trajs_values, means - stds, means + stds,
                            color=group_colour[g], alpha=0.10)

        ax.axhline(expert_return, color="black", linestyle=":", linewidth=1.8,
                   label=f"Expert ({expert_return:.1f})")
        ax.set_xlabel("Number of expert trajectories", fontsize=12)
        ax.set_title(SIZE_TITLES[target_size], fontsize=12)
        ax.set_xticks(n_trajs_values)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        ax.grid(alpha=0.35)

    axes[0].set_ylabel("Mean return (100 episodes)", fontsize=12)
    freq_label = f", sub={freq_tag[4:]}" if freq_tag else ""
    fig.suptitle(f"IL algorithms – limited data regime ({env_name}{freq_label})", fontsize=13)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    stem = f"traj_{env_name.replace('-', '_')}{freq_tag}"
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved to {path}")
    plt.close(fig)


# ===========================================================================
# Per-config result I/O  (one JSON file per config → parallel-safe)
# ===========================================================================

def _config_path(results_dir: str, env_tag: str, freq_tag: str, sa_key: str) -> str:
    return os.path.join(results_dir, f"{env_tag}{freq_tag}_{sa_key}.json")

def _expert_path(results_dir: str, env_tag: str, freq_tag: str) -> str:
    return os.path.join(results_dir, f"{env_tag}{freq_tag}_expert.json")

def _load_size_data(results_dir: str, env_tag: str, freq_tag: str,
                    sa_key: str, size: str) -> dict:
    """Return {N_str: [returns]} for the given size, or {} if not yet saved."""
    path = _config_path(results_dir, env_tag, freq_tag, sa_key)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get(size, {})
    return {}

def _save_size_data(results_dir: str, env_tag: str, freq_tag: str,
                    sa_key: str, size: str, size_data: dict,
                    inject_noise: float = 0.0) -> None:
    """Update the size-specific section of the config file, preserving other sizes."""
    path = _config_path(results_dir, env_tag, freq_tag, sa_key)
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[size] = size_data
    if inject_noise > 0.0:
        data["inject_noise"] = inject_noise
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _save_expert_return(results_dir: str, env_tag: str, freq_tag: str,
                        expert_return: float, inject_noise: float = 0.0) -> None:
    path = _expert_path(results_dir, env_tag, freq_tag)
    payload: dict = {"expert_return": expert_return}
    if inject_noise > 0.0:
        payload["inject_noise"] = inject_noise
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def _load_expert_return(results_dir: str, env_tag: str, freq_tag: str) -> Optional[float]:
    path = _expert_path(results_dir, env_tag, freq_tag)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["expert_return"]
    return None

def _assemble_results(results_dir: str, env_tag: str, freq_tag: str,
                      configs: list) -> dict:
    """Load all per-(alg/mode/L) files and reassemble the full results dict for plotting."""
    results = {}
    for cfg in configs:
        alg, mode, L, size = cfg["alg"], cfg["mode"], cfg["L"], cfg["size"]
        sa_key   = make_sa_key(alg, mode, L)
        full_key = make_config_key(alg, mode, L, size)
        results[full_key] = _load_size_data(results_dir, env_tag, freq_tag, sa_key, size)
    expert_ret = _load_expert_return(results_dir, env_tag, freq_tag)
    if expert_ret is not None:
        results["expert"] = expert_ret
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="IL trajectory experiment")
    parser.add_argument("--env", type=str, default="Acrobot-v1")
    parser.add_argument("--n_trajs", type=int, nargs="+", default=None)
    parser.add_argument("--seeds",   type=int, nargs="+", default=None)
    parser.add_argument("--subsampling_freq", type=int, default=1,
                        help="Record every k-th expert (state,action) pair (default 1)")
    parser.add_argument("--algs", nargs="+",
                        choices=["alg4", "alg5", "alg6", "alg7"],
                        default=["alg4", "alg5", "alg6", "alg7"])
    parser.add_argument("--modes", nargs="+", choices=["ma", "sm"],
                        default=["sm"],
                        help="Policy update modes for Alg4/5: ma=mirror ascent, sm=softmax")
    parser.add_argument("--l_alg4", type=int, nargs="+", default=[10],
                        help="Inner loop L values for Alg 4 (default [10])")
    parser.add_argument("--sizes", nargs="+",
                        choices=["large", "small", "verysmall", "veryverysmall", "tiny"],
                        default=["large", "small"],
                        help="Network sizes to run (default: large small)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Collect expert actions deterministically (greedy argmax) "
                             "rather than stochastically")
    parser.add_argument("--inject", action="store_true",
                        help="Inject noise: with probability 0.05 replace the expert "
                             "action with a uniformly random action during data collection")
    args = parser.parse_args()

    env_name         = args.env
    env_cfg          = get_env_config(env_name)
    n_trajs_values   = args.n_trajs if args.n_trajs is not None else N_TRAJS_VALUES
    seeds            = args.seeds   if args.seeds   is not None else SEEDS
    subsampling_freq = args.subsampling_freq
    modes            = [MODE_FULL[m] for m in args.modes]
    l_alg4           = args.l_alg4
    sizes            = args.sizes

    det_tag    = "_det" if args.deterministic else ""
    inject_tag = "_inject" if args.inject else ""
    freq_tag   = f"_sub{subsampling_freq}" if subsampling_freq > 1 else ""
    freq_tag   = freq_tag + det_tag + inject_tag
    env_tag  = env_name.replace("-", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = build_configs(args.algs, modes, l_alg4, sizes)

    print("=" * 70)
    print(f"  {env_name}  –  Limited Data Regime Experiment")
    print(f"  Device: {DEVICE}")
    print(f"  N_TRAJS: {n_trajs_values}  |  Seeds: {seeds}  |  subsampling_freq: {subsampling_freq}")
    print(f"  Algs: {args.algs}  |  Modes: {args.modes}  |  L_alg4: {l_alg4}")
    print("=" * 70)

    # Load or train expert
    if expert_exists(EXPERT_SAVE_PATH, env_name=env_name):
        print("\n[main] Expert model found – loading ...")
        expert_model = load_expert(EXPERT_SAVE_PATH, env_name=env_name)
    else:
        print("\n[main] Training expert ...")
        expert_model = train_expert(save_path=EXPERT_SAVE_PATH, env_name=env_name)

    mean_traj_len = measure_mean_traj_len(expert_model, env_name, n_episodes=200)

    _tmp = _safe_make(env_name)
    state_dim  = _tmp.observation_space.shape[0]
    action_dim = _tmp.action_space.n
    _tmp.close()
    print(f"[main] state_dim={state_dim}, action_dim={action_dim}")

    eval_env = _safe_make(env_name)
    eval_env.reset(seed=0)
    expert_return = evaluate_policy_fn(
        lambda s: int(expert_model.predict(s, deterministic=True)[0]),
        eval_env, N_EVAL_EPISODES
    )
    eval_env.close()
    print(f"[main] Expert mean return (clean): {expert_return:.2f}")

    if args.inject:
        noisy_eval_env = _safe_make(env_name)
        noisy_eval_env.reset(seed=0)
        expert_return = evaluate_policy_fn(
            lambda s: (int(expert_model.predict(s, deterministic=True)[0])
                       if np.random.random() >= 0.05
                       else noisy_eval_env.action_space.sample()),
            noisy_eval_env, N_EVAL_EPISODES
        )
        noisy_eval_env.close()
        print(f"[main] Noisy expert mean return (epsilon=0.05): {expert_return:.2f}")

    _save_expert_return(RESULTS_DIR, env_tag, freq_tag, expert_return,
                        inject_noise=0.05 if args.inject else 0.0)

    # Collect one large offline dataset (for max N) and slice it for each N,
    # so that smaller datasets are always strict prefixes of the larger ones.
    N_max      = max(n_trajs_values)
    tau_E_max  = int(N_max * mean_traj_len)
    max_offline_path = os.path.join(RESULTS_DIR,
                                    f"offline_dataset_{env_tag}_Nmax{freq_tag}.pt")
    if offline_dataset_exists(max_offline_path):
        full_offline_states, full_offline_actions = load_offline_dataset(max_offline_path)
        print(f"[main] Loaded nested offline dataset ({len(full_offline_states)} steps)")
    else:
        print(f"[main] Collecting nested offline dataset ({tau_E_max} steps, "
              f"subsampling_freq={subsampling_freq}) ...")
        set_seed(0)
        _data_env = _safe_make(env_name)
        _data_env.reset(seed=0)
        full_offline_states, full_offline_actions = collect_expert_data(
            expert_model, _data_env, tau_E_max, subsampling_freq=subsampling_freq,
            deterministic=args.deterministic,
            inject_noise=0.05 if args.inject else 0.0,
        )
        _data_env.close()
        save_offline_dataset(full_offline_states, full_offline_actions, max_offline_path)

    # Main sweep
    for N in n_trajs_values:
        tau_E = int(N * mean_traj_len)
        print(f"\n{'─' * 70}")
        print(f"  N={N} trajectories  |  tau_E={tau_E} steps")
        print(f"{'─' * 70}")

        # Slice a proportional prefix of the full dataset.
        # This avoids reasoning about subsampling arithmetic and guarantees
        # that N=N_max uses exactly the full dataset.
        n_offline              = round(len(full_offline_states) * N / N_max)
        offline_states         = full_offline_states[:n_offline]
        offline_expert_actions = full_offline_actions[:n_offline]

        for cfg in configs:
            alg, mode, L, size = cfg["alg"], cfg["mode"], cfg["L"], cfg["size"]
            sa_key   = make_sa_key(alg, mode, L)
            full_key = make_config_key(alg, mode, L, size)
            size_data = _load_size_data(RESULTS_DIR, env_tag, freq_tag, sa_key, size)

            existing = size_data.get(str(N), [])
            if len(existing) >= len(seeds):
                print(f"  [skip] {full_key} N={N} already done")
                continue

            seed_returns   = list(existing)
            start_seed_idx = len(existing)

            for seed in seeds[start_seed_idx:]:
                set_seed(seed)
                print(f"  [{full_key} N={N} seed={seed}] running ...", end=" ", flush=True)
                ret = run_one(cfg, N, seed, expert_model, DEVICE,
                              env_name, state_dim, action_dim, mean_traj_len,
                              offline_states, offline_expert_actions, subsampling_freq,
                              env_cfg=env_cfg, deterministic=args.deterministic)
                seed_returns.append(ret)
                print(f"return={ret:.1f}")

            size_data[str(N)] = seed_returns
            _save_size_data(RESULTS_DIR, env_tag, freq_tag, sa_key, size, size_data,
                            inject_noise=0.05 if args.inject else 0.0)

    # Assemble full results dict for plotting / summary
    results = _assemble_results(RESULTS_DIR, env_tag, freq_tag, configs)

    # Plot
    plot_results(results, expert_return, RESULTS_DIR, env_name,
                 n_trajs_values, configs, freq_tag)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY  (mean ± std over seeds)")
    print("=" * 70)
    print(f"  Expert: {expert_return:.1f}")
    header = f"  {'':30s}" + "".join(f"  N={N:2d}" for N in n_trajs_values)
    print(header)
    for cfg in configs:
        key = make_config_key(cfg["alg"], cfg["mode"], cfg["L"], cfg["size"])
        row = f"  {key:30s}"
        for N in n_trajs_values:
            vals = results[key].get(str(N), [])
            row += f"  {np.mean(vals):5.0f}" if vals else "      -"
        print(row)
    print("=" * 70)
    print("\n[main] Done.")


if __name__ == "__main__":
    main()
