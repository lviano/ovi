"""
replot.py
---------
Regenerate the combined IL comparison plots from the per-config JSON results
produced by run_traj_experiment.py.

File layout (new saving structure):
  results/{env_tag}{freq_tag}_{config_key}.json  →  {N_str: [returns]}
  results/{env_tag}{freq_tag}_expert.json        →  {"expert_return": float}

Usage examples:
  python replot.py --env Acrobot-v1
  python replot.py --env Acrobot-v1 --subsampling_freq 20
  python replot.py                        # auto-discovers all experiments
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def parse_sa_key(sa_key):
    """Parse a size-agnostic file key into (alg, mode_tag, L) or return None."""
    parts = sa_key.split("_")
    try:
        if parts[0] == "alg4":
            # alg4_{ma/sm}_L{L}
            return ("alg4", parts[1], int(parts[2][1:]))
        elif parts[0] == "alg5":
            # alg5_{ma/sm}
            return ("alg5", parts[1], None)
        elif parts[0] in ("alg6", "alg7"):
            return (parts[0], None, None)
    except Exception:
        pass
    return None


def make_full_key(alg, mode_tag, L, size):
    """Reconstruct the full in-memory key (alg/mode/L/size) for the results dict."""
    if alg == "alg4":
        return f"alg4_{mode_tag}_L{L}_{size}"
    elif alg == "alg5":
        return f"alg5_{mode_tag}_{size}"
    else:
        return f"{alg}_{size}"


def group_label(alg, mode_tag, L):
    base = {"alg4": r"$\pi$-first ISPIL (Ours)", "alg5": r"SPOIL", "alg6": r"BC", "alg7": r"DAgger"}[alg]
    # alg5 SM → "SPOIL" (drop mode tag)
    # alg4 SM L=50 → "SPIL (Ours)" (drop mode tag and L)
    if alg == "alg5" and mode_tag == "sm":
        return base
    if alg == "alg4" and mode_tag == "sm" and L == 50:
        return base
    if mode_tag is not None:
        base += f" {mode_tag.upper()}"
    if L is not None:
        base += f" L={L}"
    return base


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _prefix(env_tag, freq_tag):
    return f"{env_tag}{freq_tag}_"


def load_experiment(results_dir, env_tag, freq_tag):
    """
    Load all per-(alg/mode/L) JSON files for a given (env_tag, freq_tag) experiment.
    Each file has the format  {"large": {N_str: [returns]}, "small": {N_str: [returns]}}.
    Returns a combined results dict keyed by the full config key (with size), plus "expert".
    """
    prefix = _prefix(env_tag, freq_tag)
    results = {}

    # Expert return
    expert_path = os.path.join(results_dir, f"{prefix}expert.json")
    if os.path.exists(expert_path):
        with open(expert_path) as f:
            results["expert"] = json.load(f)["expert_return"]

    # Per-(alg/mode/L) files
    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        sa_key = fname[len(prefix):-len(".json")]
        if sa_key == "expert":
            continue
        parsed = parse_sa_key(sa_key)
        if parsed is None:
            continue
        alg, mode_tag, L = parsed
        with open(os.path.join(results_dir, fname)) as f:
            file_data = json.load(f)

        # Detect format:
        #   New: {"large": {N: [...]}, "small": {N: [...]}}
        #   Old (combined single-file): {"expert": float, "alg4_sm_L50_large": {N: [...]}, ...}
        # Old format always has keys starting with "alg" or an "expert" key; new format has only size names.
        top_keys = set(file_data.keys())
        if not any(k.startswith("alg") or k == "expert" for k in top_keys):
            # New per-(alg/mode/L) format
            for size, size_data in file_data.items():
                if not isinstance(size_data, dict):
                    continue
                results[make_full_key(alg, mode_tag, L, size)] = size_data
        else:
            # Old combined format — extract individual config entries directly
            for full_key, val in file_data.items():
                if full_key == "expert":
                    if results.get("expert") is None:
                        results["expert"] = val
                elif isinstance(val, dict):
                    results[full_key] = val

    return results


def discover_experiments(results_dir):
    """
    Scan results_dir for expert files and return a list of (env_tag, freq_tag)
    tuples, one per experiment.
    """
    experiments = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_expert.json"):
            continue
        stem = fname[:-len("_expert.json")]   # e.g. "Acrobot_v1" or "Acrobot_v1_sub20"
        # Split off freq_tag: it starts with "_sub"
        if "_sub" in stem:
            idx = stem.rfind("_sub")
            env_tag  = stem[:idx]
            freq_tag = stem[idx:]
        else:
            env_tag  = stem
            freq_tag = ""
        experiments.append((env_tag, freq_tag))
    return experiments


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(results, env_tag, freq_tag, n_trajs_values, results_dir, use_sem=False, max_seeds=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    })

    expert_return = results.get("expert", None)

    # Collect and parse all config keys (full keys with size, as assembled by load_experiment)
    parsed = {}
    for key in results:
        if key == "expert":
            continue
        # Full key format: alg4_{ma/sm}_L{L}_{size}, alg5_{ma/sm}_{size}, alg6_{size}, alg7_{size}
        parts = key.split("_")
        try:
            if parts[0] == "alg4":
                p = ("alg4", parts[1], int(parts[2][1:]), parts[3])
            elif parts[0] == "alg5":
                p = ("alg5", parts[1], None, parts[2])
            elif parts[0] in ("alg6", "alg7"):
                p = (parts[0], None, None, parts[1])
            else:
                continue
            parsed[key] = p
        except Exception:
            continue

    if not parsed:
        print(f"[replot] No config data found for {env_tag}{freq_tag}, skipping.")
        return

    # Build group ordering: unique (alg, mode_tag, L)
    groups = []
    seen = set()
    for key, (alg, mode_tag, L, size) in sorted(parsed.items()):
        g = (alg, mode_tag, L)
        if g not in seen:
            groups.append(g)
            seen.add(g)

    cmap = plt.cm.tab10
    group_colour = {g: cmap(i % 10) for i, g in enumerate(groups)}

    # Filter n_trajs_values to those actually present in the data
    available_N = set()
    for key in parsed:
        for n in results[key]:
            available_N.add(int(n))
    n_trajs_values = sorted(n for n in n_trajs_values if n in available_N)
    if not n_trajs_values:
        n_trajs_values = sorted(available_N)

    size_titles = {
        "large": r"$\pi_{\mathrm{expert}} \in \Pi_{\mathrm{expert}}$",
        "small": r"$\pi_{\mathrm{expert}} \notin \Pi_{\mathrm{expert}}$",
        "verysmallbc100": r"$\pi_{\mathrm{expert}} \notin \Pi_{\mathrm{expert}}$ (BC100)",
        "verysmall": r"$\pi_{\mathrm{expert}} \notin \Pi_{\mathrm{expert}}$ (very small)",
    }

    # Determine which sizes are present in the data, in a fixed display order;
    # unknown sizes are appended after the known ones.
    size_order = ["large", "small", "verysmallbc100", "verysmall"]
    all_sizes = {p[3] for p in parsed.values()}
    ordered_sizes = size_order + sorted(s for s in all_sizes if s not in size_order)
    present_sizes = [s for s in ordered_sizes if s in all_sizes]
    n_sizes = len(present_sizes)

    fig, axes = plt.subplots(1, n_sizes, figsize=(8 * n_sizes, 6), sharey=True)
    if n_sizes == 1:
        axes = [axes]

    legend_handles = []
    legend_labels  = []
    seen_legend    = set()

    for ax, target_size in zip(axes, present_sizes):
        for key, (alg, mode_tag, L, size) in sorted(parsed.items()):
            if size != target_size:
                continue
            data = results[key]
            means, stds = [], []
            for N in n_trajs_values:
                vals = data.get(str(N), [])
                if max_seeds is not None:
                    vals = vals[:max_seeds]
                means.append(np.mean(vals) if vals else np.nan)
                if vals:
                    s = np.std(vals)
                    stds.append(s / np.sqrt(len(vals)) if use_sem else s)
                else:
                    stds.append(0.0)
            means = np.array(means)
            stds  = np.array(stds)
            g = (alg, mode_tag, L)
            lbl = group_label(alg, mode_tag, L)
            line, = ax.plot(n_trajs_values, means,
                            color=group_colour[g], linestyle="-",
                            marker="o", linewidth=1.8)
            ax.fill_between(n_trajs_values, means - stds, means + stds,
                            color=group_colour[g], alpha=0.15)
            if g not in seen_legend:
                legend_handles.append(line)
                legend_labels.append(lbl)
                seen_legend.add(g)

        if expert_return is not None:
            hline = ax.axhline(expert_return, color="black", linestyle=":",
                               linewidth=2.0)
            if "expert" not in seen_legend:
                legend_handles.append(hline)
                legend_labels.append(r"Expert ($%.1f$)" % expert_return)
                seen_legend.add("expert")

        ax.set_xlabel(r"Number of expert trajectories")
        ax.set_title(size_titles.get(target_size, target_size))
        ax.set_xticks(n_trajs_values)
        ax.grid(alpha=0.35)

    axes[0].set_ylabel(r"Mean return (100 episodes)")

    fig.legend(legend_handles, legend_labels,
               loc="lower center",
               ncol=len(legend_handles),
               bbox_to_anchor=(0.5, -0.01),
               frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    stem = os.path.join(results_dir, f"{env_tag}{freq_tag}")
    for ext in ("pdf", "png"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[replot] Saved → {path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replot IL results from per-config JSONs")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name, e.g. Acrobot-v1 "
                             "(default: auto-discover all experiments in results/)")
    parser.add_argument("--subsampling_freq", type=int, default=1,
                        help="Subsampling frequency used during data collection (default 1)")
    parser.add_argument("--n_trajs", type=int, nargs="+", default=[1, 2, 3, 5, 7, 10, 30, 50, 100],
                        help="Trajectory counts to show on the x-axis")
    parser.add_argument("--divide", action="store_true",
                        help="Use standard error of the mean (std/sqrt(n)) instead of std for shaded bands")
    parser.add_argument("--max_seeds", type=int, default=None,
                        help="Maximum number of seeds to use per (N, config) entry (default: all)")
    args = parser.parse_args()

    if args.env is not None:
        freq_tag = f"_sub{args.subsampling_freq}" if args.subsampling_freq > 1 else ""
        experiments = [(args.env.replace("-", "_"), freq_tag)]
    else:
        experiments = discover_experiments(RESULTS_DIR)
        if not experiments:
            print(f"[replot] No experiment files found in {RESULTS_DIR}/")
            return

    for env_tag, freq_tag in experiments:
        print(f"\n[replot] Processing {env_tag}{freq_tag} ...")
        results = load_experiment(RESULTS_DIR, env_tag, freq_tag)
        if len(results) <= 1:   # only "expert" or empty
            print(f"[replot] No config data found, skipping.")
            continue
        plot_all(results, env_tag, freq_tag, args.n_trajs, RESULTS_DIR, use_sem=args.divide, max_seeds=args.max_seeds)


if __name__ == "__main__":
    main()
