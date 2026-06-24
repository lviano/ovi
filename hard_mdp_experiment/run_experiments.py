"""
run_experiments.py - Main entry point. Runs both experiments and saves plots.

Usage:
    python run_experiments.py

Results are saved to the results/ directory as PDF and PNG.

Experiment 1: Single MDP - Algorithm 2 vs offline value-based IL
  Shows that Algorithm 2 achieves near-zero suboptimality as tau_E grows,
  while offline IL is stuck at ~0.25 regardless of tau_E.

Experiment 2: Extended MDP - Algorithm 2 vs BC
  2a) Suboptimality vs tau_E (fixed N=100)
  2b) Suboptimality vs N     (fixed tau_E=50)
  Shows that BC suboptimality grows with N (BC lower bound from Thm 4.2),
  while Algorithm 2 remains near-zero.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from experiments import (
    run_experiment1,
    run_experiment2a,
    run_experiment2b,
    run_experiment_dagger,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, name: str) -> None:
    """Save an experiment results dict to results/<name>.npz."""
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in results.items()})
    print(f"  Data saved  -> {name}.npz")


def load_results(name: str) -> dict:
    """
    Load a previously saved results dict from results/<name>.npz.
    Returns None if the file does not exist.
    """
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    # 0-d arrays (scalars like N, tau_E, T_dagger) are unwrapped to Python scalars.
    return {k: (data[k].item() if data[k].ndim == 0 else data[k])
            for k in data.files}


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.dpi": 150,
    "lines.linewidth": 2.0,
    "lines.markersize": 9,
})

COLOR_OVI    = "#E63946"   # red    (OVI, online value-based)
COLOR_SPOIL  = "#3B7BC8"   # blue   (SPOIL, offline value-based)
COLOR_DAGGER = "#7B6EBF"   # purple (DAgger, online policy-based)


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def plot_experiment1(results: dict, save_prefix: str) -> None:
    """
    Plot suboptimality vs tau_E for Algorithm 2 and offline value-based IL.
    """
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

    tau_E = results["tau_E_values"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Algorithm 2 (interactive)
    ax.plot(tau_E, results["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(tau_E,
                    results["alg2_mean"] - results["alg2_se"],
                    results["alg2_mean"] + results["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)

    # Offline value-based IL
    ax.plot(tau_E, results["offline_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o", label=r"\textsf{SPOIL}")
    ax.fill_between(tau_E,
                    results["offline_mean"] - results["offline_se"],
                    results["offline_mean"] + results["offline_se"],
                    color=COLOR_SPOIL, alpha=0.2)

    # Theoretical lower bound for offline IL: 0.25
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.5,
               label="Offline IL lower bound (0.25)")

    ax.set_xlabel(r"Number of expert queries $\tau_E$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.02)
    leg = ax.legend()
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    leg.get_texts()[2].set_color("gray")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.{ext}"))
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved {save_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Experiment 2a
# ---------------------------------------------------------------------------

def plot_experiment2a(results: dict, save_prefix: str) -> None:
    """
    Plot suboptimality vs tau_E for Algorithm 2 and BC in extended MDP.
    """
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

    tau_E = results["tau_E_values"]
    N = results["N"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(tau_E, results["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(tau_E,
                    results["alg2_mean"] - results["alg2_se"],
                    results["alg2_mean"] + results["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)

    ax.plot(tau_E, results["bc_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o", label=r"\textsf{SPOIL} with action memorization")
    ax.fill_between(tau_E,
                    results["bc_mean"] - results["bc_se"],
                    results["bc_mean"] + results["bc_se"],
                    color=COLOR_SPOIL, alpha=0.2)

    ax.set_xlabel(r"Number of expert queries $\tau_E$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.005)
    leg = ax.legend()
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.{ext}"))
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved {save_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Experiment 2b
# ---------------------------------------------------------------------------

def plot_experiment2b(results: dict, save_prefix: str) -> None:
    """
    Plot suboptimality vs N for Algorithm 2 and BC in extended MDP.
    """
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

    N_values = results["N_values"]
    tau_E = results["tau_E"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(N_values, results["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(N_values,
                    results["alg2_mean"] - results["alg2_se"],
                    results["alg2_mean"] + results["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)

    ax.plot(N_values, results["bc_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o", label=r"\textsf{SPOIL} with action memorization")
    ax.fill_between(N_values,
                    results["bc_mean"] - results["bc_se"],
                    results["bc_mean"] + results["bc_se"],
                    color=COLOR_SPOIL, alpha=0.2)

    # Theoretical BC lower bound: N / (4 * (tau_E+1) * e)
    e = np.e
    bc_lb = N_values / (4.0 * (tau_E + 1) * e)
    ax.plot(N_values, bc_lb, color="gray", linestyle="--", linewidth=1.5,
            label=r"Offline IL Lower Bound $\frac{|\mathcal{X}|}{4(\tau_E+1)e}$")

    ax.set_xlabel(r"Number of states: $|\mathcal{X}|$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.005)
    leg = ax.legend()
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    leg.get_texts()[2].set_color("gray")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.{ext}"))
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved {save_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Experiment 3: \textsf{DAgger} vs ISPIL (fixed N=10)
# ---------------------------------------------------------------------------

def plot_experiment_dagger(results: dict, save_prefix: str) -> None:
    """
    Single-panel figure: ISPIL vs \textsf{DAgger}.
    Same color per algorithm; solid line = suboptimality,
    dashed line = avg TV over visited states D.
    Top x-axis shows the episode counter k (= tau_E in this setup).
    """
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "lines.linewidth": 4,
        "lines.markersize": 12,
        "font.size": 20,
        "axes.titlesize": 24,
    })

    tau_E = results["tau_E_values"]
    T = results["T_dagger"]

    # ----------------------------------------------------------------
    # Prepend the initial-policy point (tau_E = 0, uniform policy):
    #   suboptimality = 0.25  (exact: Unif at x_start and x_minus)
    #   trajectory TV = 1.0   (exact: 2 * 0.5 = 1.0)
    # Placed at x = 0.4 so it appears just left of tau_E=1 on log scale.
    # ----------------------------------------------------------------
    INIT_SUBOPT = 0.25
    INIT_TV     = 1.0
    X_INIT      = 0.4

    def _prepend(arr, val):
        return np.concatenate([[val], arr])

    xs              = _prepend(tau_E,                      X_INIT)
    ispil_mean      = _prepend(results["ispil_mean"],      INIT_SUBOPT)
    ispil_se        = _prepend(results["ispil_se"],        0.0)
    ispil_tv_mean   = _prepend(results["ispil_tv_mean"],   INIT_TV)
    ispil_tv_se     = _prepend(results["ispil_tv_se"],     0.0)
    dagger_mean     = _prepend(results["dagger_mean"],     INIT_SUBOPT)
    dagger_se       = _prepend(results["dagger_se"],       0.0)
    dagger_tv_mean  = _prepend(results["dagger_tv_mean"],  INIT_TV)
    dagger_tv_se    = _prepend(results["dagger_tv_se"],    0.0)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # ---- ISPIL (OVI) ----
    ax.plot(xs, ispil_mean,
            color=COLOR_OVI, ls="-", marker="o", zorder=3)
    ax.fill_between(xs, ispil_mean - ispil_se, ispil_mean + ispil_se,
                    color=COLOR_OVI, alpha=0.15)
    ax.plot(xs, ispil_tv_mean,
            color=COLOR_OVI, ls="--", marker="o", zorder=3)
    ax.fill_between(xs, ispil_tv_mean - ispil_tv_se, ispil_tv_mean + ispil_tv_se,
                    color=COLOR_OVI, alpha=0.10)

    # ---- \textsf{DAgger} ----
    ax.plot(xs, dagger_mean,
            color=COLOR_DAGGER, ls="-", marker="o", markerfacecolor="none", zorder=3)
    ax.fill_between(xs, dagger_mean - dagger_se, dagger_mean + dagger_se,
                    color=COLOR_DAGGER, alpha=0.15)
    ax.plot(xs, dagger_tv_mean,
            color=COLOR_DAGGER, ls="--", marker="o", markerfacecolor="none", zorder=3)
    ax.fill_between(xs, dagger_tv_mean - dagger_tv_se, dagger_tv_mean + dagger_tv_se,
                    color=COLOR_DAGGER, alpha=0.10)

    # ---- Custom legend — one square per algorithm + line style rows ----
    legend_handles = [
        mpatches.Patch(color=COLOR_OVI,    label=r"\textsf{OVI} (Ours)"),
        mpatches.Patch(color=COLOR_DAGGER, label=r"\textsf{DAgger}"),
        mlines.Line2D([], [], color="black", ls="-",  lw=2,
                      label=r"Return suboptimality"),
        mlines.Line2D([], [], color="black", ls="--", lw=2,
                      label=r"Total Variation suboptimality"),
    ]
    leg = ax.legend(handles=legend_handles, ncol=1, fontsize=20)
              #loc="upper center", bbox_to_anchor=(0.5, -0.18),
              #framealpha=0.92, handlelength=1.5, columnspacing=1.0)
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_DAGGER)

    # X-axis ticks: "0" label at X_INIT, then the actual tau_E values
    xtick_pos    = xs
    xtick_labels = [r"$0$"] + [str(int(v)) for v in tau_E]

    xmin, xmax = 0.25, 500
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xtick_pos)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(xtick_labels))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(r"Expert queries / iterations $k$")
    ax.set_ylabel("")
    ax.set_ylim(-0.02, 1.10)
    ax.grid(True, alpha=0.3, which="major")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved {save_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Hard MDP Experiments: Algorithm 2 vs BC / Offline IL")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Experiment 1: Single MDP
    # ------------------------------------------------------------------
    print("\n--- Experiment 1: Single MDP ---")
    results1 = load_results("exp1_data")
    if results1 is not None:
        print("  Loaded cached results from exp1_data.npz")
    else:
        print("  Running experiment  (tau_E in [1..100], K=500, n_trials=500) ...")
        tau_E_values_exp1 = np.arange(1, 101, 5)
        results1 = run_experiment1(
            tau_E_values=tau_E_values_exp1,
            K=500, eta=2.0, n_trials=500, seed=42,
        )
        save_results(results1, "exp1_data")

    print("\n  Summary (first/last tau_E):")
    print("    tau_E=%3d: Alg2=%.4f, Offline=%.4f" % (
        results1['tau_E_values'][0],
        results1['alg2_mean'][0], results1['offline_mean'][0]))
    print("    tau_E=%3d: Alg2=%.4f, Offline=%.4f" % (
        results1['tau_E_values'][-1],
        results1['alg2_mean'][-1], results1['offline_mean'][-1]))
    plot_experiment1(results1, "exp1_single_mdp")

    # ------------------------------------------------------------------
    # Experiment 2a: Extended MDP, vary tau_E
    # ------------------------------------------------------------------
    print("\n--- Experiment 2a: Extended MDP (fixed N=100, vary tau_E) ---")
    results2a = load_results("exp2a_data")
    if results2a is not None:
        print("  Loaded cached results from exp2a_data.npz")
    else:
        print("  Running experiment  (tau_E in [1,5,10,...,500], K=500, n_trials=10) ...")
        tau_E_values_exp2 = np.array([1, 5, 10, 20, 50, 100, 200, 500])
        results2a = run_experiment2a(
            tau_E_values=tau_E_values_exp2,
            N=100, K=500, eta=2.0, n_trials=10, seed=42,
        )
        save_results(results2a, "exp2a_data")

    print("\n  Summary:")
    for j, te in enumerate(results2a['tau_E_values']):
        print("    tau_E=%4d: Alg2=%.4f, BC=%.4f" % (
            te, results2a['alg2_mean'][j], results2a['bc_mean'][j]))
    plot_experiment2a(results2a, "exp2a_extended_vary_tau")

    # ------------------------------------------------------------------
    # Experiment 2b: Extended MDP, vary N
    # ------------------------------------------------------------------
    print("\n--- Experiment 2b: Extended MDP (fixed tau_E=500, vary N) ---")
    results2b = load_results("exp2b_data")
    if results2b is not None:
        print("  Loaded cached results from exp2b_data.npz")
    else:
        print("  Running experiment  (N in [1,5,...,500], K=500, n_trials=10) ...")
        N_values_exp2b = np.array([1, 5, 10, 20, 50, 100, 200, 500])
        results2b = run_experiment2b(
            N_values=N_values_exp2b,
            tau_E=500, K=500, eta=2.0, n_trials=10, seed=42,
        )
        save_results(results2b, "exp2b_data")

    print("\n  Summary:")
    for j, n in enumerate(results2b['N_values']):
        print("    N=%4d: Alg2=%.4f, BC=%.4f" % (
            n, results2b['alg2_mean'][j], results2b['bc_mean'][j]))
    plot_experiment2b(results2b, "exp2b_extended_vary_N")

    # ------------------------------------------------------------------
    # Experiment 3: \textsf{DAgger} vs ISPIL (fixed N=10)
    # ------------------------------------------------------------------
    print("\n--- Experiment 3: DAGGER vs ISPIL (fixed N=10, vary tau_E) ---")
    results3 = load_results("exp3_dagger_data_v7")
    if results3 is not None:
        print("  Loaded cached results from exp3_dagger_data_v7.npz")
    else:
        print("  Running experiment  (tau_E in [1..500], N=10, T=10, K=500, eta=10, last-iterate, n_trials=500) ...")
        tau_E_values_exp3 = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
        results3 = run_experiment_dagger(
            tau_E_values=tau_E_values_exp3,
            N=10, T_dagger=10, K=500, eta=10.0, n_trials=500, seed=42,
            use_last_iterate=True,
        )
        save_results(results3, "exp3_dagger_data_v7")

    print("\n  Summary (ISPIL eta=10, last iterate):")
    for j, te in enumerate(results3['tau_E_values']):
        print("    tau_E=%4d (k=%4d): ISPIL=%.4f  \textsf{DAgger}=%.4f | "
              "TV ISPIL=%.3f  \textsf{DAgger}=%.3f" % (
                  te, te,
                  results3['ispil_mean'][j], results3['dagger_mean'][j],
                  results3['ispil_tv_mean'][j], results3['dagger_tv_mean'][j],
              ))
    plot_experiment_dagger(results3, "exp3_dagger_vs_ispil_N10")

    # ------------------------------------------------------------------
    # Combined summary figure
    # ------------------------------------------------------------------
    print("\n--- Generating combined summary figure ---")
    _plot_combined_summary(results1, results2a, results2b)

    # ------------------------------------------------------------------
    # Combined 2b + DAgger figure
    # ------------------------------------------------------------------
    print("\n--- Generating combined 2b + DAgger figure ---")
    _plot_combined_2b_and_dagger(results2b, results3)

    print("\n" + "=" * 60)
    print(f"All results saved to: {RESULTS_DIR}/")
    print("=" * 60)


def _plot_combined_2b_and_dagger(results2b: dict, results3: dict) -> None:
    """
    Two-panel figure: left = exp2b (suboptimality vs N), right = exp3 (DAgger vs OVI).
    """
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 4,
        "lines.markersize": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # ----------------------------------------------------------------
    # Left panel: exp2b (suboptimality vs N)
    # ----------------------------------------------------------------
    ax = axes[0]
    N_values = results2b["N_values"]
    tau_E_fixed = results2b["tau_E"]

    ax.plot(N_values, results2b["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(N_values,
                    results2b["alg2_mean"] - results2b["alg2_se"],
                    results2b["alg2_mean"] + results2b["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)
    ax.plot(N_values, results2b["bc_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o",
            label=r"\textsf{SPOIL} with action memorization")
    ax.fill_between(N_values,
                    results2b["bc_mean"] - results2b["bc_se"],
                    results2b["bc_mean"] + results2b["bc_se"],
                    color=COLOR_SPOIL, alpha=0.2)
    bc_lb = N_values / (4.0 * (tau_E_fixed + 1) * np.e)
    ax.plot(N_values, bc_lb, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"Number of states: $|\mathcal{X}|$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.005)
    legend_handles_left = [
        mlines.Line2D([], [], color=COLOR_OVI,   linestyle="-",  linewidth=2, marker="o",
                      label=r"\textsf{OVI} (Ours)"),
        mlines.Line2D([], [], color=COLOR_SPOIL, linestyle="--", linewidth=2, marker="o",
                      label=r"\textsf{SPOIL} with action memorization"),
        mlines.Line2D([], [], color="gray",      linestyle="--", linewidth=1.5,
                      label=r"Offline IL Lower Bound $\frac{|\mathcal{X}|}{4(\tau_E+1)e}$"),
    ]
    leg = ax.legend(handles=legend_handles_left, fontsize=20, handlelength=3)
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    leg.get_texts()[2].set_color("gray")
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Right panel: exp3 (DAgger vs OVI)
    # ----------------------------------------------------------------
    ax = axes[1]
    tau_E = results3["tau_E_values"]

    INIT_SUBOPT = 0.25
    INIT_TV     = 1.0
    X_INIT      = 0.4

    def _prepend(arr, val):
        return np.concatenate([[val], arr])

    xs             = _prepend(tau_E,                     X_INIT)
    ispil_mean     = _prepend(results3["ispil_mean"],    INIT_SUBOPT)
    ispil_se       = _prepend(results3["ispil_se"],      0.0)
    ispil_tv_mean  = _prepend(results3["ispil_tv_mean"], INIT_TV)
    ispil_tv_se    = _prepend(results3["ispil_tv_se"],   0.0)
    dagger_mean    = _prepend(results3["dagger_mean"],   INIT_SUBOPT)
    dagger_se      = _prepend(results3["dagger_se"],     0.0)
    dagger_tv_mean = _prepend(results3["dagger_tv_mean"], INIT_TV)
    dagger_tv_se   = _prepend(results3["dagger_tv_se"],  0.0)

    ax.plot(xs, ispil_mean, color=COLOR_OVI, ls="-", marker="o", zorder=3)
    ax.fill_between(xs, ispil_mean - ispil_se, ispil_mean + ispil_se,
                    color=COLOR_OVI, alpha=0.15)
    ax.plot(xs, ispil_tv_mean, color=COLOR_OVI, ls="--", marker="o", zorder=3)
    ax.fill_between(xs, ispil_tv_mean - ispil_tv_se, ispil_tv_mean + ispil_tv_se,
                    color=COLOR_OVI, alpha=0.10)

    ax.plot(xs, dagger_mean, color=COLOR_DAGGER, ls="-", marker="o", markerfacecolor="none", zorder=3)
    ax.fill_between(xs, dagger_mean - dagger_se, dagger_mean + dagger_se,
                    color=COLOR_DAGGER, alpha=0.15)
    ax.plot(xs, dagger_tv_mean, color=COLOR_DAGGER, ls="--", marker="o", markerfacecolor="none", zorder=3)
    ax.fill_between(xs, dagger_tv_mean - dagger_tv_se, dagger_tv_mean + dagger_tv_se,
                    color=COLOR_DAGGER, alpha=0.10)

    legend_handles = [
        mpatches.Patch(color=COLOR_DAGGER, label=r"\textsf{DAgger}"),
        mpatches.Patch(color=COLOR_OVI,    label=r"\textsf{OVI} (Ours)"),
        mlines.Line2D([], [], color="black", ls="-",  lw=2,
                      label=r"$J^{\pi_{\textsf{E}}} - J^{\pi_{\mathrm{out}}}$"),
        mlines.Line2D([], [], color="black", ls="--", lw=2,
                      label=r"$\mathcal{D}_{\mathsf{TV}}(\mathbb{P}^{\pi_{\textsf{E}}}, \mathbb{P}^{\pi_{\mathrm{out}}})$"),
    ]
    leg = ax.legend(handles=legend_handles, ncol=1, fontsize=20)
    leg.get_texts()[0].set_color(COLOR_DAGGER)
    leg.get_texts()[1].set_color(COLOR_OVI)

    xtick_labels = [r"$0$"] + [str(int(v)) for v in tau_E]
    ax.set_xscale("log")
    ax.set_xlim(0.25, 500)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(xtick_labels))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(r"Expert queries / iterations $k$")
    ax.set_ylabel("")
    ax.set_ylim(-0.02, 1.10)
    ax.grid(True, alpha=0.3, which="major")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"combined_2b_and_dagger.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print("  Saved combined_2b_and_dagger.pdf / .png")


def _plot_combined_summary(results1: dict, results2a: dict,
                           results2b: dict) -> None:
    """
    Three-panel combined summary figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Experiment 1 ---
    ax = axes[0]
    tau_E = results1["tau_E_values"]
    ax.plot(tau_E, results1["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(tau_E,
                    results1["alg2_mean"] - results1["alg2_se"],
                    results1["alg2_mean"] + results1["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)
    ax.plot(tau_E, results1["offline_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o", label=r"\textsf{SPOIL}")
    ax.fill_between(tau_E,
                    results1["offline_mean"] - results1["offline_se"],
                    results1["offline_mean"] + results1["offline_se"],
                    color=COLOR_SPOIL, alpha=0.2)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.5,
               label="Lower bound")
    ax.set_xlabel("Expert queries $\\tau_E$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.02)
    leg = ax.legend(fontsize=9)
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    leg.get_texts()[2].set_color("gray")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Experiment 2a ---
    ax = axes[1]
    tau_E = results2a["tau_E_values"]
    ax.plot(tau_E, results2a["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(tau_E,
                    results2a["alg2_mean"] - results2a["alg2_se"],
                    results2a["alg2_mean"] + results2a["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)
    ax.plot(tau_E, results2a["bc_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o", label=r"\textsf{SPOIL} with tie-breaking")
    ax.fill_between(tau_E,
                    results2a["bc_mean"] - results2a["bc_se"],
                    results2a["bc_mean"] + results2a["bc_se"],
                    color=COLOR_SPOIL, alpha=0.2)
    ax.set_xlabel("Expert queries $\\tau_E$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.005)
    leg = ax.legend(fontsize=9)
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Experiment 2b ---
    ax = axes[2]
    N_vals = results2b["N_values"]
    tau_E_fixed = results2b["tau_E"]
    ax.plot(N_vals, results2b["alg2_mean"],
            color=COLOR_OVI, linestyle="-", marker="o", label=r"\textsf{OVI} (Ours)")
    ax.fill_between(N_vals,
                    results2b["alg2_mean"] - results2b["alg2_se"],
                    results2b["alg2_mean"] + results2b["alg2_se"],
                    color=COLOR_OVI, alpha=0.2)
    ax.plot(N_vals, results2b["bc_mean"],
            color=COLOR_SPOIL, linestyle="--", marker="o", label=r"\textsf{SPOIL} with tie-breaking")
    ax.fill_between(N_vals,
                    results2b["bc_mean"] - results2b["bc_se"],
                    results2b["bc_mean"] + results2b["bc_se"],
                    color=COLOR_SPOIL, alpha=0.2)
    bc_lb = N_vals / (4.0 * (tau_E_fixed + 1) * np.e)
    ax.plot(N_vals, bc_lb, color="gray", linestyle="--", linewidth=1.5,
            label=r"BC lower bound")
    ax.set_xlabel("Number of states $|\mathcal{X}|$")
    ax.set_ylabel("Suboptimality")
    ax.set_ylim(bottom=-0.005)
    leg = ax.legend(fontsize=9)
    leg.get_texts()[0].set_color(COLOR_OVI)
    leg.get_texts()[1].set_color(COLOR_SPOIL)
    leg.get_texts()[2].set_color("gray")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS_DIR, f"combined_summary.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  Saved combined_summary.pdf / .png")


if __name__ == "__main__":
    main()
