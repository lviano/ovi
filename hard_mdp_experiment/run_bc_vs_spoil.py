"""
run_bc_vs_spoil.py - BC vs SPOIL in the offline Q^{pi_E}-realizable MDP.

Environment (single-stage, H=1):
  - N+1 states: x^0, x^1, ..., x^N  (same initial distribution as exp3)
  - 3 actions per state:
        a0 (features [0,1], reward 1)  <- expert's action
        a1 (features [0,1], reward 1)
        a2 (features [1,0], reward 0)
  - Reward weights theta* = [0, 1]:
        r(x, a0) = r(x, a1) = 1,   r(x, a2) = 0   (for all x)
  - Expert: pi_E(a0 | x) = 1 for all x

Key insight:
  Q^{pi_E}(x, a) = r(x, a).  Features depend ONLY on action, not state.
  Q-class: { Q_{[1,0]}, Q_{[0,1]} } where Q_theta(x, a) = theta^T phi(a).

      Q_{[0,1]}:  Q(x, a0) = Q(x, a1) = 1,  Q(x, a2) = 0   <- correct
      Q_{[1,0]}:  Q(x, a0) = Q(x, a1) = 0,  Q(x, a2) = 1   <- wrong

  SPOIL always selects Q_{[0,1]} (score = n * pi_k(a2) >= 0 always).
  Mirror ascent converges to pi_SPOIL = [1/2, 1/2, 0] for ALL states,
  regardless of which states were observed.

  Consequence:
    SPOIL: suboptimality = 0 everywhere (Q generalizes across states),
           BUT TV = |1-1/2|+|0-1/2|+|0-0| = 1 constant for all tau_E.
    BC:    seen states -> TV=0, subopt=0;
           unseen states -> Uniform(3) -> TV=4/3, subopt=1/3.
           Both TV and subopt decrease as tau_E grows.

This mirrors exp3 (DAgger vs OVI):
    SPOIL  <-> OVI     (Q-based, low subopt, high constant TV)
    BC     <-> DAgger  (policy-based, TV -> 0, subopt > 0 at unseen states)

Initial distribution (same as extended MDP / exp3):
  nu_0[0] = max(0, 1 - N/(tau_E+1)),  nu_0[i] = 1/(tau_E+1) for i >= 1
  (normalised to sum to 1)
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ACTIONS      = 3
EXPERT_ACTION  = 0                          # always a0

# phi(a0) = phi(a1) = [0,1],  phi(a2) = [1,0]
PHI = np.array([[0., 1.],
                [0., 1.],
                [1., 0.]])

THETA_STAR = np.array([0., 1.])             # reward weights
REWARDS    = PHI @ THETA_STAR               # [1, 1, 0]
PI_EXPERT  = np.array([1., 0., 0.])        # deterministic expert

# Q-class: wrong hypothesis first, correct second
Q_WRONG   = PHI @ np.array([1., 0.])       # [0, 0, 1]
Q_CORRECT = PHI @ np.array([0., 1.])       # [1, 1, 0]
Q_CLASS   = [Q_WRONG, Q_CORRECT]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Colors
COLOR_BC    = "#E63946"   # red
COLOR_SPOIL = "#3B7BC8"   # blue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nu0(N: int, tau_E: int) -> np.ndarray:
    """Initial distribution over N+1 copies (same formula as extended_mdp.py)."""
    nu        = np.zeros(N + 1)
    rare      = 1.0 / (tau_E + 1)
    nu[1:]    = rare
    nu[0]     = max(0.0, 1.0 - N * rare)
    total     = nu.sum()
    if total > 0:
        nu /= total
    return nu


def _softmax_update(pi: np.ndarray, q_vals: np.ndarray, eta: float) -> np.ndarray:
    """Multiplicative-weights update for a single-stage, single policy vector."""
    log_new  = np.log(np.clip(pi, 1e-300, None)) + eta * q_vals
    log_new -= log_new.max()
    pi_new   = np.exp(log_new)
    return pi_new / pi_new.sum()


# ---------------------------------------------------------------------------
# BC evaluation
# ---------------------------------------------------------------------------

def evaluate_bc(N: int, tau_E: int, rng: np.random.Generator) -> tuple:
    """
    Run BC and return (suboptimality, avg_tv) under nu_0.

    BC copies the expert at visited states, plays Uniform(3) elsewhere.

    suboptimality = E_{x~nu_0}[ r(x,pi_E) - r(x,pi_BC) ]
    avg_tv        = E_{x~nu_0}[ sum_a |pi_E(a|x) - pi_BC(a|x)| ]

    Analytic per-copy values:
      seen   -> subopt = 0,    TV = 0
      unseen -> subopt = 1/3,  TV = 4/3  (Uniform vs [1,0,0])
    """
    nu = nu0(N, tau_E)
    n_copies = N + 1

    # Which copies does the expert visit?
    visited       = rng.choice(n_copies, size=tau_E, p=nu)
    seen          = np.zeros(n_copies, dtype=bool)
    seen[visited] = True

    p_unseen = (~seen).astype(float)        # 1 if unseen, 0 if seen

    subopt = float(np.dot(nu, p_unseen)) * (1.0 / 3.0)
    avg_tv = float(np.dot(nu, p_unseen)) * (4.0 / 3.0)

    return subopt, avg_tv


# ---------------------------------------------------------------------------
# SPOIL evaluation
# ---------------------------------------------------------------------------

def evaluate_spoil(N: int, tau_E: int, K: int, eta: float,
                   rng: np.random.Generator) -> tuple:
    """
    Run SPOIL (offline mirror ascent with Q-class) and return
    (suboptimality, avg_tv) under nu_0.

    Because Q_theta(x,a) = theta^T phi(a) does NOT depend on x,
    the selected Q is the same for all states and the output policy
    pi_out is also the same for ALL states (seen and unseen).

    SPOIL always selects Q_CORRECT = Q_{[0,1]}:
      score(Q_{[0,1]}) = n * pi_k(a2) >= 0
      score(Q_{[1,0]}) = -n * pi_k(a2) <= 0

    Mirror ascent with Q_CORRECT fixed converges to [1/2, 1/2, 0].
    => subopt = 0,  TV = 1  (for any tau_E >= 1)
    """
    nu = nu0(N, tau_E)
    n_copies = N + 1

    # Collect expert data (we only need the count: n_samples)
    visited   = rng.choice(n_copies, size=tau_E, p=nu)
    n_samples = len(visited)   # = tau_E always

    if n_samples == 0:
        # No data: default to Uniform
        pi_out = np.ones(N_ACTIONS) / N_ACTIONS
    else:
        # Mirror ascent with Q-class (action-only, same for all states)
        pi_k   = np.ones(N_ACTIONS) / N_ACTIONS
        pi_sum = np.zeros(N_ACTIONS)

        for _ in range(K):
            # Select best Q
            best_score, best_q = -np.inf, Q_CLASS[0]
            for q_vals in Q_CLASS:
                score = n_samples * (q_vals[EXPERT_ACTION] - np.dot(pi_k, q_vals))
                if score > best_score:
                    best_score, best_q = score, q_vals

            pi_sum += pi_k
            pi_k    = _softmax_update(pi_k, best_q, eta)

        pi_out = pi_sum / K

    # Policy is the same for all states -> compute metrics once
    expert_reward = float(REWARDS[EXPERT_ACTION])           # = 1.0
    learner_reward = float(np.dot(pi_out, REWARDS))
    subopt = max(0.0, expert_reward - learner_reward)
    avg_tv = float(np.sum(np.abs(PI_EXPERT - pi_out)))

    return subopt, avg_tv


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(tau_E_values: np.ndarray,
                   N: int       = 10,
                   K: int       = 500,
                   eta: float   = 10.0,
                   n_trials: int = 500,
                   seed: int    = 42) -> dict:
    """
    For each tau_E, run n_trials Monte Carlo trials and record
    mean ± std-error of suboptimality and avg_tv for BC and SPOIL.
    """
    rng = np.random.default_rng(seed)

    bc_subopt    = np.zeros((len(tau_E_values), n_trials))
    bc_tv        = np.zeros((len(tau_E_values), n_trials))
    spoil_subopt = np.zeros((len(tau_E_values), n_trials))
    spoil_tv     = np.zeros((len(tau_E_values), n_trials))

    for i, tau_E in enumerate(tqdm(tau_E_values, desc="BC vs SPOIL: tau_E")):
        for t in range(n_trials):
            s, tv                = evaluate_bc(N, int(tau_E), rng)
            bc_subopt[i, t]      = s
            bc_tv[i, t]          = tv

            s, tv                = evaluate_spoil(N, int(tau_E), K, eta, rng)
            spoil_subopt[i, t]   = s
            spoil_tv[i, t]       = tv

    return {
        'tau_E_values':       tau_E_values,
        'N':                  N,
        'bc_subopt_mean':     bc_subopt.mean(axis=1),
        'bc_subopt_se':       bc_subopt.std(axis=1)    / np.sqrt(n_trials),
        'bc_tv_mean':         bc_tv.mean(axis=1),
        'bc_tv_se':           bc_tv.std(axis=1)        / np.sqrt(n_trials),
        'spoil_subopt_mean':  spoil_subopt.mean(axis=1),
        'spoil_subopt_se':    spoil_subopt.std(axis=1) / np.sqrt(n_trials),
        'spoil_tv_mean':      spoil_tv.mean(axis=1),
        'spoil_tv_se':        spoil_tv.std(axis=1)     / np.sqrt(n_trials),
    }


# ---------------------------------------------------------------------------
# Plotting  (mirrors plot_experiment_dagger in run_experiments.py)
# ---------------------------------------------------------------------------

def plot_bc_vs_spoil(results: dict, save_prefix: str) -> None:
    """
    Single-panel figure mirroring exp3 (DAgger vs OVI):
      solid lines  = suboptimality
      dashed lines = avg TV  ( sum_a |pi_E(a|x) - pi_out(a|x)| )
    X-axis: tau_E on log scale with a synthetic "0" point at X_INIT.
    """
    plt.rcParams.update({
        "text.usetex":           True,
        "font.family":           "serif",
        "text.latex.preamble":   r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.labelsize":        24,
        "xtick.labelsize":       20,
        "ytick.labelsize":       20,
        "legend.fontsize":       18,
        "lines.linewidth":       4,
        "lines.markersize":      12,
        "font.size":             20,
        "axes.titlesize":        24,
    })

    tau_E = results['tau_E_values']

    # --- Prepend the tau_E = 0 (Uniform) initial point ---
    # Uniform(3) vs pi_E=[1,0,0]: subopt = 1/3, TV = 4/3
    INIT_SUBOPT = 1.0 / 3.0
    INIT_TV     = 4.0 / 3.0
    X_INIT      = 0.4          # left of tau_E=1 on log scale

    def _prep(arr, val):
        return np.concatenate([[val], arr])

    xs              = _prep(tau_E,                         X_INIT)
    bc_sub          = _prep(results['bc_subopt_mean'],     INIT_SUBOPT)
    bc_sub_se       = _prep(results['bc_subopt_se'],       0.0)
    bc_tv           = _prep(results['bc_tv_mean'],         INIT_TV)
    bc_tv_se        = _prep(results['bc_tv_se'],           0.0)
    sp_sub          = _prep(results['spoil_subopt_mean'],  INIT_SUBOPT)
    sp_sub_se       = _prep(results['spoil_subopt_se'],    0.0)
    sp_tv           = _prep(results['spoil_tv_mean'],      INIT_TV)
    sp_tv_se        = _prep(results['spoil_tv_se'],        0.0)

    fig, ax = plt.subplots(figsize=(10, 7))

    # ---- SPOIL (red, solid = subopt, dashed = TV) ----
    ax.plot(xs, sp_sub,  color=COLOR_SPOIL, ls="-",  marker="o", zorder=3)
    ax.fill_between(xs, sp_sub - sp_sub_se, sp_sub + sp_sub_se,
                    color=COLOR_SPOIL, alpha=0.15)
    ax.plot(xs, sp_tv,   color=COLOR_SPOIL, ls="--", marker="o", zorder=3)
    ax.fill_between(xs, sp_tv - sp_tv_se, sp_tv + sp_tv_se,
                    color=COLOR_SPOIL, alpha=0.10)

    # ---- BC (purple, open markers) ----
    ax.plot(xs, bc_sub,  color=COLOR_BC, ls="-",  marker="o",
            markerfacecolor="none", zorder=3)
    ax.fill_between(xs, bc_sub - bc_sub_se, bc_sub + bc_sub_se,
                    color=COLOR_BC, alpha=0.15)
    ax.plot(xs, bc_tv,   color=COLOR_BC, ls="--", marker="o",
            markerfacecolor="none", zorder=3)
    ax.fill_between(xs, bc_tv - bc_tv_se, bc_tv + bc_tv_se,
                    color=COLOR_BC, alpha=0.10)

    # ---- Legend ----
    legend_handles = [
        mpatches.Patch(color=COLOR_SPOIL, label=r"\texttt{SPOIL} (Ours)"),
        mpatches.Patch(color=COLOR_BC,    label=r"\texttt{BC}"),
        mlines.Line2D([], [], color="black", ls="-",  lw=2,
                      label=r"$\rho^{\pi_{\texttt{E}}} - \rho^{\pi_{\mathrm{out}}}$"),
        mlines.Line2D([], [], color="black", ls="--", lw=2,
                      label=r"$\mathcal{D}_{\texttt{TV}}(\mathbb{P}^{\pi_{\texttt{E}}},\mathbb{P}^{\pi_{\mathrm{out}}})$"),
    ]
    leg = ax.legend(handles=legend_handles, ncol=1, fontsize=20)
    leg.get_texts()[0].set_color(COLOR_SPOIL)
    leg.get_texts()[1].set_color(COLOR_BC)

    # ---- X-axis: log scale, "0" label at X_INIT ----
    xtick_labels = [r"$0$"] + [str(int(v)) for v in tau_E]
    ax.set_xscale("log")
    ax.set_xlim(0.25, 500)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(xtick_labels))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(r"Expert trajectories $\tau_E$")
    ax.set_ylabel("")
    ax.set_ylim(-0.05, 1.55)
    ax.grid(True, alpha=0.3, which="major")

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, f"{save_prefix}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved {save_prefix}.pdf / .png")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(results: dict, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in results.items()})
    print(f"  Data saved -> {path}")


def load_results(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return {k: (data[k].item() if data[k].ndim == 0 else data[k])
            for k in data.files}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("BC vs SPOIL: Offline IL with Q^{pi_E} realizability")
    print("Environment: N+1 states, 3 actions, H=1")
    print("  a0: phi=[0,1], r=1 (expert)  |  a1: phi=[0,1], r=1  |  a2: phi=[1,0], r=0")
    print("  SPOIL -> policy [1/2,1/2,0] everywhere (subopt=0, TV=1 constant)")
    print("  BC    -> copies expert at seen states, Uniform elsewhere")
    print("=" * 60)

    tau_E_values = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    N = 10      # N+1 = 11 states (same as exp3)

    results = load_results("bc_vs_spoil_data")
    if results is not None:
        print("  Loaded cached results from bc_vs_spoil_data.npz")
    else:
        print(f"  Running (N={N}, tau_E={tau_E_values}, K=500, eta=10, n_trials=500) ...")
        results = run_experiment(
            tau_E_values=tau_E_values,
            N=N, K=500, eta=10.0, n_trials=500, seed=42,
        )
        save_results(results, "bc_vs_spoil_data")

    print("\n  Summary:")
    print(f"  {'tau_E':>6}  {'SPOIL subopt':>14}  {'SPOIL TV':>10}  "
          f"{'BC subopt':>10}  {'BC TV':>8}")
    for j, te in enumerate(results['tau_E_values']):
        print(f"  {int(te):>6}  "
              f"{results['spoil_subopt_mean'][j]:>14.4f}  "
              f"{results['spoil_tv_mean'][j]:>10.4f}  "
              f"{results['bc_subopt_mean'][j]:>10.4f}  "
              f"{results['bc_tv_mean'][j]:>8.4f}")

    plot_bc_vs_spoil(results, "bc_vs_spoil_N10")

    print("\n" + "=" * 60)
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
