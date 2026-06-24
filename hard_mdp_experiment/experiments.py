"""
experiments.py - Main experiment logic comparing Algorithm 2 vs BC.

Experiment 1: Single MDP - Algorithm 2 vs offline value-based IL
  - Vary tau_E from 1 to 100
  - Compare Algorithm 2 (interactive) vs "offline value-based IL"
    (offline IL that plays Unif at x_start since expert data has 0 signal
     there, picks the best Q from offline data, suboptimality ≈ 0.25 always)
  - Average over n_trials random MDP choices (M1 or M2 with equal prob)
  - Fixed K=50, eta=0.5

Experiment 2: Extended MDP - Algorithm 2 vs BC
  2a) Fix N=100, vary tau_E - suboptimality vs tau_E
  2b) Fix tau_E=50, vary N   - suboptimality vs N
  - Average over n_trials random MDP type choices
"""

import numpy as np
from tqdm import tqdm

from mdp import (
    XSTART, XMINUS, XEND, N_STATES, N_ACTIONS, H,
    M1, M2, suboptimality, get_q_class, get_expert_policy,
)
from algorithms import (
    algorithm2, collect_offline_expert_data, behavioural_cloning,
    select_best_q, softmax_update,
)
from extended_mdp import evaluate_bc_extended, evaluate_alg2_extended, evaluate_dagger_extended


# ---------------------------------------------------------------------------
# Experiment 1 helpers
# ---------------------------------------------------------------------------

def offline_value_based_il(mdp_type: int, tau_E: int,
                            rng: np.random.Generator) -> float:
    """
    Simulate "offline value-based IL" suboptimality in the single MDP.

    Offline IL proceeds as follows:
      1. Collect tau_E expert trajectories (all from x_start with action 0).
      2. At h=1: expert data only shows x_start with action 0, but the
         Q-class gives 0 values at x_start for BOTH actions -> the Q-based
         update cannot distinguish actions -> policy remains Unif at x_start.
      3. At h=2: expert data only shows x_end (never x_minus) -> no signal
         about which Q is correct -> pi_h2 remains Unif at x_minus.

    So offline value-based IL gets:
      subopt = pi_h1[x_start,1] * 0.5 = 0.5 * 0.5 = 0.25

    This is a CONSTANT lower bound independent of tau_E, because the expert
    offline trajectory never covers x_minus.

    NOTE: We simulate the exact offline IL procedure using the same
    mirror-ascent algorithm but with data only from the offline expert
    dataset (no interactive rollout for data collection).

    Parameters
    ----------
    mdp_type : int   - M1 or M2
    tau_E    : int   - number of offline expert trajectories
    rng      : np.random.Generator

    Returns
    -------
    suboptimality : float
    """
    # Offline data: tau_E trajectories from x_start
    # Stage h=1 samples: all at x_start with expert action 0
    states_h1 = np.full(tau_E, XSTART, dtype=int)
    expert_actions_h1 = np.zeros(tau_E, dtype=int)  # always action 0

    # Stage h=2 offline data: expert goes x_start->x_end->... so all at x_end
    # The offline expert dataset has NO x_minus samples
    states_h2_offline = np.full(tau_E, XEND, dtype=int)
    # Expert action at x_end (absorbing, arbitrary but say 0)
    expert_actions_h2_offline = np.zeros(tau_E, dtype=int)

    K = 50
    eta = 0.5
    q_cls = get_q_class()

    # h=1: run mirror ascent on offline data
    pi_k_h1 = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
    pi_sum_h1 = np.zeros((N_STATES, N_ACTIONS))
    for k in range(K):
        Q_k = select_best_q(q_cls, states_h1, expert_actions_h1, pi_k_h1)
        pi_sum_h1 += pi_k_h1
        pi_k_h1 = softmax_update(pi_k_h1, Q_k, eta)
    pi_out_h1 = pi_sum_h1 / K

    # h=2: run mirror ascent on offline data (only x_end states, no x_minus)
    pi_k_h2 = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
    pi_sum_h2 = np.zeros((N_STATES, N_ACTIONS))
    for k in range(K):
        Q_k = select_best_q(q_cls, states_h2_offline,
                            expert_actions_h2_offline, pi_k_h2)
        pi_sum_h2 += pi_k_h2
        pi_k_h2 = softmax_update(pi_k_h2, Q_k, eta)
    pi_out_h2 = pi_sum_h2 / K

    return suboptimality(pi_out_h1, pi_out_h2, mdp_type)


# ---------------------------------------------------------------------------
# Experiment 1: Single MDP
# ---------------------------------------------------------------------------

def run_experiment1(tau_E_values: np.ndarray,
                    K: int = 50,
                    eta: float = 0.5,
                    n_trials: int = 500,
                    seed: int = 42) -> dict:
    """
    Experiment 1: Single MDP - Algorithm 2 vs offline value-based IL.

    For each tau_E:
      - Run n_trials trials, each with a randomly chosen MDP type (M1 or M2).
      - Compute mean and std-error of suboptimality for both methods.

    Returns
    -------
    dict with keys:
      'tau_E_values'
      'alg2_mean', 'alg2_se'         (mean ± std-error)
      'offline_mean', 'offline_se'
    """
    rng = np.random.default_rng(seed)

    alg2_results = np.zeros((len(tau_E_values), n_trials))
    offline_results = np.zeros((len(tau_E_values), n_trials))

    for i, tau_E in enumerate(tqdm(tau_E_values, desc="Exp 1: tau_E")):
        for t in range(n_trials):
            mdp_type = rng.integers(0, 2)  # M1 or M2 uniformly

            # Algorithm 2 (interactive)
            pi_h1, pi_h2 = algorithm2(mdp_type, int(tau_E), K, eta=eta,
                                      rng=rng)
            alg2_results[i, t] = suboptimality(pi_h1, pi_h2, mdp_type)

            # Offline value-based IL
            offline_results[i, t] = offline_value_based_il(
                mdp_type, int(tau_E), rng
            )

    return {
        'tau_E_values': tau_E_values,
        'alg2_mean': alg2_results.mean(axis=1),
        'alg2_se': alg2_results.std(axis=1) / np.sqrt(n_trials),
        'offline_mean': offline_results.mean(axis=1),
        'offline_se': offline_results.std(axis=1) / np.sqrt(n_trials),
    }


# ---------------------------------------------------------------------------
# Experiment 2: Extended MDP
# ---------------------------------------------------------------------------

def run_experiment2a(tau_E_values: np.ndarray,
                     N: int = 100,
                     K: int = 50,
                     eta: float = 2.0,
                     n_trials: int = 200,
                     seed: int = 42) -> dict:
    """
    Experiment 2a: Extended MDP, fix N, vary tau_E.

    Returns suboptimality vs tau_E for Algorithm 2 and BC.
    """
    rng = np.random.default_rng(seed)

    alg2_results = np.zeros((len(tau_E_values), n_trials))
    bc_results = np.zeros((len(tau_E_values), n_trials))

    for i, tau_E in enumerate(tqdm(tau_E_values, desc="Exp 2a: tau_E")):
        for t in range(n_trials):
            mdp_type = rng.integers(0, 2)  # M1 or M2

            alg2_results[i, t] = evaluate_alg2_extended(
                mdp_type, int(tau_E), K, N, eta=eta, rng=rng
            )
            bc_results[i, t] = evaluate_bc_extended(
                mdp_type, int(tau_E), N, rng=rng
            )

    return {
        'tau_E_values': tau_E_values,
        'N': N,
        'alg2_mean': alg2_results.mean(axis=1),
        'alg2_se': alg2_results.std(axis=1) / np.sqrt(n_trials),
        'bc_mean': bc_results.mean(axis=1),
        'bc_se': bc_results.std(axis=1) / np.sqrt(n_trials),
    }


def run_experiment2b(N_values: np.ndarray,
                     tau_E: int = 50,
                     K: int = 50,
                     eta: float = 2.0,
                     n_trials: int = 200,
                     seed: int = 42) -> dict:
    """
    Experiment 2b: Extended MDP, fix tau_E, vary N.

    Returns suboptimality vs N for Algorithm 2 and BC.
    """
    rng = np.random.default_rng(seed)

    alg2_results = np.zeros((len(N_values), n_trials))
    bc_results = np.zeros((len(N_values), n_trials))

    for i, N in enumerate(tqdm(N_values, desc="Exp 2b: N")):
        for t in range(n_trials):
            mdp_type = rng.integers(0, 2)

            alg2_results[i, t] = evaluate_alg2_extended(
                mdp_type, tau_E, K, int(N), eta=eta, rng=rng
            )
            bc_results[i, t] = evaluate_bc_extended(
                mdp_type, tau_E, int(N), rng=rng
            )

    return {
        'N_values': N_values,
        'tau_E': tau_E,
        'alg2_mean': alg2_results.mean(axis=1),
        'alg2_se': alg2_results.std(axis=1) / np.sqrt(n_trials),
        'bc_mean': bc_results.mean(axis=1),
        'bc_se': bc_results.std(axis=1) / np.sqrt(n_trials),
    }


# ---------------------------------------------------------------------------
# Experiment 3: DAgger vs ISPIL in extended MDP with fixed N
# ---------------------------------------------------------------------------

def run_experiment_dagger(tau_E_values: np.ndarray,
                          N: int = 10,
                          T_dagger: int = 10,
                          K: int = 500,
                          eta: float = 2.0,
                          n_trials: int = 500,
                          seed: int = 42,
                          use_last_iterate: bool = False) -> dict:
    """
    Experiment 3: Fixed N, vary tau_E. Compare ISPIL vs DAgger (T=T_dagger).

    Tracks two metrics per algorithm:
      (a) Suboptimality under nu_0.
      (b) Average total variation over visited states D:
              (1/|D|) * sum_{x in D} sum_a |pi_E(a|x) - pi_learner(a|x)|
          where D is the multiset of states visited during interaction.
          - DAgger: labels every visited state with the expert action, so the
            final policy matches the expert on D -> avg TV = 0.
          - ISPIL: pi_out_h1 = Uniform at every x^i_start (TV = 1.0 there);
            avg TV stays bounded away from zero regardless of tau_E.

    Returns
    -------
    dict with keys:
      'tau_E_values', 'N', 'T_dagger'
      'ispil_mean', 'ispil_se'
      'ispil_tv_mean', 'ispil_tv_se'    (avg TV over visited states)
      'dagger_mean', 'dagger_se'
      'dagger_tv_mean', 'dagger_tv_se'  (avg TV over visited states)
    """
    rng = np.random.default_rng(seed)

    ispil_results  = np.zeros((len(tau_E_values), n_trials))
    ispil_tv       = np.zeros((len(tau_E_values), n_trials))
    dagger_results = np.zeros((len(tau_E_values), n_trials))
    dagger_tv      = np.zeros((len(tau_E_values), n_trials))

    for i, tau_E in enumerate(tqdm(tau_E_values, desc="Exp 3 (DAgger): tau_E")):
        for t in range(n_trials):
            mdp_type = rng.integers(0, 2)

            subopt_ispil, avg_tv_ispil = evaluate_alg2_extended(
                mdp_type, int(tau_E), K, N, eta=eta, rng=rng, return_tv=True,
                use_last_iterate=use_last_iterate,
            )
            ispil_results[i, t] = subopt_ispil
            ispil_tv[i, t]      = avg_tv_ispil

            subopt_dag, avg_tv_dag = evaluate_dagger_extended(
                mdp_type, int(tau_E), T_dagger, N, rng=rng
            )
            dagger_results[i, t] = subopt_dag
            dagger_tv[i, t]      = avg_tv_dag

    return {
        'tau_E_values': tau_E_values,
        'N': N,
        'T_dagger': T_dagger,
        'ispil_mean':    ispil_results.mean(axis=1),
        'ispil_se':      ispil_results.std(axis=1) / np.sqrt(n_trials),
        'ispil_tv_mean': ispil_tv.mean(axis=1),
        'ispil_tv_se':   ispil_tv.std(axis=1) / np.sqrt(n_trials),
        'dagger_mean':   dagger_results.mean(axis=1),
        'dagger_se':     dagger_results.std(axis=1) / np.sqrt(n_trials),
        'dagger_tv_mean': dagger_tv.mean(axis=1),
        'dagger_tv_se':   dagger_tv.std(axis=1) / np.sqrt(n_trials),
    }
