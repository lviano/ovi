"""
extended_mdp.py - Extended MDP with N+1 copies for the BC lower bound.

From Theorem 4.2, the hard instance uses N+1 copies of the basic MDP
structure. All copies share the SAME underlying MDP type (M1 or M2, chosen
uniformly at random). Copy i (i=0,...,N) has its own state space
{x^i_start, x^i_minus, x^i_end} but identical reward/transition structure.

Initial distribution for evaluation:
  nu_0(x^0_start) = 1 - N/(tau_E + 1)
  nu_0(x^i_start) = 1/(tau_E + 1)   for i in {1, ..., N}

(x^0_start has the "main" probability mass; extra copies are rare.)

BC lower bound intuition:
  Expert takes '+' (action 0) at every x^i_start -> goes directly to x^i_end,
  never visiting x^i_minus. With tau_E trajectories and N rare starts,
  each rare start x^i_start is unseen with probability ~ 1/e. When unseen,
  BC plays Unif at x^i_start -> reaches x^i_minus with prob 0.5 -> plays Unif
  there -> suboptimality at that copy = 0.5 * 0.5 = 0.25.

Algorithm 2 advantage:
  Uses a SHARED Q-class {Q_M1_ext, Q_M2_ext} (valid because all copies have
  the same MDP type). At h=2, pools all x^i_minus samples to identify the
  correct Q-function and update the policy at x_minus. This identification
  improves with tau_E regardless of N.
"""

import numpy as np
from mdp import (
    XSTART, XMINUS, XEND, N_STATES, N_ACTIONS, H,
    M1, M2, get_expert_policy, get_q_class, transition,
    expert_value,
)
from algorithms import softmax_update, select_best_q, q_dot_pi


# ---------------------------------------------------------------------------
# Helpers for the extended MDP
# ---------------------------------------------------------------------------

def _nu0(N: int, tau_E: int) -> np.ndarray:
    """
    Initial distribution over copies 0, 1, ..., N.

    nu_0[0] = 1 - N/(tau_E + 1)
    nu_0[i] = 1/(tau_E + 1)   for i >= 1

    Clipped to [0, 1] in case tau_E < N.
    """
    nu = np.zeros(N + 1)
    rare = 1.0 / (tau_E + 1)
    nu[1:] = rare
    nu[0] = max(0.0, 1.0 - N * rare)
    # Renormalise (nu[0] might be 0 when N >= tau_E+1)
    total = nu.sum()
    if total > 0:
        nu /= total
    return nu


def _subopt_from_policy(pi_h1: np.ndarray, pi_h2: np.ndarray,
                        mdp_type: int, N: int, tau_E: int) -> float:
    """
    Compute expected suboptimality of a STATIONARY policy (pi_h1, pi_h2)
    in the extended MDP under nu_0.

    All copies share the same reward structure, so we only need to evaluate
    the policy once and weight by nu_0.

    Subopt for a single copy starting at x^i_start:
      = pi_h1[x_start, 1] * (- V_2(x_minus))
      where V_2(x_minus) = sum_a pi_h2[x_minus, a] * r[x_minus, a]

    Total subopt = sum_i nu_0[i] * subopt_per_copy
                 = 1.0 * subopt_per_copy    (since all copies identical)
    """
    from mdp import evaluate_policy, suboptimality
    subopt_per_copy = suboptimality(pi_h1, pi_h2, mdp_type)
    return float(subopt_per_copy)  # nu_0 weights sum to 1, all copies equal


# ---------------------------------------------------------------------------
# BC in the extended MDP
# ---------------------------------------------------------------------------

def evaluate_bc_extended(mdp_type: int, tau_E: int, N: int,
                         rng: np.random.Generator = None) -> float:
    """
    Simulate BC in the extended MDP and return the mean suboptimality.

    Procedure:
      1. Draw tau_E starting copies from nu_0 (multinomial).
      2. For each drawn copy, the expert takes action 0 at x^i_start.
         Record which copies are seen (x^i_start observed in dataset).
      3. BC policy:
           - pi_BC(0 | x^i_start) = 1   if copy i was seen
           - pi_BC(  | x^i_start) = Unif if copy i was NOT seen
           - pi_BC(  | x^i_minus) = Unif always (never seen by expert)
      4. Evaluate under nu_0:
           subopt = sum_i nu_0[i] * pi_BC(1|x^i_start) * 0.5
                                              ^-------^   ^--^
                                         (prob reach x^i_minus)
                                                         (Unif at x^i_minus
                                                          -> 0.5 prob bad action)

    The factor 0.5 at x^i_minus comes from playing Unif({0,1}) where exactly
    one action is bad (reward -1) and the other is neutral (reward 0):
      V_2(x_minus) = 0.5 * r(x_minus, 0) + 0.5 * r(x_minus, 1) = -0.5
      subopt contribution = pi_BC(1|x^i_start) * 0.5  [since -V_2 = 0.5]

    Parameters
    ----------
    mdp_type : int  - M1=0 or M2=1
    tau_E    : int  - number of expert trajectories
    N        : int  - number of extra copies (total copies = N+1)
    rng      : np.random.Generator or None

    Returns
    -------
    suboptimality : float >= 0
    """
    if rng is None:
        rng = np.random.default_rng()

    nu = _nu0(N, tau_E)  # shape [N+1]
    n_copies = N + 1

    # Sample which copies are visited by the tau_E expert trajectories
    visited_copies = rng.choice(n_copies, size=tau_E, p=nu)
    seen = np.zeros(n_copies, dtype=bool)
    seen[visited_copies] = True

    # BC policy at x_start for each copy:
    # seen -> action 0 with prob 1 (no prob of going to x_minus)
    # unseen -> Unif -> prob 0.5 of going to x_minus
    prob_reach_xminus = np.where(seen, 0.0, 0.5)

    # At x_minus, BC plays Unif -> V_2(x_minus) = -0.5 for both M1 and M2
    # Suboptimality at x_minus given reached = 0.5
    subopt_at_xminus = 0.5  # = -V_2(x_minus) with Unif policy

    # Expected suboptimality under nu_0
    subopt = np.sum(nu * prob_reach_xminus) * subopt_at_xminus
    return float(subopt)


# ---------------------------------------------------------------------------
# Algorithm 2 in the extended MDP
# ---------------------------------------------------------------------------

def evaluate_alg2_extended(mdp_type: int, tau_E: int, K: int, N: int,
                            eta: float = None,
                            rng: np.random.Generator = None,
                            return_tv: bool = False,
                            use_last_iterate: bool = False):
    """
    Run Algorithm 2 in the extended MDP and return the mean suboptimality.

    Key design choice: Algorithm 2 uses a SHARED Q-class for the extended MDP:
      Q_M1_ext[x_minus, 0] = 0,  Q_M1_ext[x_minus, 1] = -1   (same for all copies)
      Q_M2_ext[x_minus, 0] = -1, Q_M2_ext[x_minus, 1] = 0

    Since all N+1 copies share the same MDP type, pooling ALL x^i_minus samples
    at h=2 (from any copy) gives information about the correct Q-function.
    This is the key advantage over BC: Algorithm 2 can query x_minus states
    from its own rollout, while BC's expert never visits x_minus.

    Procedure:
      Stage h=1:
        All tau_E starts are at x^i_start (drawn from nu_0). Expert always
        takes action 0. Q-class gives 0 at all x_start states -> tie ->
        pi_out_h1 = Unif for ALL x^i_start.

      Stage h=2:
        Roll out pi_out_h1 from tau_E starting states (drawn from nu_0).
        Each copy: pi_out_h1 = Unif -> reaches x^i_minus with prob 0.5.
        Pool ALL x^i_minus samples across copies.
        Run K mirror-ascent steps using pooled data to find correct Q.
        pi_out_h2 shared across all copies (same MDP type).

      Evaluation under nu_0:
        subopt = sum_i nu_0[i] * pi_out_h1[x_start,1] * subopt_at_xminus(pi_out_h2)

    Parameters
    ----------
    mdp_type : int  - M1=0 or M2=1
    tau_E    : int  - number of expert queries per stage
    K        : int  - mirror-ascent iterations
    N        : int  - number of extra copies
    eta      : float or None - learning rate
    rng      : np.random.Generator or None

    Returns
    -------
    suboptimality : float >= 0
    """
    if rng is None:
        rng = np.random.default_rng()

    if eta is None:
        eta = np.sqrt(np.log(N_ACTIONS) / (K * H ** 2))

    q_cls = get_q_class()
    pi_E = get_expert_policy(mdp_type)
    nu = _nu0(N, tau_E)  # shape [N+1]
    n_copies = N + 1

    # -----------------------------------------------------------------------
    # Stage h=1
    # -----------------------------------------------------------------------
    # All tau_E samples are at x^i_start for various i (drawn from nu_0).
    # Expert always takes action 0 at any x^i_start.
    # Q-class: Q(x_start, *) = 0 for all Q in class -> always a tie.
    # Therefore pi_out_h1 = Unif({0,1}) for all copies (identical states).

    # No need to run mirror ascent explicitly: the result is Unif regardless.
    # We verify: select_best_q would return q_cls[0] (Q_M1) for any pi since
    # all scores are 0. The MW update exp(eta*0)=1 does not change Unif.
    pi_out_h1 = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
    # pi_out_h1[x_start, 1] = 0.5  (prob of going to x_minus)

    # -----------------------------------------------------------------------
    # Stage h=2
    # -----------------------------------------------------------------------
    # Sample tau_E starting copies from nu_0, then roll out pi_out_h1.
    start_copies = rng.choice(n_copies, size=tau_E, p=nu)

    # At x^i_start, pi_out_h1 is Unif -> action 0 (->x_end) or 1 (->x_minus)
    actions_h1 = rng.integers(0, N_ACTIONS, size=tau_E)
    # Deterministic transition: action 0 -> x_end, action 1 -> x_minus
    states_h2 = np.where(actions_h1 == 0, XEND, XMINUS)

    # Pool all x_minus samples (from any copy)
    xminus_mask = states_h2 == XMINUS
    states_h2_xminus = states_h2[xminus_mask]  # all = XMINUS

    # Expert action at x_minus (same for all copies since same MDP type)
    expert_action_at_xminus = int(pi_E[XMINUS].argmax())
    expert_actions_h2 = np.full(xminus_mask.sum(), expert_action_at_xminus,
                                dtype=int)

    if len(states_h2_xminus) == 0:
        # No x_minus samples at all: policy stays Unif
        pi_out_h2 = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
    else:
        # Same sign argument as in algorithm2: correct Q always wins.
        pi_unif_xminus = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
        scores = [
            len(states_h2_xminus) * (
                Q[XMINUS, expert_action_at_xminus]
                - Q[XMINUS].dot(pi_unif_xminus)
            )
            for Q in q_cls
        ]
        Q_fixed = q_cls[int(np.argmax(scores))]
        from algorithms import _vectorized_mw_average
        pi_out_h2 = _vectorized_mw_average(Q_fixed, K, eta,
                                           return_last=use_last_iterate)

    # -----------------------------------------------------------------------
    # Evaluate under nu_0
    # -----------------------------------------------------------------------
    from mdp import suboptimality
    subopt_per_copy = suboptimality(pi_out_h1, pi_out_h2, mdp_type)
    subopt = float(max(0.0, subopt_per_copy))

    if not return_tv:
        return subopt

    # Trajectory TV = sum_tau |P_E(tau) - P_L(tau)|
    # Expert always takes action 0 at x_start; both expert and learner play
    # Uniform at x_end.  ISPIL uses pi_out_h1[XSTART, 1] = 0.5 for all copies
    # -> TV = 2 * 0.5 * sum_i nu_0(i) = 1.0 always.
    traj_tv = 2.0 * float(pi_out_h1[XSTART, 1])  # = 1.0 for ISPIL
    return subopt, traj_tv


# ---------------------------------------------------------------------------
# DAgger in the extended MDP
# ---------------------------------------------------------------------------

def evaluate_dagger_extended(mdp_type: int, tau_E: int, T: int, N: int,
                              rng: np.random.Generator = None) -> float:
    """
    Run DAgger in the extended N+1 copy MDP and return the suboptimality.

    DAgger uses a PER-COPY tabular policy (no Q-class sharing across copies).
    T rounds, budget tau_E//T per round:

      Round t:
        1. Current per-copy policy pi_t derived via BC from aggregated dataset.
        2. Roll out pi_t for budget trajectories starting from nu_0.
        3. Query expert at every visited state (x^i_start and x^i_minus if
           reached); aggregate into dataset.
      Final policy: BC on full aggregated dataset.

    Why DAgger ~ BC here:
      At x^i_start the expert ALWAYS labels action 0, so once any trajectory
      starts at copy i the learned pi_start[i] -> action 0.  After that,
      x^i_minus is never reached again.  The "never visited" probability for
      rare copy i equals (1 - nu_i)^tau_E -- identical to BC -- so the
      expected suboptimality matches BC regardless of T.

    Parameters
    ----------
    mdp_type : int  - M1=0 or M2=1
    tau_E    : int  - total interaction budget (split equally across T rounds)
    T        : int  - number of DAgger rounds (T=1 is best for this instance)
    N        : int  - number of extra copies
    rng      : np.random.Generator or None

    Returns
    -------
    (suboptimality, avg_tv) : (float, float)
        avg_tv = (1/|D|) * sum_{x in D} sum_a |pi_E(a|x) - pi_final(a|x)|
        where D is the multiset of states visited during all T rounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    nu = _nu0(N, tau_E)
    n_copies = N + 1

    pi_E = get_expert_policy(mdp_type)
    expert_a_xminus = int(pi_E[XMINUS].argmax())
    bad_action_xminus = 1 - expert_a_xminus

    # Per-copy dataset: action counts at x^i_start and x^i_minus
    counts_start = np.zeros((n_copies, N_ACTIONS))
    counts_minus = np.zeros((n_copies, N_ACTIONS))

    # Cap rounds so we never run more trajectories than tau_E.
    # For tau_E < T: run tau_E rounds of 1 trajectory each (ceil-like spread).
    actual_T = min(T, max(1, tau_E))
    budget_per_round = max(1, tau_E // actual_T)

    for _ in range(actual_T):
        # Per-copy BC policy at x^i_start derived from current dataset
        total_start = counts_start.sum(axis=1)           # [n_copies]
        pi_starts = np.where(
            total_start[:, None] > 0,
            counts_start / np.maximum(total_start[:, None], 1.0),
            np.full((n_copies, N_ACTIONS), 1.0 / N_ACTIONS),
        )                                                  # [n_copies, N_ACTIONS]

        # Sample starting copies from nu_0
        start_copies = rng.choice(n_copies, size=budget_per_round, p=nu)

        # Expert at x^i_start always labels action 0
        np.add.at(counts_start[:, 0], start_copies, 1)

        # DAgger rolls out current policy (not expert) to determine transitions
        pi_for_starts = pi_starts[start_copies]            # [budget, N_ACTIONS]
        u = rng.random(budget_per_round)
        # action = 1 iff u >= p(action 0)  =>  transition to x^i_minus
        actions = (u >= pi_for_starts[:, 0]).astype(int)

        # Expert at x^i_minus labels the correct action for this MDP type
        xminus_copies = start_copies[actions == 1]
        np.add.at(counts_minus[:, expert_a_xminus], xminus_copies, 1)

    # Final per-copy BC policies
    total_start = counts_start.sum(axis=1)
    pi_starts_final = np.where(
        total_start[:, None] > 0,
        counts_start / np.maximum(total_start[:, None], 1.0),
        np.full((n_copies, N_ACTIONS), 1.0 / N_ACTIONS),
    )

    total_minus = counts_minus.sum(axis=1)
    pi_minus_final = np.where(
        total_minus[:, None] > 0,
        counts_minus / np.maximum(total_minus[:, None], 1.0),
        np.full((n_copies, N_ACTIONS), 1.0 / N_ACTIONS),
    )

    # subopt_i = pi_start_i(action 1) * pi_minus_i(bad_action)
    subopt_per_copy = pi_starts_final[:, 1] * pi_minus_final[:, bad_action_xminus]
    subopt = float(max(0.0, float(nu.dot(subopt_per_copy))))

    # Trajectory TV = sum_tau |P_E(tau) - P_L(tau)|
    # Expert always takes action 0 at x_start (never visits x_minus), and both
    # expert and learner play Uniform at x_end.  This yields:
    #   TV = 2 * sum_i nu_0(i) * pi_L(1 | x^i_start)
    traj_tv = 2.0 * float(nu.dot(pi_starts_final[:, 1]))

    return subopt, traj_tv
