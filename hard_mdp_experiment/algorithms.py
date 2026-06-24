"""
algorithms.py - Implementations of Algorithm 2 and Behavioural Cloning (BC).

Algorithm 2 (Algorithm 2 from the paper):
  Interactive finite-horizon IL under Q^{pi_E} realizability.
  Runs online mirror ascent (multiplicative weights) at each stage h,
  using expert labels collected by rolling out the current policy.

Behavioural Cloning (BC):
  Offline IL that clones the expert from a pre-collected dataset.
  Copies the expert action distribution where data is seen; plays
  Uniform at unseen states.
"""

import numpy as np
from mdp import (
    XSTART, XMINUS, XEND, N_STATES, N_ACTIONS, H,
    M1, M2, get_expert_policy, get_q_class, transition,
)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def softmax_update(pi: np.ndarray, Q_h: np.ndarray, eta: float) -> np.ndarray:
    """
    Multiplicative-weights / mirror-ascent update for a single stage:

        pi^{k+1}(a|x) proportional to pi^k(a|x) * exp(eta * Q^k_h(x, a))

    Uses the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    pi  : shape [N_STATES, N_ACTIONS] - current policy (rows sum to 1)
    Q_h : shape [N_STATES, N_ACTIONS] - Q-values at stage h
    eta : learning rate (> 0)

    Returns
    -------
    Updated policy of shape [N_STATES, N_ACTIONS].
    """
    log_pi = np.log(np.clip(pi, 1e-300, None))  # avoid log(0)
    log_pi_new = log_pi + eta * Q_h
    # Numerically stable softmax row-by-row
    log_pi_new -= log_pi_new.max(axis=1, keepdims=True)
    pi_new = np.exp(log_pi_new)
    pi_new /= pi_new.sum(axis=1, keepdims=True)
    return pi_new


def q_dot_pi(Q_h: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Compute Q(x, pi) = sum_a pi(a|x) * Q_h(x, a) for every state x.

    Returns shape [N_STATES].
    """
    return (Q_h * pi).sum(axis=1)


# ---------------------------------------------------------------------------
# Algorithm 2 - Online Mirror Ascent for each stage h
# ---------------------------------------------------------------------------

def select_best_q(q_cls: list, states: np.ndarray,
                  expert_actions: np.ndarray, pi_k: np.ndarray) -> np.ndarray:
    """
    Select the Q-function from q_cls that maximises the empirical objective:

        argmax_{Q in Q_class}  sum_i [ Q(x^i, a^{E,i}) - Q(x^i, pi^k) ]

    where Q(x, pi) = sum_a pi(a|x) Q(x, a).

    This is the inner argmax from Algorithm 2, line 3.

    Parameters
    ----------
    q_cls         : list of np.ndarray, each [N_STATES, N_ACTIONS]
    states        : int array [tau_E] - sampled states at stage h
    expert_actions: int array [tau_E] - expert actions at those states
    pi_k          : np.ndarray [N_STATES, N_ACTIONS] - current iterate policy

    Returns
    -------
    Q_best : np.ndarray [N_STATES, N_ACTIONS]
    """
    best_score = -np.inf
    best_Q = q_cls[0]

    for Q in q_cls:
        # Expert Q-value at each sample: Q(x^i, a^{E,i})
        q_expert = Q[states, expert_actions]
        # Policy-averaged Q-value at each sample: Q(x^i, pi^k)
        q_pi = q_dot_pi(Q, pi_k)[states]
        score = np.sum(q_expert - q_pi)
        if score > best_score:
            best_score = score
            best_Q = Q

    return best_Q


def _sample_from_start(pi_h1: np.ndarray, tau_E: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Roll out pi_h1 from x_start to collect tau_E stage-2 states.

    Starting from x_start:
      - With prob pi_h1[x_start, 0] -> next state x_end
      - With prob pi_h1[x_start, 1] -> next state x_minus
    Each of the tau_E trajectories is independent.

    Returns int array [tau_E] of states at stage h=2.
    """
    probs = pi_h1[XSTART]  # [p(0), p(1)]
    actions_h1 = rng.choice(N_ACTIONS, size=tau_E, p=probs)
    # Deterministic transition from x_start
    states_h2 = np.where(actions_h1 == 0, XEND, XMINUS)
    return states_h2


def _vectorized_mw_average(Q_fixed: np.ndarray, K: int, eta: float,
                           return_last: bool = False) -> np.ndarray:
    """
    Compute the MW policy from K iterates starting at Uniform with Q_fixed.

    The k-th iterate is softmax(k * eta * Q_fixed).

    If return_last=False (default): return the average (1/K) sum_{k=0}^{K-1} pi_k.
    If return_last=True:            return pi_{K-1}, the final iterate only.

    Returns shape [N_STATES, N_ACTIONS].
    """
    k_vals = np.arange(K)  # k=0,1,...,K-1
    logits = k_vals[:, None, None] * eta * Q_fixed[None, :, :]  # [K, S, A]
    logits -= logits.max(axis=2, keepdims=True)  # numerical stability
    pi_k = np.exp(logits)
    pi_k /= pi_k.sum(axis=2, keepdims=True)
    if return_last:
        return pi_k[-1]   # [S, A]
    return pi_k.mean(axis=0)  # [S, A]


def algorithm2(mdp_type: int, tau_E: int, K: int,
               eta: float = None,
               rng: np.random.Generator = None) -> tuple:
    """
    Algorithm 2: Interactive finite-horizon IL with Q^{pi_E} realizability.

    Iterates over stages h=1, 2 (H=2). At each stage:
      1. Collect tau_E samples by rolling out the current output policy up to
         stage h-1, then querying the expert at stage h.
      2. Run K steps of online mirror ascent using the finite Q-class.
      3. Output the average policy across all K iterates.

    The K-loop is vectorised analytically:
      - At h=1: both Q functions give score 0 at x_start (Q(x_start,*)=0
        for all Q in the class), so the policy stays Unif for all K steps.
      - At h=2: once any x_minus sample is observed the correct Q always
        dominates (proved by a sign argument), so the same Q is selected at
        every mirror-ascent step and the average can be computed in closed form
        via _vectorized_mw_average.

    Parameters
    ----------
    mdp_type : int  - M1 or M2 (use constants from mdp.py)
    tau_E    : int  - number of expert queries per stage
    K        : int  - number of mirror-ascent iterations
    eta      : float or None  - learning rate.
               Default: sqrt(log|A| / (K * H^2))
    rng      : np.random.Generator or None

    Returns
    -------
    (pi_out_h1, pi_out_h2) : each np.ndarray [N_STATES, N_ACTIONS]
    """
    if rng is None:
        rng = np.random.default_rng()

    if eta is None:
        eta = np.sqrt(np.log(N_ACTIONS) / (K * H ** 2))

    q_cls = get_q_class()
    pi_E = get_expert_policy(mdp_type)

    # -----------------------------------------------------------------------
    # Stage h = 1  (analytical shortcut)
    # -----------------------------------------------------------------------
    # Q_M1(x_start,*) = Q_M2(x_start,*) = 0 for all actions, so every Q
    # gives score 0 on the h=1 dataset.  The MW update exp(eta*0)=1 leaves
    # Unif unchanged.  Therefore pi_out_h1 = Unif for all K and eta.
    pi_out_h1 = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)

    # -----------------------------------------------------------------------
    # Stage h = 2
    # -----------------------------------------------------------------------
    # Sample stage-2 states by rolling out pi_out_h1 (= Unif) from x_start.
    # action 0 -> x_end,  action 1 -> x_minus  (each with prob 0.5)
    actions_h1 = rng.integers(0, N_ACTIONS, size=tau_E)
    states_h2 = np.where(actions_h1 == 0, XEND, XMINUS)

    n_xminus = int((states_h2 == XMINUS).sum())

    if n_xminus == 0:
        # No x_minus samples: Q scores are both 0, policy stays Unif.
        pi_out_h2 = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
    else:
        # Identify which Q to use.
        # Expert action at x_minus (deterministic for this mdp_type).
        expert_a = int(pi_E[XMINUS].argmax())
        # Scores at the initial Unif policy (= 0.5 each action):
        #   score(Q) = n_xminus * (Q[x_minus, expert_a] - 0.5*sum(Q[x_minus]))
        # The correct Q gives score > 0; the wrong one gives score < 0.
        # This ordering is preserved for all subsequent pi_k (sign argument),
        # so the SAME Q is selected at every mirror-ascent step.
        pi_unif_xminus = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
        scores = [
            n_xminus * (Q[XMINUS, expert_a] - Q[XMINUS].dot(pi_unif_xminus))
            for Q in q_cls
        ]
        Q_fixed = q_cls[int(np.argmax(scores))]

        # Vectorised average of K mirror-ascent iterates with Q_fixed.
        pi_out_h2 = _vectorized_mw_average(Q_fixed, K, eta)

    return pi_out_h1, pi_out_h2


# ---------------------------------------------------------------------------
# Behavioural Cloning (offline)
# ---------------------------------------------------------------------------

def collect_offline_expert_data(mdp_type: int, tau_E: int,
                                rng: np.random.Generator = None) -> list:
    """
    Collect tau_E expert trajectories from x_start in the single-copy MDP.

    The expert always takes action 0 at x_start -> transitions to x_end.
    Therefore the expert NEVER visits x_minus.

    Each trajectory is represented as a list of (state, action) pairs at
    each stage. With H=2:
      h=1: (x_start, 0) -> x_end
      h=2: (x_end,   ?)   x_end is absorbing; we record (x_end, action)
                          but BC never needs to act there.

    For BC we only care about which states are covered:
      - x_start: always seen with action 0
      - x_minus: NEVER seen
      - x_end:   seen but irrelevant (absorbing)

    Returns a flat list of (state, action) pairs across all tau_E
    trajectories and both time steps.
    """
    if rng is None:
        rng = np.random.default_rng()

    pi_E = get_expert_policy(mdp_type)
    dataset = []

    for _ in range(tau_E):
        # Stage h=1: always at x_start, expert takes action 0
        s1 = XSTART
        a1 = int(pi_E[s1].argmax())  # = 0
        dataset.append((s1, a1))

        # Stage h=2: transition to x_end (since a1=0 at x_start -> x_end)
        s2 = transition(s1, a1)  # = XEND
        # Expert action at x_end is arbitrary; record as 0 by convention
        a2 = int(pi_E[s2].argmax())
        dataset.append((s2, a2))

    return dataset


def behavioural_cloning(dataset: list) -> np.ndarray:
    """
    Fit a BC policy from a dataset of (state, action) pairs.

    For each state seen in the dataset:
      - Use the empirical action distribution (majority vote / counts).
    For each state NOT seen:
      - Default to Uniform(A).

    Since the single-MDP expert dataset consists only of (x_start, 0) and
    (x_end, 0) pairs:
      pi_BC(0 | x_start) = 1  (always seen with action 0)
      pi_BC at x_minus    = Unif({0,1})  (never seen)
      pi_BC at x_end      = (1, 0)       (seen with action 0; absorbing anyway)

    The SAME policy table is used at both h=1 and h=2 (stationary BC).

    Parameters
    ----------
    dataset : list of (state, action) tuples

    Returns
    -------
    pi_bc : np.ndarray [N_STATES, N_ACTIONS]  (rows sum to 1)
    """
    # Count action occurrences per state
    counts = np.zeros((N_STATES, N_ACTIONS))
    for state, action in dataset:
        counts[state, action] += 1

    pi_bc = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)  # default: Unif

    for s in range(N_STATES):
        total = counts[s].sum()
        if total > 0:
            pi_bc[s] = counts[s] / total  # empirical distribution

    return pi_bc
