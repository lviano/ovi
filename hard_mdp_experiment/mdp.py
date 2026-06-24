"""
mdp.py - Core MDP definitions for the hard construction from Theorem 4.2.

The MDP has:
  - H = 2 (finite horizon)
  - States: x_start=0, x_minus=1, x_end=2
  - Actions: {0 ('+'), 1 ('-')}
  - Deterministic transitions:
      (x_start, 0) -> x_end  (reward 0)
      (x_start, 1) -> x_minus (reward 0)
      (x_minus, *) -> x_end  (reward depends on MDP type M1 or M2)
      x_end: absorbing, reward 0
  - Two MDP variants:
      M1: r(x_minus, 0)=0, r(x_minus, 1)=-1  (action 0 is good)
      M2: r(x_minus, 0)=-1, r(x_minus, 1)=0  (action 1 is good)
  - 2D linear features for Q^{pi_E} realizability (Sec. 4.1):
      phi(x_start, 0) = phi(x_start, 1) = [0, 0]
      phi(x_minus, 0) = [0, 1]
      phi(x_minus, 1) = [1, 0]
      theta_1 = [-1, 0] -> Q_M1 = phi^T theta_1
      theta_2 = [0, -1] -> Q_M2 = phi^T theta_2
  - Expert policy (deterministic):
      pi_E(0 | x_start) = 1  (always takes '+')
      pi_E(0 | x_minus) = 1 in M1;  pi_E(1 | x_minus) = 1 in M2
"""

import numpy as np

# ---------------------------------------------------------------------------
# State and action constants
# ---------------------------------------------------------------------------
XSTART = 0
XMINUS = 1
XEND = 2

N_STATES = 3
N_ACTIONS = 2
H = 2

# MDP type identifiers
M1 = 0
M2 = 1


def get_rewards(mdp_type: int) -> np.ndarray:
    """
    Return reward table r[state, action] for the given MDP type.

    M1: r(x_minus, 0)=0,  r(x_minus, 1)=-1  (action 0 / '+' is the good one)
    M2: r(x_minus, 0)=-1, r(x_minus, 1)=0   (action 1 / '-' is the good one)
    All other (state, action) rewards are 0.
    """
    r = np.zeros((N_STATES, N_ACTIONS))
    if mdp_type == M1:
        r[XMINUS, 0] = 0.0
        r[XMINUS, 1] = -1.0
    elif mdp_type == M2:
        r[XMINUS, 0] = -1.0
        r[XMINUS, 1] = 0.0
    else:
        raise ValueError(f"Unknown mdp_type: {mdp_type}. Use M1=0 or M2=1.")
    return r


def get_expert_policy(mdp_type: int) -> np.ndarray:
    """
    Return the deterministic expert policy pi_E[state, action].

    The expert always takes action 0 ('+') at x_start.
    At x_minus the expert takes the reward-maximising action:
      M1: action 0 ('+') has reward 0 vs action 1 ('-') reward -1  -> takes 0
      M2: action 1 ('-') has reward 0 vs action 0 ('+') reward -1  -> takes 1
    x_end is absorbing; policy value there is arbitrary (set to Unif for
    completeness but never reached on the expert trajectory).
    """
    pi = np.zeros((N_STATES, N_ACTIONS))
    # x_start: always action 0
    pi[XSTART, 0] = 1.0
    pi[XSTART, 1] = 0.0
    # x_minus: good action depends on MDP type
    if mdp_type == M1:
        pi[XMINUS, 0] = 1.0  # action 0 is optimal (reward 0 vs -1)
        pi[XMINUS, 1] = 0.0
    else:  # M2
        pi[XMINUS, 0] = 0.0
        pi[XMINUS, 1] = 1.0  # action 1 is optimal (reward 0 vs -1)
    # x_end: absorbing, uniform by convention
    pi[XEND, 0] = 0.5
    pi[XEND, 1] = 0.5
    return pi


def get_q_class() -> list:
    """
    Return the finite Q-function class Q = {Q_M1, Q_M2}.

    Each Q function is a numpy array of shape [N_STATES, N_ACTIONS].

    Q values are derived from the linear features phi and parameter vectors:
      theta_1 = [-1, 0]:
        Q_M1(x_minus, 0) = phi(x_minus,0)^T theta_1 = [0,1]^T [-1,0] = 0
        Q_M1(x_minus, 1) = phi(x_minus,1)^T theta_1 = [1,0]^T [-1,0] = -1
      theta_2 = [0, -1]:
        Q_M2(x_minus, 0) = [0,1]^T [0,-1] = -1
        Q_M2(x_minus, 1) = [1,0]^T [0,-1] = 0
      All other entries are 0 (features at x_start are [0,0]; x_end absorbing).
    """
    Q_M1 = np.zeros((N_STATES, N_ACTIONS))
    Q_M1[XMINUS, 0] = 0.0
    Q_M1[XMINUS, 1] = -1.0

    Q_M2 = np.zeros((N_STATES, N_ACTIONS))
    Q_M2[XMINUS, 0] = -1.0
    Q_M2[XMINUS, 1] = 0.0

    return [Q_M1, Q_M2]


def transition(state: int, action: int) -> int:
    """
    Deterministic transition function (same for both MDP types).

    (x_start, 0) -> x_end
    (x_start, 1) -> x_minus
    (x_minus, *) -> x_end
    (x_end,   *) -> x_end  (absorbing)
    """
    if state == XSTART:
        return XEND if action == 0 else XMINUS
    else:
        return XEND


def evaluate_policy(pi_h1: np.ndarray, pi_h2: np.ndarray,
                    mdp_type: int) -> float:
    """
    Evaluate V^pi(x_start) for a two-stage policy (pi_h1, pi_h2).

    pi_h1[state, action] = probability of action at stage h=1
    pi_h2[state, action] = probability of action at stage h=2

    Value computation (horizon H=2, no discounting):
      V_2(x_minus) = sum_a pi_h2[x_minus, a] * r[x_minus, a]
      V_2(x_end)   = 0  (absorbing)
      V_1(x_start) = pi_h1[x_start, 0] * V_2(x_end)
                   + pi_h1[x_start, 1] * V_2(x_minus)
                   = pi_h1[x_start, 1] * V_2(x_minus)

    Since r is only nonzero at x_minus and all paths end at x_end after
    two steps, this is exact.
    """
    r = get_rewards(mdp_type)
    # Stage-2 value at x_minus
    V2_xminus = float(np.dot(pi_h2[XMINUS], r[XMINUS]))
    # Stage-1 value at x_start
    V1_xstart = pi_h1[XSTART, 0] * 0.0 + pi_h1[XSTART, 1] * V2_xminus
    return V1_xstart


def expert_value() -> float:
    """
    V^{pi_E}(x_start) = 0.

    The expert always takes action 0 at x_start -> goes to x_end directly
    and collects 0 reward. So the expert value is 0 in both M1 and M2.
    """
    return 0.0


def suboptimality(pi_h1: np.ndarray, pi_h2: np.ndarray,
                  mdp_type: int) -> float:
    """
    Suboptimality = V^{pi_E}(x_start) - V^pi(x_start).

    Since expert_value() = 0 and V^pi <= 0, suboptimality >= 0.

    Closed-form:
      subopt = -V^pi(x_start)
             = -pi_h1[x_start, 1] * V2_xminus
             = pi_h1[x_start, 1] * (- sum_a pi_h2[x_minus,a] * r[x_minus,a])
    """
    return expert_value() - evaluate_policy(pi_h1, pi_h2, mdp_type)
