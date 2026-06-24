"""
algorithms.py
-------------
Implements the four imitation learning algorithms from the paper:

  Algorithm 4 – Deep Interactive IL with Q^{π_E} realizability
  Algorithm 5 – SPOIL (offline version)
  Algorithm 6 – Deep Behavioural Cloning
  Algorithm 7 – DAgger (non-aggregating variant)

Shared helper functions are defined first, then each algorithm.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm

from networks import PolicyNet, QNet
from expert import get_expert_action, collect_expert_data


# ===========================================================================
# Shared helpers
# ===========================================================================

def collect_rollout_data(policy_net: PolicyNet,
                         expert_model,
                         env: gym.Env,
                         n_steps: int,
                         device: torch.device,
                         subsampling_freq: int = 1,
                         deterministic: bool = False):
    """
    Roll out policy_net in the environment for n_steps total steps.
    At each visited state, query the expert for its action, recording
    only every subsampling_freq-th (state, expert_action) pair.

    Parameters
    ----------
    policy_net       : PolicyNet – acts in the environment
    expert_model     : SB3 PPO   – labelled at each state (oracle)
    env              : gymnasium.Env
    n_steps          : int
    device           : torch.device
    subsampling_freq : int – record every k-th step (default 1)

    Returns
    -------
    states         : torch.Tensor [n_recorded, state_dim] float32
    expert_actions : torch.Tensor [n_recorded]             long
    """
    states_list = []
    actions_list = []

    obs, _ = env.reset()
    steps_collected = 0
    step_in_episode = 0

    while steps_collected < n_steps:
        expert_action = get_expert_action(expert_model, obs, deterministic=deterministic)

        if step_in_episode % subsampling_freq == 0:
            states_list.append(obs.astype(np.float32))
            actions_list.append(expert_action)

        # Step with the LEARNER's action
        learner_action = policy_net.act(obs)
        obs, _, terminated, truncated, _ = env.step(learner_action)
        steps_collected += 1
        step_in_episode += 1

        if terminated or truncated:
            obs, _ = env.reset()
            step_in_episode = 0

    states_t = torch.tensor(np.array(states_list), dtype=torch.float32).to(device)
    actions_t = torch.tensor(np.array(actions_list), dtype=torch.long).to(device)
    return states_t, actions_t


def compute_mirror_ascent_target(q_net: QNet,
                                 pi_k_net: PolicyNet,
                                 states: torch.Tensor,
                                 eta: float,
                                 device: torch.device) -> torch.Tensor:
    """
    Compute the mirror-ascent target distribution:

        π_target(a|s) ∝ π^k(a|s) · exp(η · Q^k(s, a))
      = softmax( log π^k(a|s) + η · Q^k(s, a) )

    All computation under no_grad (target is treated as fixed).

    Returns
    -------
    pi_target : [B, A] float tensor (probabilities summing to 1)
    """
    with torch.no_grad():
        log_pi_k = pi_k_net.log_prob(states)   # [B, A]
        q_vals   = q_net(states)                # [B, A]
        log_target = log_pi_k + eta * q_vals
        pi_target  = F.softmax(log_target, dim=1)   # [B, A]
    return pi_target


def update_q_network(q_net: QNet,
                     pi_k_net: PolicyNet,
                     states: torch.Tensor,
                     expert_actions: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     n_steps: int) -> None:
    """
    Maximize  Σ [ Q(s, a_E) − Q(s, π^k) ]

    Equivalently, minimize the negation:
        loss = −mean( Q(s, a_E) − Σ_a π^k(a|s) Q(s, a) )

    Parameters
    ----------
    q_net          : QNet being updated
    pi_k_net       : current policy (fixed; used for π^k weighting)
    states         : [B, state_dim]
    expert_actions : [B] long
    optimizer      : Adam optimizer for q_net
    n_steps        : number of gradient steps
    """
    B = states.size(0)

    for _ in range(n_steps):
        q_vals   = q_net(states)                                       # [B, A]
        q_expert = q_vals[torch.arange(B), expert_actions]             # [B]

        with torch.no_grad():
            pi_k = pi_k_net.get_probs(states)                          # [B, A]

        q_pi  = (q_vals * pi_k).sum(dim=1)                             # [B]
        loss  = -(q_expert - q_pi).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def update_policy_mirror_ascent(pi_net: PolicyNet,
                                pi_k_net: PolicyNet,
                                q_net: QNet,
                                states: torch.Tensor,
                                eta: float,
                                optimizer: torch.optim.Optimizer,
                                n_steps: int) -> None:
    """
    Maximize the mirror-ascent objective from the paper (Algorithm 4/5):

        Σ_{i,h} Σ_a π_ψ(a|s) [ η Q(s,a) − log(π_ψ(a|s) / π^k(a|s)) ]

    which equals minimizing  KL(π_ψ || π^k) − η ⟨π_ψ, Q⟩.

    Gradient is taken through π_ψ only; π^k and Q are treated as fixed.

    Parameters
    ----------
    pi_net   : PolicyNet being updated (π_ψ)
    pi_k_net : current iterate policy (π^k, fixed)
    q_net    : current Q network (fixed)
    states   : [B, state_dim]
    eta      : mirror-ascent step size
    optimizer: Adam optimizer for pi_net
    n_steps  : number of gradient steps
    """
    for _ in range(n_steps):
        pi_probs = pi_net.get_probs(states)     # [B, A]
        log_pi   = pi_net.log_prob(states)      # [B, A]

        with torch.no_grad():
            log_pi_k = pi_k_net.log_prob(states)    # [B, A]
            q_vals   = q_net(states)                # [B, A]

        # Maximize: Σ_a π_ψ(a|s) [η Q(s,a) + log π^k(a|s) − log π_ψ(a|s)]
        loss = -(pi_probs * (eta * q_vals + log_pi_k - log_pi)).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def update_policy_softmax(pi_net: PolicyNet,
                          q_net: QNet,
                          states: torch.Tensor,
                          eta: float,
                          optimizer: torch.optim.Optimizer,
                          n_steps: int) -> None:
    """
    Softmax mode policy update (Alg 4/5):

        π^{k+1}(·|s) = softmax(η Q^k(s,·))

    Fit π_ψ by minimising KL(softmax(η Q^k) ∥ π_ψ):
        loss = −Σ_a softmax(η Q^k(s,a)) · log π_ψ(a|s)

    Q^k is treated as fixed (no_grad).

    Parameters
    ----------
    pi_net    : PolicyNet being updated
    q_net     : current Q network (fixed)
    states    : [B, state_dim]
    eta       : step size
    optimizer : Adam optimizer for pi_net
    n_steps   : number of gradient steps
    """
    with torch.no_grad():
        pi_target = F.softmax(eta * q_net(states), dim=1)   # [B, A]

    for _ in range(n_steps):
        log_pi = pi_net.log_prob(states)                     # [B, A]
        loss   = -(pi_target * log_pi).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def update_policy_bc(pi_net: PolicyNet,
                     states: torch.Tensor,
                     expert_actions: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     n_steps: int) -> None:
    """
    Maximize log-likelihood: Σ log π(a_E | s)

    Parameters
    ----------
    pi_net         : PolicyNet being updated
    states         : [B, state_dim]
    expert_actions : [B] long
    optimizer      : Adam optimizer for pi_net
    n_steps        : number of gradient steps
    """
    for _ in range(n_steps):
        logits = pi_net(states)                                        # [B, A]
        loss   = F.cross_entropy(logits, expert_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ===========================================================================
# Algorithm 4 – Deep Interactive IL with Q^{π_E} realizability
# ===========================================================================

def run_alg4_interactive_il(expert_model,
                             env: gym.Env,
                             hidden_sizes: list,
                             K: int,
                             tau_E: int,
                             eta: float,
                             q_lr: float,
                             pi_lr: float,
                             q_steps: int,
                             pi_steps: int,
                             device: torch.device,
                             offline_states: torch.Tensor = None,
                             offline_expert_actions: torch.Tensor = None,
                             mode: str = "mirror ascent",
                             L: int = 1,
                             state_dim: int = 6,
                             action_dim: int = 3,
                             subsampling_freq: int = 1,
                             deterministic: bool = False) -> PolicyNet:
    """
    Algorithm 4: Deep Interactive Imitation Learning with Q^{π_E} realizability.

    Outer loop (k=1,...,K):
      1. Collect τ_E/K steps: half from rolling out π^k (learner occupancy),
         half sampled from the offline expert dataset (expert occupancy).
         Aggregate with all past data D_k = ∪_{s≤k} D_s.
    Inner loop (ℓ=1,...,L) on the aggregated dataset:
      2. Fit fresh Q^k_ℓ maximising Σ[Q(s,a_E) − Q(s,π^k_ℓ)].
      3a. (mirror ascent) Fit π^k_{ℓ+1} maximising
            Σ_a π_ψ(a|s)[η Q^k_ℓ(s,a) − log(π_ψ(a|s)/π^k_ℓ(a|s))].
      3b. (softmax) Fit π^k_{ℓ+1} to softmax(η Q^k_ℓ).

    mode : "mirror ascent" (default) or "softmax"
    L    : number of inner Q+π update steps per outer iteration (default 1)

    Parameters
    ----------
    expert_model           : SB3 PPO – oracle labeller (used for learner rollout)
    env                    : gymnasium.Env
    hidden_sizes           : list – e.g. [64,64] or [16,16]
    K                      : int – outer iterations (data collection rounds)
    tau_E                  : int – total expert interaction budget
    eta                    : float – mirror-ascent step size
    q_lr                   : float – learning rate for Q network
    pi_lr                  : float – learning rate for policy network
    q_steps                : int – Adam steps for Q update per inner step
    pi_steps               : int – Adam steps for π update per inner step
    device                 : torch.device
    offline_states         : torch.Tensor [N, state_dim] – pre-collected expert states
    offline_expert_actions : torch.Tensor [N]            – corresponding expert actions
    L                      : int – inner loop iterations per outer iteration

    Returns
    -------
    pi_net : PolicyNet – final trained policy π^K_L
    """
    tau_per_iter = tau_E // K   # data budget per outer iteration

    # Half from learner rollout, half from offline expert dataset
    n_learner = tau_per_iter // 2
    n_expert  = tau_per_iter - n_learner

    offline_states_dev         = offline_states.to(device)
    offline_expert_actions_dev = offline_expert_actions.to(device)

    # π^1_1 – initialised as a random (≈ uniform) network
    pi_current = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)

    # Q network created once; weights persist across all inner and outer
    # iterations (warm start: θ^k_1 = θ^{k-1}_L, θ^k_ℓ = θ^k_{ℓ-1})
    q_net = QNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)

    # Aggregated dataset (grows each outer iteration)
    all_states: list = []
    all_expert_actions: list = []

    print(f"  [Alg4] K={K}, L={L}, τ_E/iter={tau_per_iter} "
          f"(n_learner={n_learner}, n_expert={n_expert}), hidden={hidden_sizes}")

    for k in tqdm(range(1, K + 1), desc="  Alg4 outer iter"):
        # ------------------------------------------------------------------
        # Data collection: half from learner rollout, half from expert dataset
        # ------------------------------------------------------------------
        # Learner half: roll out π^k_1 and query expert at visited states
        states_learner, actions_learner = collect_rollout_data(
            pi_current, expert_model, env, n_learner, device,
            subsampling_freq=subsampling_freq, deterministic=deterministic,
        )

        # Expert half: random sample from offline expert dataset
        idx = torch.randint(0, offline_states_dev.size(0), (n_expert,))
        states_expert  = offline_states_dev[idx]
        actions_expert = offline_expert_actions_dev[idx]

        states_k         = torch.cat([states_learner, states_expert], dim=0)
        expert_actions_k = torch.cat([actions_learner, actions_expert], dim=0)

        # Aggregate: D_k = D_{k-1} ∪ D_k
        all_states.append(states_k)
        all_expert_actions.append(expert_actions_k)
        states_agg         = torch.cat(all_states,         dim=0)
        expert_actions_agg = torch.cat(all_expert_actions, dim=0)

        # ------------------------------------------------------------------
        # Inner loop: ℓ = 1,...,L  (warm-started from pi_current = π^k_1)
        # ------------------------------------------------------------------
        pi_inner = pi_current   # warm start: π^k_1 = π^{k-1}_L

        for _ in range(L):
            # Q update: q_net weights warm-started from θ^k_{ℓ-1}
            update_q_network(q_net, pi_inner, states_agg, expert_actions_agg, q_opt, q_steps)

            # π update → π^k_{ℓ+1}: weights warm-started from ψ^*_{k,ℓ-1}
            pi_next = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
            pi_next.load_state_dict(copy.deepcopy(pi_inner.state_dict()))
            pi_opt  = torch.optim.Adam(pi_next.parameters(), lr=pi_lr)

            if mode == "mirror ascent":
                update_policy_mirror_ascent(pi_next, pi_inner, q_net, states_agg, eta, pi_opt, pi_steps)
            else:  # softmax
                update_policy_softmax(pi_next, q_net, states_agg, eta, pi_opt, pi_steps)

            pi_inner = pi_next  # advance inner iterate

        # π^k_L becomes the warm start for the next outer iteration
        pi_current = pi_inner

    return pi_current


# ===========================================================================
# Algorithm 5 – SPOIL (offline)
# ===========================================================================

def run_alg5_spoil(offline_states: torch.Tensor,
                   offline_expert_actions: torch.Tensor,
                   hidden_sizes: list,
                   K: int,
                   eta: float,
                   q_lr: float,
                   pi_lr: float,
                   q_steps: int,
                   pi_steps: int,
                   device: torch.device,
                   mode: str = "mirror ascent",
                   state_dim: int = 6,
                   action_dim: int = 3) -> PolicyNet:
    """
    Algorithm 5: SPOIL – offline variant of Algorithm 4.

    Receives a pre-collected offline expert dataset (shared with Alg 6).
    All K outer iterations use this fixed dataset.

    Inner update structure:
      - Q-network warm-started across iterations; maximize Q(s, a_E) − Q(s, π^k)
      - π^{k+1} warm-started from π^k weights; mirror-ascent or softmax update

    Parameters
    ----------
    offline_states         : torch.Tensor [N, state_dim] – pre-collected expert states
    offline_expert_actions : torch.Tensor [N]            – corresponding expert actions
    hidden_sizes : list
    K            : int – outer iterations
    eta          : float – mirror-ascent step size
    q_lr, pi_lr  : float
    q_steps, pi_steps : int
    device       : torch.device
    mode         : "mirror ascent" or "softmax"
    state_dim, action_dim : int

    Returns
    -------
    pi_net : PolicyNet – final trained policy π^K
    """
    states         = offline_states.to(device)
    expert_actions = offline_expert_actions.to(device)

    # Initialise π^1
    pi_k = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)

    # Q network created once; weights persist across outer iterations (warm start)
    q_net = QNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)

    print(f"  [Alg5] K={K}, hidden={hidden_sizes}, dataset size={len(states)}")

    for k in tqdm(range(1, K + 1), desc="  Alg5 outer iter"):
        # ------------------------------------------------------------------
        # Q update: q_net weights warm-started from θ_{k-1}
        # ------------------------------------------------------------------
        update_q_network(q_net, pi_k, states, expert_actions, q_opt, q_steps)

        # ------------------------------------------------------------------
        # Policy update: π^{k+1} warm-started from π^k weights
        # ------------------------------------------------------------------
        pi_next = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        pi_next.load_state_dict(copy.deepcopy(pi_k.state_dict()))
        pi_opt  = torch.optim.Adam(pi_next.parameters(), lr=pi_lr)

        if mode == "mirror ascent":
            update_policy_mirror_ascent(pi_next, pi_k, q_net, states, eta, pi_opt, pi_steps)
        else:  # softmax
            update_policy_softmax(pi_next, q_net, states, eta, pi_opt, pi_steps)

        pi_k = pi_next

    return pi_k


# ===========================================================================
# Algorithm 6 – Deep Behavioural Cloning
# ===========================================================================

def run_alg6_bc(offline_states: torch.Tensor,
                offline_expert_actions: torch.Tensor,
                hidden_sizes: list,
                pi_lr: float,
                pi_steps: int,
                device: torch.device,
                state_dim: int = 6,
                action_dim: int = 3) -> PolicyNet:
    """
    Algorithm 6: Deep Behavioural Cloning.

    Receives a pre-collected offline expert dataset (shared with Alg 5).
    Performs a single MLE pass:
        θ* = argmax_θ  Σ_{i,h} log π_θ(a_E | s)

    Parameters
    ----------
    offline_states         : torch.Tensor [N, state_dim] – pre-collected expert states
    offline_expert_actions : torch.Tensor [N]            – corresponding expert actions
    hidden_sizes : list
    pi_lr        : float
    pi_steps     : int – number of gradient steps
    device       : torch.device
    state_dim, action_dim : int

    Returns
    -------
    pi_net : PolicyNet
    """
    states         = offline_states.to(device)
    expert_actions = offline_expert_actions.to(device)

    # Initialise policy
    pi_net = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
    pi_opt = torch.optim.Adam(pi_net.parameters(), lr=pi_lr)

    print(f"  [Alg6/BC] Training for {pi_steps} gradient steps, hidden={hidden_sizes}, dataset size={len(states)}")

    # Single MLE pass (pi_steps gradient steps over entire dataset)
    update_policy_bc(pi_net, states, expert_actions, pi_opt, pi_steps)

    return pi_net


# ===========================================================================
# Algorithm 7 – DAgger (with dataset aggregation)
# ===========================================================================

def run_alg7_dagger(expert_model,
                    env: gym.Env,
                    hidden_sizes: list,
                    K: int,
                    tau_E: int,
                    pi_lr: float,
                    pi_steps: int,
                    device: torch.device,
                    alpha: float = 0.1,
                    state_dim: int = 6,
                    action_dim: int = 3,
                    subsampling_freq: int = 1,
                    deterministic: bool = False) -> PolicyNet:
    """
    Algorithm 7: DAgger with decaying expert mixture (as in the paper).

    At each outer iteration k:
      1. Define the sampling policy:
            π^{sampl}_k = (1−(1−α)^k)·π^k + (1−α)^k·π_E
         States are visited by rolling out π^{sampl}_k; the expert is queried
         at each visited state for its action label.
      2. Aggregate all data collected so far: D_k = D_1 ∪ ... ∪ D_k.
      3. Fit a FRESH policy π^{k+1} via MLE on the aggregated dataset:
            θ_k* = argmax_θ Σ_{s=1}^k Σ_i Σ_h log π_θ(a^{E,i}_{h,s} | x^i_{h,s})

    Parameters
    ----------
    expert_model : SB3 PPO
    env          : gymnasium.Env
    hidden_sizes : list
    K            : int – outer iterations
    tau_E        : int – total expert interaction budget
    pi_lr        : float
    pi_steps     : int – gradient steps per outer iteration
    device       : torch.device
    alpha        : float – mixture decay parameter (default 0.1)

    Returns
    -------
    pi_net : PolicyNet – π^K (last policy)
    """
    tau_per_iter = tau_E // K

    # Initialise π^1 (random)
    pi_k = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)

    # Aggregated dataset lists
    all_states: list = []
    all_expert_actions: list = []

    print(f"  [Alg7/DAgger] K={K}, τ_E/iter={tau_per_iter}, α={alpha}, hidden={hidden_sizes}")

    for k in tqdm(range(1, K + 1), desc="  Alg7 outer iter"):
        # ------------------------------------------------------------------
        # Roll out mixture policy π^{sampl}_k, collect fresh data.
        # At each step: use expert action with prob (1−α)^k, else learner.
        # Record (state, expert_action) regardless of who stepped.
        # ------------------------------------------------------------------
        expert_prob = (1.0 - alpha) ** k

        states_list: list = []
        actions_list: list = []

        obs, _ = env.reset()
        steps_collected = 0
        step_in_episode = 0

        while steps_collected < tau_per_iter:
            expert_action = get_expert_action(expert_model, obs, deterministic=deterministic)

            if step_in_episode % subsampling_freq == 0:
                states_list.append(obs.astype(np.float32))
                actions_list.append(expert_action)

            # Step with the mixture policy
            if np.random.random() < expert_prob:
                step_action = expert_action
            else:
                step_action = pi_k.act(obs)

            obs, _, terminated, truncated, _ = env.step(step_action)
            steps_collected += 1
            step_in_episode += 1

            if terminated or truncated:
                obs, _ = env.reset()
                step_in_episode = 0

        states_k         = torch.tensor(np.array(states_list),  dtype=torch.float32).to(device)
        expert_actions_k = torch.tensor(np.array(actions_list), dtype=torch.long).to(device)

        # ------------------------------------------------------------------
        # Aggregate datasets: D_k = D_{k-1} ∪ D_k
        # ------------------------------------------------------------------
        all_states.append(states_k)
        all_expert_actions.append(expert_actions_k)

        states_agg        = torch.cat(all_states, dim=0)
        expert_actions_agg = torch.cat(all_expert_actions, dim=0)

        # ------------------------------------------------------------------
        # Fit a FRESH π^{k+1} via MLE on the aggregated dataset
        # ------------------------------------------------------------------
        pi_next = PolicyNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes).to(device)
        pi_next.load_state_dict(copy.deepcopy(pi_k.state_dict()))
        pi_opt  = torch.optim.Adam(pi_next.parameters(), lr=pi_lr)

        update_policy_bc(pi_next, states_agg, expert_actions_agg, pi_opt, pi_steps)

        pi_k = pi_next

    return pi_k
