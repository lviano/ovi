"""
expert.py
---------
Train, load, and query an expert on a gymnasium environment using stable-baselines3.

Algorithm dispatch:
  MountainCar-v0  → DQN with potential-based reward shaping (energy-based potential).
                    PPO fails on this env due to sparse rewards and poor exploration.
  All other envs  → PPO (MLP 2×64, tanh, entropy bonus).

Public API
----------
train_expert(total_timesteps, save_path, env_name)
load_expert(save_path, env_name)
expert_exists(save_path, env_name)
get_expert_action(expert_model, state)
collect_expert_data(expert_model, env, n_steps)
evaluate_policy_fn(policy_fn, env, n_episodes)
"""

import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper

# ---------------------------------------------------------------------------
# Environment → algorithm routing
# ---------------------------------------------------------------------------

# Environments that need DQN instead of PPO.
_DQN_ENVS = { }

# Continuous-action envs that are discretized before training.
# The expert is DQN on the wrapped (discrete) env; no reward shaping needed.
_PENDULUM_ENVS = {"Pendulum-v1"}


def _use_dqn(env_name: str) -> bool:
    return env_name in _DQN_ENVS


def _model_stem(env_name: str, n_bins: int = None) -> str:
    """Return the filename stem (without .zip) for the given environment.

    For Pendulum-v1 the stem encodes n_bins so experts trained with different
    discretizations are cached separately.
    """
    is_discrete_wrapped = env_name in _PENDULUM_ENVS
    algo = "dqn" if (_use_dqn(env_name) or is_discrete_wrapped) else "ppo"
    stem = f"{algo}_{env_name.replace('-', '_')}"
    if is_discrete_wrapped and n_bins is not None:
        stem += f"_{n_bins}bins"
    return stem


# ---------------------------------------------------------------------------
# Reward-shaping wrapper for MountainCar-v0
# ---------------------------------------------------------------------------

class _MountainCarShapedVecEnv(VecEnvWrapper):
    """
    Potential-based reward shaping for MountainCar-v0, applied at the VecEnv
    level so it is compatible with SB3's internal gymnasium→gym shim.

    Potential  Φ(s) = sin(3 · position)     [proportional to height]

    Shaping bonus = γ · Φ(s') − Φ(s)

    This is the standard potential-based shaping (Ng et al., 1999) which
    provably preserves the set of optimal policies while providing dense
    gradient information to the learner.  The car is incentivised to climb
    as high as possible and to build kinetic energy by oscillating, which is
    exactly the behaviour needed to reach the goal.

    When an episode terminates the VecEnv auto-resets, so obs[i] is already
    the initial observation of the next episode; prev_potential is updated
    accordingly to avoid a spurious shaping signal at episode boundaries.
    """

    def __init__(self, venv, gamma: float = 0.99):
        super().__init__(venv)
        self._gamma = gamma
        self._prev_potentials = np.zeros(venv.num_envs, dtype=np.float64)

    @staticmethod
    def _potential(obs: np.ndarray) -> np.ndarray:
        # obs shape: (n_envs, obs_dim); position is obs[:, 0]
        return np.sin(3.0 * obs[:, 0])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._prev_potentials = self._potential(obs)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        new_potentials = self._potential(obs)
        rewards = rewards + self._gamma * new_potentials - self._prev_potentials
        # obs[i] is the reset obs when dones[i]; update potential for next step
        self._prev_potentials = new_potentials
        return obs, rewards, dones, infos


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_pendulum_expert(total_timesteps: int, n_bins: int) -> DQN:
    """Train a DQN expert on the discretized Pendulum-v1 env.

    Uses VecEnvWrapper (post-shimmy) rather than a gymnasium factory so that
    the action space seen by SB3 is gym.spaces.Discrete (not gymnasium.spaces.Discrete),
    avoiding the action-space type assertion in older SB3 versions.
    """
    import gym as _old_gym

    class _DiscretizedPendulumVecEnv(VecEnvWrapper):
        """VecEnvWrapper that maps Discrete(n_bins) integer actions to Pendulum torques."""
        def __init__(self, venv, n_bins: int):
            torques = np.linspace(-2.0, 2.0, n_bins, dtype=np.float32)
            # Pass gym.spaces.Discrete so SB3's action-space type check passes.
            super().__init__(venv, action_space=_old_gym.spaces.Discrete(n_bins))
            self._torques = torques

        def step_async(self, actions: np.ndarray) -> None:
            continuous = np.array(
                [[self._torques[int(a)]] for a in actions], dtype=np.float32
            )
            self.venv.step_async(continuous)

        def step_wait(self):
            return self.venv.step_wait()

        def reset(self) -> np.ndarray:
            return self.venv.reset()

    print(f"[expert] Training DQN expert on Pendulum-v1 "
          f"({n_bins} discrete actions) for {total_timesteps:,} timesteps ...")

    base_vec = make_vec_env("Pendulum-v1", n_envs=1)
    vec_env  = _DiscretizedPendulumVecEnv(base_vec, n_bins=n_bins)

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
        verbose=1,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=50_000,
        learning_starts=1_000,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


def _train_ppo_expert(total_timesteps: int, save_path: str, env_name: str) -> PPO:
    """Train a PPO expert (used for Acrobot-v1, CartPole-v1, etc.)."""
    print(f"[expert] Training PPO expert on {env_name} for {total_timesteps:,} timesteps ...")

    vec_env = make_vec_env(env_name, n_envs=8)

    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


def _train_dqn_expert(total_timesteps: int, save_path: str, env_name: str) -> DQN:
    """
    Train a DQN expert for MountainCar-v0 using potential-based reward shaping.

    DQN is preferred over PPO here because:
      • epsilon-greedy exploration reliably discovers the sparse goal;
      • the experience replay buffer makes efficient use of rare successes.
    Reward shaping (energy-based potential) densifies the reward signal while
    preserving the optimal policy.
    """
    print(f"[expert] Training DQN expert on {env_name} (with reward shaping) "
          f"for {total_timesteps:,} timesteps ...")

    # make_vec_env handles the gymnasium→gym shimmy compatibility.
    # Apply reward shaping on top via VecEnvWrapper (avoids gym/gymnasium type conflicts).
    # DQN requires n_envs=1.
    base_vec = make_vec_env(env_name, n_envs=1)
    shaped_env = _MountainCarShapedVecEnv(base_vec, gamma=0.99)

    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.Tanh,
    )

    model = DQN(
        policy="MlpPolicy",
        env=shaped_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=100_000,
        learning_starts=1_000,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.5,       # explore for first 50 % of training
        exploration_final_eps=0.05,
        optimize_memory_usage=False,
    )
    model.learn(total_timesteps=total_timesteps)
    shaped_env.close()
    return model


# ---------------------------------------------------------------------------
# Public API – Training
# ---------------------------------------------------------------------------

def train_expert(total_timesteps: int = 1_000_000,
                 save_path: str = "expert_model",
                 env_name: str = "Acrobot-v1",
                 n_bins: int = None):
    """
    Train an expert on the given gymnasium environment and save the model.

    Routes to:
      - DQN on discretized env for Pendulum-v1 (pass n_bins to set discretization)
      - DQN + reward shaping for MountainCar-v0
      - PPO for all other environments

    Parameters
    ----------
    total_timesteps : int
    save_path : str
    env_name : str
    n_bins : int, optional
        Number of discrete bins for Pendulum-v1 (default 11).

    Returns
    -------
    model : PPO or DQN (stable-baselines3)
    """
    if env_name in _PENDULUM_ENVS:
        bins = n_bins or 11
        ts = total_timesteps if total_timesteps != 1_000_000 else 300_000
        model = _train_pendulum_expert(ts, n_bins=bins)
    elif _use_dqn(env_name):
        ts = total_timesteps if total_timesteps != 1_000_000 else 500_000
        model = _train_dqn_expert(ts, save_path, env_name)
    else:
        model = _train_ppo_expert(total_timesteps, save_path, env_name)

    os.makedirs(save_path, exist_ok=True)
    stem = _model_stem(env_name, n_bins=n_bins)
    model_file = os.path.join(save_path, stem)
    model.save(model_file)
    print(f"[expert] Model saved to {model_file}.zip")
    return model


# ---------------------------------------------------------------------------
# Public API – Loading
# ---------------------------------------------------------------------------

def load_expert(save_path: str = "expert_model", env_name: str = "Acrobot-v1",
                n_bins: int = None):
    """Load a previously trained expert (PPO or DQN depending on env_name)."""
    stem = _model_stem(env_name, n_bins=n_bins)
    model_file = os.path.join(save_path, stem)
    print(f"[expert] Loading expert from {model_file}.zip ...")
    if _use_dqn(env_name) or env_name in _PENDULUM_ENVS:
        model = DQN.load(model_file)
    else:
        model = PPO.load(model_file)
    return model


def expert_exists(save_path: str = "expert_model", env_name: str = "Acrobot-v1",
                  n_bins: int = None) -> bool:
    """Return True if the expert model file already exists."""
    stem = _model_stem(env_name, n_bins=n_bins)
    return os.path.isfile(os.path.join(save_path, f"{stem}.zip"))


# ---------------------------------------------------------------------------
# Querying the expert
# ---------------------------------------------------------------------------

def get_expert_action(expert_model: PPO, state: np.ndarray,
                      deterministic: bool = False) -> int:
    """
    Return the expert's action for a single state.

    Parameters
    ----------
    expert_model  : PPO
    state         : np.ndarray of shape (state_dim,)
    deterministic : bool – if True use the greedy (argmax) action (default False)

    Returns
    -------
    action : int
    """
    action, _ = expert_model.predict(state, deterministic=deterministic)
    return int(action)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_expert_data(expert_model: PPO,
                        env: gym.Env,
                        n_steps: int,
                        subsampling_freq: int = 1,
                        deterministic: bool = False,
                        inject_noise: float = 0.0):
    """
    Roll out the EXPERT policy for n_steps total environment steps,
    recording only every subsampling_freq-th (state, action) pair.

    With subsampling_freq=1 (default) every step is recorded.
    With subsampling_freq=k only steps 0, k, 2k, ... within each
    episode are added to the dataset, making BC harder while keeping
    the same trajectory budget.

    Parameters
    ----------
    expert_model     : PPO
    env              : gymnasium.Env (single, not vectorized)
    n_steps          : int – total steps to visit (trajectory budget)
    subsampling_freq : int – record every k-th step (default 1)
    inject_noise     : float – probability of replacing the expert action
                       with a uniformly random action (default 0.0)

    Returns
    -------
    states  : torch.Tensor [n_recorded, state_dim] float32
    actions : torch.Tensor [n_recorded]             long
    """
    states_list = []
    actions_list = []

    obs, _ = env.reset()
    steps_collected = 0
    step_in_episode = 0

    while steps_collected < n_steps:
        expert_action = get_expert_action(expert_model, obs, deterministic=deterministic)
        if inject_noise > 0.0 and np.random.random() < inject_noise:
            action = int(env.action_space.sample())
        else:
            action = expert_action

        if step_in_episode % subsampling_freq == 0:
            states_list.append(obs.astype(np.float32))
            actions_list.append(action)

        obs, _, terminated, truncated, _ = env.step(action)
        steps_collected += 1
        step_in_episode += 1

        if terminated or truncated:
            obs, _ = env.reset()
            step_in_episode = 0

    states_t = torch.tensor(np.array(states_list), dtype=torch.float32)
    actions_t = torch.tensor(np.array(actions_list), dtype=torch.long)
    return states_t, actions_t


def save_offline_dataset(states: torch.Tensor,
                         actions: torch.Tensor,
                         path: str) -> None:
    """Save a (states, actions) offline dataset to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"states": states, "actions": actions}, path)
    print(f"[expert] Offline dataset saved to {path} ({len(states)} steps)")


def load_offline_dataset(path: str):
    """
    Load a (states, actions) offline dataset from disk.

    Returns
    -------
    states  : torch.Tensor [N, state_dim] float32
    actions : torch.Tensor [N]            long
    """
    data = torch.load(path)
    states, actions = data["states"], data["actions"]
    print(f"[expert] Offline dataset loaded from {path} ({len(states)} steps)")
    return states, actions


def offline_dataset_exists(path: str) -> bool:
    """Return True if the offline dataset file already exists."""
    return os.path.isfile(path)


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policy_fn(policy_fn,
                       env: gym.Env,
                       n_episodes: int = 100) -> float:
    """
    Evaluate a callable policy over n_episodes and return the mean return.

    Parameters
    ----------
    policy_fn  : callable(state: np.ndarray) -> action: int
    env        : gymnasium.Env
    n_episodes : int

    Returns
    -------
    mean_return : float
    """
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)

    mean_return = float(np.mean(returns))
    return mean_return
