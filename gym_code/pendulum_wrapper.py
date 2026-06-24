"""
pendulum_wrapper.py
-------------------
Discretizes the continuous Box(-2, 2) action space of Pendulum-v1 into
N_BINS uniformly spaced integer actions, making it compatible with the
discrete-action IL algorithms (Alg 4–7).

Action i → torque = linspace(-2, 2, n_bins)[i].
The observation space (cos θ, sin θ, θ̇) is unchanged.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_BINS = 11  # default discretization; gives step size 0.4 N·m


class DiscretizedPendulumWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that converts Pendulum-v1's continuous torque action
    space Box([-2], [2]) into Discrete(n_bins).

    Action i maps to torque = linspace(-2, 2, n_bins)[i].
    """

    def __init__(self, env: gym.Env, n_bins: int = N_BINS):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = spaces.Discrete(n_bins)
        self._torques = np.linspace(-2.0, 2.0, n_bins, dtype=np.float32)

    def step(self, action: int):
        torque = self._torques[int(action)]
        return self.env.step(np.array([torque], dtype=np.float32))
