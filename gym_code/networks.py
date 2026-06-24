"""
networks.py
-----------
Neural network architectures for imitation learning on Acrobot-v1.

Two policy network sizes:
  - Large: 2 hidden layers of 64 units (matches expert MLP)
  - Small: 2 hidden layers of 16 units

One Q-network class (same hidden-size interface).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNet(nn.Module):
    """
    MLP policy network with tanh activations.
    Outputs logits over the 3 Acrobot actions.

    Sizes:
      Large (matching expert): hidden_sizes=[64, 64]
      Small:                   hidden_sizes=[16, 16]
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 3,
                 hidden_sizes: list = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))  # logits, no activation

        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, state_dim] float tensor

        Returns
        -------
        logits : [B, action_dim]
        """
        return self.net(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns log-softmax of logits.

        Returns
        -------
        log_probs : [B, action_dim]
        """
        return F.log_softmax(self.forward(x), dim=1)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities.

        Returns
        -------
        probs : [B, action_dim]
        """
        return F.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def act(self, state_np: np.ndarray) -> int:
        """
        Greedy action selection (no gradient).

        Parameters
        ----------
        state_np : numpy array of shape (state_dim,)

        Returns
        -------
        action : int
        """
        state_t = torch.FloatTensor(state_np).unsqueeze(0)
        logits = self.forward(state_t)
        return int(logits.argmax(dim=1).item())


class QNet(nn.Module):
    """
    MLP Q-network with tanh activations.
    Outputs Q-values for all actions (no output activation).

    Same hidden-size interface as PolicyNet so Large/Small variants
    are created identically.
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 3,
                 hidden_sizes: list = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))  # Q-values, no activation

        self.net = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, state_dim] float tensor

        Returns
        -------
        q_values : [B, action_dim]
        """
        return self.net(x)
