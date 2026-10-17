"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # Init NN
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.net(state).reshape(action_chunk.shape)
        return nn.functional.mse_loss(prediction, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.net(state).reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        flat_dim = chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + flat_dim + 1  # state + noisy chunk + tau
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, flat_dim))  # velocity, no activation
        self.net = nn.Sequential(*layers)

    def _velocity(
        self,
        state: torch.Tensor,     # (B, state_dim)
        noisy_chunk: torch.Tensor,  # (B, chunk_size, action_dim)
        tau: torch.Tensor,       # (B, 1)
    ) -> torch.Tensor:
        x = torch.cat([state, noisy_chunk.flatten(1), tau], dim=-1)
        return self.net(x).reshape(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch = state.shape[0]
        noise = torch.randn_like(action_chunk)                  # A_0
        tau = torch.rand(batch, 1, device=state.device)         # (B, 1)
        tau_b = tau.view(batch, 1, 1)                           # broadcast over (K, D)

        noisy_chunk = tau_b * action_chunk + (1 - tau_b) * noise  # A_tau
        target_velocity = action_chunk - noise

        pred_velocity = self._velocity(state, noisy_chunk, tau)
        return nn.functional.mse_loss(pred_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch = state.shape[0]
        chunk = torch.randn(
            batch, self.chunk_size, self.action_dim, device=state.device
        )
        dt = 1.0 / num_steps

        with torch.no_grad():
            for i in range(num_steps):
                tau = torch.full((batch, 1), i * dt, device=state.device)
                chunk = chunk + dt * self._velocity(state, chunk, tau)
        return chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
