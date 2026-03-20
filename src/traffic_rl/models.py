"""Neural network modules for PPO variants."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(input_dim: int, hidden_sizes: Iterable[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(layer_init(nn.Linear(last_dim, int(hidden))))
        layers.append(nn.Tanh())
        last_dim = int(hidden)
    return nn.Sequential(*layers)


class LocalActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.encoder = build_mlp(obs_dim, hidden_sizes)
        last_dim = hidden_sizes[-1]
        self.actor = layer_init(nn.Linear(last_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(last_dim, 1), std=1.0)

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        hidden = self.encoder(obs)
        logits = self.actor(hidden)
        return Categorical(logits=logits)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(obs)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        value = self.critic(hidden).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(obs)).squeeze(-1)


class FactorizedJointActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.encoder = build_mlp(obs_dim, hidden_sizes)
        last_dim = hidden_sizes[-1]
        self.actor = layer_init(nn.Linear(last_dim, num_agents * action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(last_dim, 1), std=1.0)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(obs)
        logits = self.actor(hidden).view(-1, self.num_agents, self.action_dim)
        dists = [Categorical(logits=logits[:, idx, :]) for idx in range(self.num_agents)]
        if action is None:
            action_chunks = []
            for idx, dist in enumerate(dists):
                if deterministic:
                    action_chunks.append(torch.argmax(logits[:, idx, :], dim=-1))
                else:
                    action_chunks.append(dist.sample())
            action = torch.stack(action_chunks, dim=-1)
        logprob = torch.stack(
            [dist.log_prob(action[:, idx]) for idx, dist in enumerate(dists)],
            dim=-1,
        ).sum(dim=-1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).sum(dim=-1)
        value = self.critic(hidden).squeeze(-1)
        return action, logprob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(obs)).squeeze(-1)


class MAPPOActorCritic(nn.Module):
    def __init__(
        self,
        local_obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_sizes: Iterable[int],
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.num_agents = num_agents
        self.actor_encoder = build_mlp(local_obs_dim, hidden_sizes)
        self.critic_encoder = build_mlp(global_obs_dim, hidden_sizes)
        last_dim = hidden_sizes[-1]
        self.actor = layer_init(nn.Linear(last_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(last_dim, num_agents), std=1.0)

    def get_action_and_value(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        agent_indices: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_hidden = self.actor_encoder(local_obs)
        dist = Categorical(logits=self.actor(actor_hidden))
        if action is None:
            action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        values = self.get_value(global_obs, agent_indices)
        return action, dist.log_prob(action), dist.entropy(), values

    def get_value(self, global_obs: torch.Tensor, agent_indices: torch.Tensor) -> torch.Tensor:
        critic_hidden = self.critic_encoder(global_obs)
        all_values = self.critic(critic_hidden)
        return all_values[torch.arange(global_obs.shape[0], device=global_obs.device), agent_indices]
