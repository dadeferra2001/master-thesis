"""Inference helpers for trained policies."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .scenario import TLS_IDS


def stack_local_observations(obs_dict: dict[str, np.ndarray], agent_order: Iterable[str] = TLS_IDS) -> np.ndarray:
    return np.stack([np.asarray(obs_dict[agent], dtype=np.float32) for agent in agent_order], axis=0)


def concatenate_global_observation(obs_dict: dict[str, np.ndarray], agent_order: Iterable[str] = TLS_IDS) -> np.ndarray:
    return stack_local_observations(obs_dict, agent_order).reshape(-1).astype(np.float32)


def append_agent_id_features(obs_batch: np.ndarray, agent_order: Iterable[str] = TLS_IDS) -> np.ndarray:
    order = list(agent_order)
    eye = np.eye(len(order), dtype=np.float32)
    return np.concatenate([obs_batch.astype(np.float32), eye], axis=-1)


class CentralizedController:
    def __init__(self, model: torch.nn.Module, device: torch.device, agent_order: Iterable[str] = TLS_IDS) -> None:
        self.model = model
        self.device = device
        self.agent_order = tuple(agent_order)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=deterministic)
        return actions.squeeze(0).cpu().numpy().astype(np.int64)


class SharedController:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        agent_order: Iterable[str] = TLS_IDS,
        use_agent_id: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.agent_order = tuple(agent_order)
        self.use_agent_id = use_agent_id

    def act(self, obs_dict: dict[str, np.ndarray], deterministic: bool = True) -> dict[str, int]:
        obs_batch = stack_local_observations(obs_dict, self.agent_order)
        if self.use_agent_id:
            obs_batch = append_agent_id_features(obs_batch, self.agent_order)
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=deterministic)
        actions_np = actions.cpu().numpy().astype(np.int64)
        return {agent: int(actions_np[idx]) for idx, agent in enumerate(self.agent_order)}


class IndependentController:
    def __init__(self, models: list[torch.nn.Module], device: torch.device, agent_order: Iterable[str] = TLS_IDS) -> None:
        self.models = models
        self.device = device
        self.agent_order = tuple(agent_order)

    def act(self, obs_dict: dict[str, np.ndarray], deterministic: bool = True) -> dict[str, int]:
        actions: dict[str, int] = {}
        for idx, agent in enumerate(self.agent_order):
            obs_tensor = torch.as_tensor(obs_dict[agent], dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, _, _ = self.models[idx].get_action_and_value(obs_tensor, deterministic=deterministic)
            actions[agent] = int(action.item())
        return actions


class MAPPOController:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        agent_order: Iterable[str] = TLS_IDS,
        use_agent_id: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.agent_order = tuple(agent_order)
        self.use_agent_id = use_agent_id

    def act(self, obs_dict: dict[str, np.ndarray], deterministic: bool = True) -> dict[str, int]:
        obs_batch = stack_local_observations(obs_dict, self.agent_order)
        if self.use_agent_id:
            obs_batch = append_agent_id_features(obs_batch, self.agent_order)
        global_obs = concatenate_global_observation(obs_dict, self.agent_order)
        local_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        global_tensor = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).repeat(len(self.agent_order), 1)
        agent_indices = torch.arange(len(self.agent_order), device=self.device)
        actions, _, _, _ = self.model.get_action_and_value(
            local_tensor,
            global_tensor,
            agent_indices,
            deterministic=deterministic,
        )
        actions_np = actions.cpu().numpy().astype(np.int64)
        return {agent: int(actions_np[idx]) for idx, agent in enumerate(self.agent_order)}
