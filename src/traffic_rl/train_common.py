"""Shared PPO training utilities."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import resolve_path
from .utils import tensorboard_run_dir


def sample_route(route_specs: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    return route_specs[int(rng.integers(len(route_specs)))]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[step + 1]
            nextvalues = values[step + 1]
        delta = rewards[step] + gamma * nextvalues * nextnonterminal - values[step]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[step] = lastgaelam
    returns = advantages + values
    return advantages, returns


def minibatch_indices(batch_size: int, num_minibatches: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(batch_size)
    rng.shuffle(indices)
    return np.array_split(indices, num_minibatches)


def maybe_anneal_lr(
    optimizer: torch.optim.Optimizer,
    initial_lr: float,
    update: int,
    num_updates: int,
    anneal_lr: bool,
) -> float:
    if not anneal_lr:
        return initial_lr
    frac = 1.0 - (update - 1.0) / float(max(num_updates, 1))
    lr_now = frac * initial_lr
    optimizer.param_groups[0]["lr"] = lr_now
    return lr_now


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


class ProgressReporter:
    def __init__(
        self,
        algo: str,
        intensity: str,
        seed: int,
        total_timesteps: int,
        total_updates: int,
        variant: str | None = None,
    ) -> None:
        self.algo = algo
        self.intensity = intensity
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.total_updates = total_updates
        self.start_time = time.time()
        self.episode_returns: deque[float] = deque(maxlen=10)
        self.episode_count = 0
        self.tensorboard_dir = tensorboard_run_dir(algo, intensity, seed, variant=variant)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        print(f"[{self.algo}][{self.intensity}][seed={self.seed}] tensorboard {self.tensorboard_dir}", flush=True)

    def record_episode(
        self,
        episode_return: float,
        global_step: int | None = None,
        episode_summary: dict[str, Any] | None = None,
    ) -> None:
        self.episode_returns.append(float(episode_return))
        self.episode_count += 1
        if global_step is None:
            return
        self.writer.add_scalar("episode/return", float(episode_return), global_step)
        self.writer.add_scalar("episode/count", self.episode_count, global_step)
        if episode_summary is not None:
            for key, value in sorted(episode_summary.items()):
                if key in {"global_step", "update"}:
                    continue
                if isinstance(value, bool):
                    self.writer.add_scalar(f"episode/{key}", int(value), global_step)
                elif isinstance(value, (int, float)):
                    self.writer.add_scalar(f"episode/{key}", float(value), global_step)

    def update(
        self,
        current_update: int,
        global_step: int,
        learning_rate: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        elapsed = max(time.time() - self.start_time, 1e-6)
        sps = global_step / elapsed
        progress = 100.0 * current_update / max(self.total_updates, 1)
        remaining_steps = max(self.total_timesteps - global_step, 0)
        eta_seconds = remaining_steps / sps if sps > 0 else 0.0
        recent_return = np.mean(self.episode_returns) if self.episode_returns else float("nan")
        recent_return_text = f"{recent_return:8.2f}" if self.episode_returns else "   n/a  "
        self.writer.add_scalar("train/update", current_update, global_step)
        self.writer.add_scalar("train/progress_pct", progress, global_step)
        self.writer.add_scalar("train/learning_rate", learning_rate, global_step)
        self.writer.add_scalar("train/sps", sps, global_step)
        self.writer.add_scalar("train/episodes_completed", self.episode_count, global_step)
        if self.episode_returns:
            self.writer.add_scalar("train/recent_return_mean", float(recent_return), global_step)
        if metrics:
            for key, value in sorted(metrics.items()):
                self.writer.add_scalar(f"losses/{key}", float(value), global_step)
        print(
            (
                f"[{self.algo}][{self.intensity}][seed={self.seed}] "
                f"update {current_update}/{self.total_updates} "
                f"({progress:5.1f}%) | "
                f"steps {global_step}/{self.total_timesteps} | "
                f"episodes {self.episode_count} | "
                f"recent_return {recent_return_text} | "
                f"lr {learning_rate:.2e} | "
                f"sps {sps:7.1f} | "
                f"eta {eta_seconds:7.1f}s"
            ),
            flush=True,
        )

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
