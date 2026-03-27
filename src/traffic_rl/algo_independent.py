"""Independent PPO with one policy per traffic light."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime import bootstrap_sumo

bootstrap_sumo(prefer_libsumo=True)

import numpy as np
import torch
import torch.optim as optim

from .controllers import IndependentController, stack_local_observations
from .config import experiment_variant
from .envs import ParallelGridEnv
from .metrics import summarize_episode
from .models import LocalActorCritic
from .scenario import TLS_IDS
from .train_common import (
    ProgressReporter,
    compute_gae,
    maybe_anneal_lr,
    minibatch_indices,
    sample_route,
    save_checkpoint,
)
from .utils import append_jsonl, checkpoint_dir, get_device, train_log_path


def train(
    config: dict[str, Any],
    ppo_config: dict[str, Any],
    route_specs: list[dict[str, Any]],
    intensity: str,
    seed: int,
) -> Path:
    train_cfg = ppo_config["train"]
    rng = np.random.default_rng(seed)
    device = get_device(train_cfg.get("device", "auto"))
    variant = experiment_variant(config)

    route_spec = sample_route(route_specs, rng)
    env = ParallelGridEnv(config, route_spec["path"], seed=seed)
    obs_dict, _ = env.reset(seed=seed)
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    num_agents = len(TLS_IDS)

    models = [LocalActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=train_cfg["hidden_sizes"]).to(device) for _ in TLS_IDS]
    optimizers = [optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]), eps=1e-5) for model in models]

    total_timesteps = int(train_cfg["total_timesteps"])
    num_steps = int(train_cfg["num_steps"])
    num_updates = total_timesteps // num_steps
    batch_size = num_steps

    obs_buffer = torch.zeros((num_steps, num_agents, obs_dim), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((num_steps, num_agents), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((num_steps,), dtype=torch.float32, device=device)
    values_buffer = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)

    episode_return = 0.0
    global_step = 0
    log_path = train_log_path("independent_ppo", intensity, seed, variant=variant)
    progress = ProgressReporter("independent_ppo", intensity, seed, total_timesteps, num_updates, variant=variant)

    next_obs = torch.as_tensor(stack_local_observations(obs_dict, TLS_IDS), dtype=torch.float32, device=device)
    next_done = torch.zeros((), dtype=torch.float32, device=device)

    for update in range(1, num_updates + 1):
        last_pg_losses: list[float] = []
        last_v_losses: list[float] = []
        last_entropies: list[float] = []
        last_total_losses: list[float] = []
        last_approx_kls: list[float] = []
        clipfracs: list[float] = []
        for optimizer in optimizers:
            maybe_anneal_lr(
                optimizer,
                initial_lr=float(train_cfg["learning_rate"]),
                update=update,
                num_updates=num_updates,
                anneal_lr=bool(train_cfg.get("anneal_lr", True)),
            )

        for step in range(num_steps):
            global_step += 1
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            chosen_actions = []
            chosen_logprobs = []
            chosen_values = []
            with torch.no_grad():
                for idx, model in enumerate(models):
                    action, logprob, _, value = model.get_action_and_value(next_obs[idx].unsqueeze(0))
                    chosen_actions.append(action.squeeze(0))
                    chosen_logprobs.append(logprob.squeeze(0))
                    chosen_values.append(value.squeeze(0))
            action_tensor = torch.stack(chosen_actions)
            actions_buffer[step] = action_tensor
            logprobs_buffer[step] = torch.stack(chosen_logprobs)
            values_buffer[step] = torch.stack(chosen_values)

            action_dict = {agent: int(action_tensor[idx].item()) for idx, agent in enumerate(TLS_IDS)}
            next_obs_dict, rewards, terminations, truncations, _ = env.step(action_dict)
            rewards_buffer[step] = torch.as_tensor(
                [float(rewards[agent]) for agent in TLS_IDS],
                dtype=torch.float32,
                device=device,
            )
            episode_return += float(sum(float(rewards[agent]) for agent in TLS_IDS))
            done = bool(all(terminations.values()) or all(truncations.values()))
            next_done = torch.as_tensor(float(done), dtype=torch.float32, device=device)

            if done:
                episode_summary = summarize_episode(
                    env.history,
                    extra={
                        "update": update,
                        "global_step": global_step,
                        "episode_return": episode_return,
                    },
                )
                append_jsonl(log_path, episode_summary)
                progress.record_episode(episode_return, global_step=global_step, episode_summary=episode_summary)
                env.close()
                route_spec = sample_route(route_specs, rng)
                env = ParallelGridEnv(config, route_spec["path"], seed=int(rng.integers(1_000_000)))
                next_obs_dict, _ = env.reset(seed=int(rng.integers(1_000_000)))
                episode_return = 0.0

            next_obs = torch.as_tensor(stack_local_observations(next_obs_dict, TLS_IDS), dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = torch.stack([model.get_value(next_obs[idx].unsqueeze(0)).squeeze(0) for idx, model in enumerate(models)])
            advantages, returns = compute_gae(
                rewards_buffer,
                values_buffer,
                dones_buffer,
                next_value,
                next_done,
                gamma=float(train_cfg["gamma"]),
                gae_lambda=float(train_cfg["gae_lambda"]),
            )

        for idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            b_obs = obs_buffer[:, idx, :].reshape((-1, obs_dim))
            b_actions = actions_buffer[:, idx].reshape(-1)
            b_logprobs = logprobs_buffer[:, idx].reshape(-1)
            b_advantages = advantages[:, idx].reshape(-1)
            b_returns = returns[:, idx].reshape(-1)
            b_values = values_buffer[:, idx].reshape(-1)

            for epoch in range(int(train_cfg["update_epochs"])):
                for mb_inds in minibatch_indices(batch_size, int(train_cfg["num_minibatches"]), rng):
                    _, newlogprob, entropy, newvalue = model.get_action_and_value(
                        b_obs[mb_inds],
                        action=b_actions[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        last_approx_kls.append(float(((ratio - 1.0) - logratio).mean().item()))
                        clipfracs.append(
                            float(((ratio - 1.0).abs() > float(train_cfg["clip_coef"])).float().mean().item())
                        )
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - float(train_cfg["clip_coef"]),
                        1.0 + float(train_cfg["clip_coef"]),
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -float(train_cfg["clip_coef"]),
                        float(train_cfg["clip_coef"]),
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - float(train_cfg["ent_coef"]) * entropy_loss
                        + float(train_cfg["vf_coef"]) * v_loss
                    )
                    last_pg_losses.append(float(pg_loss.item()))
                    last_v_losses.append(float(v_loss.item()))
                    last_entropies.append(float(entropy_loss.item()))
                    last_total_losses.append(float(loss.item()))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                    optimizer.step()

        y_pred = values_buffer.detach().cpu().numpy().reshape(-1)
        y_true = returns.detach().cpu().numpy().reshape(-1)
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)
        progress.update(
            update,
            global_step,
            float(optimizers[0].param_groups[0]["lr"]),
            metrics={
                "policy_loss": float(np.mean(last_pg_losses)) if last_pg_losses else 0.0,
                "value_loss": float(np.mean(last_v_losses)) if last_v_losses else 0.0,
                "entropy": float(np.mean(last_entropies)) if last_entropies else 0.0,
                "total_loss": float(np.mean(last_total_losses)) if last_total_losses else 0.0,
                "approx_kl": float(np.mean(last_approx_kls)) if last_approx_kls else 0.0,
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "explained_variance": explained_var,
            },
        )

        if update % int(train_cfg["save_every_updates"]) == 0 or update == num_updates:
            checkpoint_path = checkpoint_dir("independent_ppo", intensity, seed, variant=variant) / f"update_{update:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "algo": "independent_ppo",
                    "variant": variant,
                    "agent_order": list(TLS_IDS),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "num_agents": num_agents,
                    "seed": seed,
                    "intensity": intensity,
                    "train_config": train_cfg,
                    "model_state_dicts": [model.state_dict() for model in models],
                    "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
                    "global_step": global_step,
                },
            )

    env.close()
    final_path = checkpoint_dir("independent_ppo", intensity, seed, variant=variant) / "final.pt"
    save_checkpoint(
        final_path,
        {
            "algo": "independent_ppo",
            "variant": variant,
            "agent_order": list(TLS_IDS),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "num_agents": num_agents,
            "seed": seed,
            "intensity": intensity,
            "train_config": train_cfg,
            "model_state_dicts": [model.state_dict() for model in models],
            "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
            "global_step": global_step,
        },
    )
    progress.close()
    return final_path


def load_controller(checkpoint_path: str | Path, device: torch.device | None = None) -> IndependentController:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    if device is None:
        device = torch.device("cpu")
    models = []
    for state_dict in checkpoint["model_state_dicts"]:
        model = LocalActorCritic(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_sizes=checkpoint["train_config"]["hidden_sizes"],
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return IndependentController(models=models, device=device, agent_order=checkpoint["agent_order"])
