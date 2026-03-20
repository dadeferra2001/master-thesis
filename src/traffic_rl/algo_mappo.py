"""MAPPO with shared actors and a centralized value network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime import bootstrap_sumo

bootstrap_sumo(prefer_libsumo=True)

import numpy as np
import torch
import torch.optim as optim

from .controllers import MAPPOController, append_agent_id_features, concatenate_global_observation, stack_local_observations
from .envs import ParallelGridEnv
from .metrics import summarize_episode
from .models import MAPPOActorCritic
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
    use_agent_id = bool(train_cfg.get("agent_id", True))

    route_spec = sample_route(route_specs, rng)
    env = ParallelGridEnv(config, route_spec["path"], seed=seed)
    obs_dict, _ = env.reset(seed=seed)

    raw_local_obs_dim = env.observation_dim
    local_obs_dim = raw_local_obs_dim + (len(TLS_IDS) if use_agent_id else 0)
    global_obs_dim = raw_local_obs_dim * len(TLS_IDS)
    action_dim = env.action_dim
    num_agents = len(TLS_IDS)

    model = MAPPOActorCritic(
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_sizes=train_cfg["hidden_sizes"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]), eps=1e-5)

    total_timesteps = int(train_cfg["total_timesteps"])
    num_steps = int(train_cfg["num_steps"])
    num_updates = total_timesteps // num_steps
    batch_size = num_steps * num_agents

    local_obs_buffer = torch.zeros((num_steps, num_agents, local_obs_dim), dtype=torch.float32, device=device)
    global_obs_buffer = torch.zeros((num_steps, global_obs_dim), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((num_steps, num_agents), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((num_steps,), dtype=torch.float32, device=device)
    values_buffer = torch.zeros((num_steps, num_agents), dtype=torch.float32, device=device)

    episode_return = 0.0
    global_step = 0
    log_path = train_log_path("mappo", intensity, seed)
    progress = ProgressReporter("mappo", intensity, seed, total_timesteps, num_updates)
    agent_indices = torch.arange(num_agents, device=device)

    def local_features(current_obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        batch = stack_local_observations(current_obs_dict, TLS_IDS)
        return append_agent_id_features(batch, TLS_IDS) if use_agent_id else batch

    next_local_np = local_features(obs_dict)
    next_global_np = concatenate_global_observation(obs_dict, TLS_IDS)
    next_local = torch.as_tensor(next_local_np, dtype=torch.float32, device=device)
    next_global = torch.as_tensor(next_global_np, dtype=torch.float32, device=device)
    next_done = torch.zeros((), dtype=torch.float32, device=device)

    for update in range(1, num_updates + 1):
        last_pg_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0
        last_total_loss = 0.0
        last_approx_kl = 0.0
        clipfracs: list[float] = []
        lr_now = maybe_anneal_lr(
            optimizer,
            initial_lr=float(train_cfg["learning_rate"]),
            update=update,
            num_updates=num_updates,
            anneal_lr=bool(train_cfg.get("anneal_lr", True)),
        )

        for step in range(num_steps):
            global_step += 1
            local_obs_buffer[step] = next_local
            global_obs_buffer[step] = next_global
            dones_buffer[step] = next_done

            with torch.no_grad():
                repeated_global = next_global.unsqueeze(0).repeat(num_agents, 1)
                action, logprob, _, value = model.get_action_and_value(
                    next_local,
                    repeated_global,
                    agent_indices,
                )
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob
            values_buffer[step] = value

            action_dict = {agent: int(action[idx].item()) for idx, agent in enumerate(TLS_IDS)}
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

            next_local_np = local_features(next_obs_dict)
            next_global_np = concatenate_global_observation(next_obs_dict, TLS_IDS)
            next_local = torch.as_tensor(next_local_np, dtype=torch.float32, device=device)
            next_global = torch.as_tensor(next_global_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = model.get_value(
                next_global.unsqueeze(0).repeat(num_agents, 1),
                agent_indices,
            )
            advantages, returns = compute_gae(
                rewards_buffer,
                values_buffer,
                dones_buffer,
                next_value,
                next_done,
                gamma=float(train_cfg["gamma"]),
                gae_lambda=float(train_cfg["gae_lambda"]),
            )

        b_local_obs = local_obs_buffer.reshape((-1, local_obs_dim))
        b_global_obs = global_obs_buffer.unsqueeze(1).repeat(1, num_agents, 1).reshape((-1, global_obs_dim))
        b_agent_indices = agent_indices.repeat(num_steps)
        b_actions = actions_buffer.reshape(-1)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        for epoch in range(int(train_cfg["update_epochs"])):
            for mb_inds in minibatch_indices(batch_size, int(train_cfg["num_minibatches"]), rng):
                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_local_obs[mb_inds],
                    b_global_obs[mb_inds],
                    b_agent_indices[mb_inds],
                    action=b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    last_approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                    clipfracs.append(float(((ratio - 1.0).abs() > float(train_cfg["clip_coef"])).float().mean().item()))
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
                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())
                last_total_loss = float(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)
        progress.update(
            update,
            global_step,
            lr_now,
            metrics={
                "policy_loss": last_pg_loss,
                "value_loss": last_v_loss,
                "entropy": last_entropy,
                "total_loss": last_total_loss,
                "approx_kl": last_approx_kl,
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "explained_variance": explained_var,
            },
        )

        if update % int(train_cfg["save_every_updates"]) == 0 or update == num_updates:
            checkpoint_path = checkpoint_dir("mappo", intensity, seed) / f"update_{update:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "algo": "mappo",
                    "agent_order": list(TLS_IDS),
                    "local_obs_dim": local_obs_dim,
                    "global_obs_dim": global_obs_dim,
                    "action_dim": action_dim,
                    "num_agents": num_agents,
                    "seed": seed,
                    "intensity": intensity,
                    "use_agent_id": use_agent_id,
                    "train_config": train_cfg,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "learning_rate": lr_now,
                },
            )

    env.close()
    final_path = checkpoint_dir("mappo", intensity, seed) / "final.pt"
    save_checkpoint(
        final_path,
        {
            "algo": "mappo",
            "agent_order": list(TLS_IDS),
            "local_obs_dim": local_obs_dim,
            "global_obs_dim": global_obs_dim,
            "action_dim": action_dim,
            "num_agents": num_agents,
            "seed": seed,
            "intensity": intensity,
            "use_agent_id": use_agent_id,
            "train_config": train_cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
    )
    progress.close()
    return final_path


def load_controller(checkpoint_path: str | Path, device: torch.device | None = None) -> MAPPOController:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    if device is None:
        device = torch.device("cpu")
    model = MAPPOActorCritic(
        local_obs_dim=int(checkpoint["local_obs_dim"]),
        global_obs_dim=int(checkpoint["global_obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        num_agents=int(checkpoint["num_agents"]),
        hidden_sizes=checkpoint["train_config"]["hidden_sizes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return MAPPOController(
        model=model,
        device=device,
        agent_order=checkpoint["agent_order"],
        use_agent_id=bool(checkpoint.get("use_agent_id", True)),
    )
