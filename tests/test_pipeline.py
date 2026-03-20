from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.algo_shared import train as train_shared
from traffic_rl.comparison import write_comparison_outputs
from traffic_rl.config import load_experiment_config
from traffic_rl.envs import CentralizedGridEnv, ParallelGridEnv, configure_sumo_backend
from traffic_rl.routes import generate_route_manifest


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_experiment_config("configs/env.yaml")
        cls.config["scenario"]["episode_seconds"] = 60
        cls.config["route_generation"]["episode_seconds"] = 60
        cls.config["route_generation"]["num_windows"] = 2
        cls.manifest_path = "routes/manifests/test_manifest.json"
        cls.manifest = generate_route_manifest(
            config=cls.config,
            train_seeds=[0],
            test_seeds=[100],
            intensities=["low"],
            output_path=cls.manifest_path,
        )

    def test_route_manifest_created(self) -> None:
        self.assertTrue((ROOT / self.manifest_path).exists())
        self.assertEqual(len(self.manifest["routes"]), 2)

    def test_centralized_env_step(self) -> None:
        route_file = self.manifest["routes"][0]["path"]
        env = CentralizedGridEnv(self.config, route_file, seed=0)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        obs, reward, _, truncated, _ = env.step([0, 0, 0, 0])
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        env.close()

    def test_parallel_env_step(self) -> None:
        route_file = self.manifest["routes"][0]["path"]
        env = ParallelGridEnv(self.config, route_file, seed=0)
        obs, _ = env.reset(seed=0)
        self.assertEqual(set(obs.keys()), {"1", "2", "5", "6"})
        next_obs, rewards, _, _, _ = env.step({agent: 0 for agent in ("1", "2", "5", "6")})
        self.assertEqual(set(next_obs.keys()), {"1", "2", "5", "6"})
        self.assertEqual(set(rewards.keys()), {"1", "2", "5", "6"})
        env.close()

    def test_sumo_backend_switching(self) -> None:
        self.assertEqual(configure_sumo_backend(use_gui=False), "libsumo")
        self.assertEqual(configure_sumo_backend(use_gui=True), "traci")
        self.assertEqual(configure_sumo_backend(use_gui=False), "libsumo")

    def test_centralized_controls_all_traffic_lights(self) -> None:
        route_file = self.manifest["routes"][0]["path"]
        env = CentralizedGridEnv(self.config, route_file, seed=0)
        env.reset(seed=0)

        # The first two decisions are blocked by min_green. On the third
        # decision, non-zero actions should change the other three lights.
        env.step([0, 1, 2, 3])
        env.step([0, 1, 2, 3])
        env.step([0, 1, 2, 3])

        phases = {ts: env.env.traffic_signals[ts].green_phase for ts in env.env.ts_ids}
        self.assertEqual(phases["1"], 0)
        self.assertEqual(phases["2"], 1)
        self.assertEqual(phases["5"], 2)
        self.assertEqual(phases["6"], 3)
        env.close()

    def test_shared_training_smoke(self) -> None:
        route_specs = [route for route in self.manifest["routes"] if route["split"] == "train"]
        ppo_config = load_experiment_config("configs/ppo_common.yaml", "configs/ppo_shared.yaml")
        ppo_config = copy.deepcopy(ppo_config)
        ppo_config["train"]["total_timesteps"] = 16
        ppo_config["train"]["num_steps"] = 8
        ppo_config["train"]["update_epochs"] = 1
        ppo_config["train"]["num_minibatches"] = 2
        checkpoint_path = train_shared(self.config, ppo_config, route_specs, intensity="low", seed=0)
        self.assertTrue(Path(checkpoint_path).exists())
        tb_root = ROOT / "results" / "tensorboard" / "shared_ppo" / "low" / "seed_0"
        self.assertTrue(tb_root.exists())
        self.assertTrue(any(path.name.startswith("events.out.tfevents") for path in tb_root.rglob("*")))

    def test_comparison_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            results_root = tmp_root / "results" / "eval"
            for algorithm, waiting_time, speed in (
                ("baseline", 10.0, 5.0),
                ("shared_ppo", 7.0, 6.5),
            ):
                target = results_root / algorithm / "test" / "medium"
                target.mkdir(parents=True, exist_ok=True)
                payload = {
                    "algorithm": algorithm,
                    "split": "test",
                    "intensity": "medium",
                    "episodes": 3,
                    "mean_average_waiting_time_mean": waiting_time,
                    "mean_average_waiting_time_std": 0.5,
                    "mean_average_speed_mean": speed,
                    "mean_average_speed_std": 0.2,
                    "mean_travel_time_mean": 50.0,
                    "mean_travel_time_std": 1.0,
                    "throughput_mean": 100.0,
                    "throughput_std": 2.0,
                    "mean_queue_length_mean": 12.0,
                    "mean_queue_length_std": 0.5,
                    "mean_time_loss_mean": 20.0,
                    "mean_time_loss_std": 1.0,
                    "teleports_mean": 0.0,
                    "teleports_std": 0.0,
                }
                with (target / "aggregate.json").open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            outputs = write_comparison_outputs(
                results_root=results_root,
                output_dir=tmp_root / "results" / "compare",
                split="test",
                intensities=["medium"],
            )

            self.assertTrue(outputs["csv"].exists())
            self.assertTrue(outputs["markdown"].exists())
            self.assertTrue(outputs["html"].exists())
            html = outputs["html"].read_text(encoding="utf-8")
            self.assertIn("Traffic RL Result Comparison", html)
            self.assertIn("Shared PPO", html)
            self.assertIn("+30.0%", html)

    def test_comparison_report_generation_all_intensities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            results_root = tmp_root / "results" / "eval"
            fixtures = {
                ("baseline", "low"): {
                    "mean_average_waiting_time_mean": 10.0,
                    "mean_average_speed_mean": 5.0,
                    "mean_travel_time_mean": 50.0,
                    "throughput_mean": 100.0,
                    "mean_queue_length_mean": 12.0,
                    "mean_time_loss_mean": 20.0,
                    "teleports_mean": 0.0,
                },
                ("shared_ppo", "low"): {
                    "mean_average_waiting_time_mean": 8.0,
                    "mean_average_speed_mean": 6.2,
                    "mean_travel_time_mean": 46.0,
                    "throughput_mean": 108.0,
                    "mean_queue_length_mean": 9.0,
                    "mean_time_loss_mean": 16.0,
                    "teleports_mean": 0.0,
                },
                ("baseline", "high"): {
                    "mean_average_waiting_time_mean": 22.0,
                    "mean_average_speed_mean": 4.2,
                    "mean_travel_time_mean": 78.0,
                    "throughput_mean": 200.0,
                    "mean_queue_length_mean": 24.0,
                    "mean_time_loss_mean": 36.0,
                    "teleports_mean": 1.0,
                },
                ("shared_ppo", "high"): {
                    "mean_average_waiting_time_mean": 16.0,
                    "mean_average_speed_mean": 5.0,
                    "mean_travel_time_mean": 69.0,
                    "throughput_mean": 212.0,
                    "mean_queue_length_mean": 18.0,
                    "mean_time_loss_mean": 28.0,
                    "teleports_mean": 0.0,
                },
            }
            for (algorithm, intensity), metrics in fixtures.items():
                target = results_root / algorithm / "test" / intensity
                target.mkdir(parents=True, exist_ok=True)
                payload = {
                    "algorithm": algorithm,
                    "split": "test",
                    "intensity": intensity,
                    "episodes": 3,
                    "mean_average_waiting_time_std": 0.5,
                    "mean_average_speed_std": 0.2,
                    "mean_travel_time_std": 1.0,
                    "throughput_std": 2.0,
                    "mean_queue_length_std": 0.5,
                    "mean_time_loss_std": 1.0,
                    "teleports_std": 0.0,
                    **metrics,
                }
                with (target / "aggregate.json").open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle)

            outputs = write_comparison_outputs(
                results_root=results_root,
                output_dir=tmp_root / "results" / "compare",
                split="test",
            )

            self.assertEqual(outputs["html"].name, "comparison_test_all.html")
            html = outputs["html"].read_text(encoding="utf-8")
            self.assertIn("Overall Overview", html)
            self.assertIn("Average Improvement vs Baseline", html)
            self.assertIn('aria-label="Mean waiting time across intensities"', html)
            self.assertIn("Low", html)
            self.assertIn("High", html)


if __name__ == "__main__":
    unittest.main()
