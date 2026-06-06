from __future__ import annotations

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.algo_shared import train as train_shared
from traffic_rl.comparison import write_comparison_outputs
from traffic_rl.config import load_experiment_config, set_pedestrians_enabled
from traffic_rl.envs import CentralizedGridEnv, ParallelGridEnv, configure_sumo_backend
from traffic_rl.evaluation import validate_checkpoint_variant
from traffic_rl.metrics import summarize_episode
from traffic_rl.pedestrians import make_pedestrian_reward_fn
from traffic_rl.routes import generate_route_file, generate_route_manifest
from traffic_rl.train_common import maybe_anneal_coefficient
from traffic_rl.utils import route_specs_for_split
from sumo_rl.environment.traffic_signal import TrafficSignal


class _FakeEdgeAPI:
    def __init__(self, edge_people: dict[str, list[str]]) -> None:
        self.edge_people = edge_people

    def getLastStepPersonIDs(self, edge_id: str) -> list[str]:
        return list(self.edge_people.get(edge_id, []))


class _FakePersonAPI:
    def __init__(self, waiting_times: dict[str, float]) -> None:
        self.waiting_times = waiting_times

    def getWaitingTime(self, person_id: str) -> float:
        return float(self.waiting_times.get(person_id, 0.0))


class _FakeTrafficLightAPI:
    def __init__(self, state: str) -> None:
        self.state = state

    def getControlledLinks(self, tls_id: str) -> list[list[tuple[str, str, str]]]:
        return [
            [(f":{tls_id}_w0_0", f":{tls_id}_c0_0", "s")],
            [(f":{tls_id}_w1_0", f":{tls_id}_c1_0", "s")],
        ]

    def getRedYellowGreenState(self, tls_id: str) -> str:
        return self.state


class _FakeSumo:
    def __init__(self, state: str, edge_people: dict[str, list[str]], waiting_times: dict[str, float]) -> None:
        self.edge = _FakeEdgeAPI(edge_people)
        self.person = _FakePersonAPI(waiting_times)
        self.trafficlight = _FakeTrafficLightAPI(state)


class _FakeTrafficSignal:
    def __init__(self, state: str, edge_people: dict[str, list[str]], waiting_times: dict[str, float]) -> None:
        self.id = "1"
        self.delta_time = 5
        self.sumo = _FakeSumo(state, edge_people, waiting_times)


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp_root = Path(cls.tmpdir.name)
        cls.config = load_experiment_config("configs/env.yaml")
        cls.config["scenario"]["episode_seconds"] = 60
        cls.config["route_generation"]["episode_seconds"] = 60
        cls.config["route_generation"]["num_windows"] = 2
        cls.manifest_path = cls.tmp_root / "test_manifest.json"
        cls.manifest = generate_route_manifest(
            config=cls.config,
            train_seeds=[0],
            test_seeds=[100],
            intensities=["low"],
            output_path=cls.manifest_path,
        )
        cls.ped_config = copy.deepcopy(cls.config)
        set_pedestrians_enabled(cls.ped_config, True)
        cls.ped_manifest_path = cls.tmp_root / "test_manifest_peds.json"
        cls.ped_manifest = generate_route_manifest(
            config=cls.ped_config,
            train_seeds=[0],
            test_seeds=[100],
            intensities=["low"],
            output_path=cls.ped_manifest_path,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_route_manifest_created(self) -> None:
        self.assertTrue(self.manifest_path.exists())
        self.assertEqual(len(self.manifest["routes"]), 2)

    def test_pedestrian_route_manifest_created(self) -> None:
        self.assertTrue(self.ped_manifest_path.exists())
        self.assertEqual(len(self.ped_manifest["routes"]), 2)
        self.assertTrue(all(route["with_pedestrians"] for route in self.ped_manifest["routes"]))
        self.assertTrue(all("routes_peds/" in route["path"] for route in self.ped_manifest["routes"]))

    def test_route_specs_reject_manifest_route_mismatch(self) -> None:
        stale_route_path = self.tmp_root / "stale_seed_321.rou.xml"
        route_record = generate_route_file(stale_route_path, intensity="low", seed=321, config=self.config)
        route_record["split"] = "test"

        root = ET.parse(stale_route_path).getroot()
        for element in list(root)[len(list(root)) // 2 :]:
            root.remove(element)
        ET.ElementTree(root).write(stale_route_path, encoding="utf-8", xml_declaration=False)

        manifest = {"routes": [route_record]}
        with self.assertRaisesRegex(ValueError, "Route manifest mismatch"):
            route_specs_for_split(manifest, split="test", intensity="low")

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

    def test_pedestrian_parallel_env_step(self) -> None:
        route_file = self.ped_manifest["routes"][0]["path"]
        env = ParallelGridEnv(self.ped_config, route_file, seed=0)
        obs, _ = env.reset(seed=0)
        self.assertEqual(set(obs.keys()), {"1", "2", "5", "6"})
        self.assertGreater(env.observation_dim, 0)
        next_obs, rewards, _, _, _ = env.step({agent: 0 for agent in ("1", "2", "5", "6")})
        self.assertEqual(set(next_obs.keys()), {"1", "2", "5", "6"})
        self.assertEqual(set(rewards.keys()), {"1", "2", "5", "6"})
        summary = summarize_episode(env.history)
        self.assertIn("mean_average_pedestrian_waiting_time", summary)
        self.assertIn("mean_pedestrian_queue_length", summary)
        self.assertIn("mean_max_pedestrian_waiting_time", summary)
        self.assertIn("max_pedestrian_waiting_time", summary)
        self.assertIn("pedestrian_total_waiting_time", env.history[-1])
        self.assertIn("pedestrian_total_queue_length", env.history[-1])
        self.assertIn("pedestrian_max_waiting_time", env.history[-1])
        env.close()

    def test_pedestrian_reward_adds_fairness_penalties(self) -> None:
        config = load_experiment_config("configs/env.yaml")
        set_pedestrians_enabled(config, True)
        config["scenario"]["pedestrians"]["reward_weight"] = 0.0
        config["scenario"]["pedestrians"]["fairness_wait_threshold"] = 15.0
        config["scenario"]["pedestrians"]["max_wait_penalty"] = 2.0
        config["scenario"]["pedestrians"]["starvation_penalty"] = 3.0

        ts = _FakeTrafficSignal(
            state="rr",
            edge_people={":1_w0": ["p0"], ":1_w1": []},
            waiting_times={"p0": 20.0},
        )

        with patch.dict(TrafficSignal.reward_fns, {"diff-waiting-time": lambda _ts: 0.0}, clear=False):
            reward_fn = make_pedestrian_reward_fn(config)
            rewards = [reward_fn(ts) for _ in range(4)]

            self.assertLess(rewards[0], 0.0)
            self.assertLess(rewards[-1], rewards[0])

            ts.sumo.trafficlight.state = "Gr"
            served_reward = reward_fn(ts)
            self.assertGreater(served_reward, rewards[-1])

    def test_pedestrian_reward_uses_mean_wait_signal_by_default(self) -> None:
        config = load_experiment_config("configs/env.yaml")
        set_pedestrians_enabled(config, True)
        config["scenario"]["pedestrians"]["reward_weight"] = 1.0
        config["scenario"]["pedestrians"]["waiting_time_scale"] = 20.0
        config["scenario"]["pedestrians"]["fairness_wait_threshold"] = 999.0
        config["scenario"]["pedestrians"]["max_wait_penalty"] = 0.0
        config["scenario"]["pedestrians"]["starvation_penalty"] = 0.0

        single_wait_ts = _FakeTrafficSignal(
            state="Gr",
            edge_people={":1_w0": ["p0"], ":1_w1": []},
            waiting_times={"p0": 20.0},
        )
        double_wait_ts = _FakeTrafficSignal(
            state="Gr",
            edge_people={":1_w0": ["p0", "p1"], ":1_w1": []},
            waiting_times={"p0": 20.0, "p1": 20.0},
        )

        with patch.dict(TrafficSignal.reward_fns, {"diff-waiting-time": lambda _ts: 0.0}, clear=False):
            reward_fn = make_pedestrian_reward_fn(config)
            self.assertAlmostEqual(reward_fn(single_wait_ts), reward_fn(double_wait_ts))

    def test_sumo_backend_switching(self) -> None:
        self.assertEqual(configure_sumo_backend(use_gui=False), "libsumo")
        self.assertEqual(configure_sumo_backend(use_gui=True), "traci")
        self.assertEqual(configure_sumo_backend(use_gui=False), "libsumo")

    def test_centralized_step_averages_agent_rewards(self) -> None:
        class _FakeCentralizedRawEnv:
            def step(self, action_dict: dict[str, int]):
                self.last_action_dict = action_dict
                obs_dict = {
                    "1": np.asarray([0.0], dtype=np.float32),
                    "2": np.asarray([1.0], dtype=np.float32),
                    "5": np.asarray([2.0], dtype=np.float32),
                    "6": np.asarray([3.0], dtype=np.float32),
                }
                rewards = {"1": 1.0, "2": 2.0, "5": 3.0, "6": 4.0}
                dones = {"__all__": False}
                return obs_dict, rewards, dones, {}

        fake_wrapper = type("FakeCentralizedWrapper", (), {})()
        fake_wrapper.agent_order = ("1", "2", "5", "6")
        fake_wrapper.env = _FakeCentralizedRawEnv()

        obs, reward, terminated, truncated, _ = CentralizedGridEnv.step(
            fake_wrapper,
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )

        self.assertAlmostEqual(reward, 2.5)
        self.assertEqual(obs.shape, (4,))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(fake_wrapper.env.last_action_dict, {"1": 0, "2": 1, "5": 2, "6": 3})

    def test_entropy_coefficient_anneals_linearly(self) -> None:
        self.assertAlmostEqual(maybe_anneal_coefficient(0.01, 0.001, update=1, num_updates=5, anneal=True), 0.01)
        self.assertAlmostEqual(maybe_anneal_coefficient(0.01, 0.001, update=3, num_updates=5, anneal=True), 0.0055)
        self.assertAlmostEqual(maybe_anneal_coefficient(0.01, 0.001, update=5, num_updates=5, anneal=True), 0.001)
        self.assertAlmostEqual(maybe_anneal_coefficient(0.01, 0.001, update=5, num_updates=5, anneal=False), 0.01)

    def test_eval_rejects_variant_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Checkpoint variant mismatch"):
            validate_checkpoint_variant({"variant": None}, "peds", "results/checkpoints/shared_ppo/medium/seed_0/final.pt")
        validate_checkpoint_variant({"variant": "peds"}, "peds", "results/checkpoints/shared_ppo/peds/medium/seed_0/final.pt")

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

    def test_shared_training_pedestrian_variant_smoke(self) -> None:
        route_specs = [route for route in self.ped_manifest["routes"] if route["split"] == "train"]
        ppo_config = load_experiment_config("configs/ppo_common.yaml", "configs/ppo_shared.yaml")
        ppo_config = copy.deepcopy(ppo_config)
        ppo_config["train"]["total_timesteps"] = 16
        ppo_config["train"]["num_steps"] = 8
        ppo_config["train"]["update_epochs"] = 1
        ppo_config["train"]["num_minibatches"] = 2
        checkpoint_path = train_shared(self.ped_config, ppo_config, route_specs, intensity="low", seed=1)
        self.assertTrue(Path(checkpoint_path).exists())
        self.assertIn("/peds/", checkpoint_path.as_posix())
        tb_root = ROOT / "results" / "tensorboard" / "shared_ppo" / "peds" / "low" / "seed_1"
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

    def test_comparison_report_separates_vehicle_and_pedestrian_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            results_root = tmp_root / "results" / "eval"
            fixtures = {
                ("baseline", "vehicle"): 10.0,
                ("shared_ppo", "vehicle"): 7.0,
                ("baseline", "peds"): 14.0,
                ("shared_ppo", "peds"): 9.0,
            }
            for (algorithm, variant), waiting_time in fixtures.items():
                target = results_root / algorithm
                if variant == "peds":
                    target = target / "peds"
                target = target / "test" / "medium"
                target.mkdir(parents=True, exist_ok=True)
                payload = {
                    "algorithm": algorithm,
                    "split": "test",
                    "intensity": "medium",
                    "variant": variant,
                    "episodes": 3,
                    "mean_average_waiting_time_mean": waiting_time,
                    "mean_average_waiting_time_std": 0.5,
                    "mean_average_pedestrian_waiting_time_mean": 12.0 if variant == "peds" else None,
                    "mean_average_pedestrian_waiting_time_std": 0.4 if variant == "peds" else None,
                    "mean_average_speed_mean": 5.0,
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

            html = outputs["html"].read_text(encoding="utf-8")
            self.assertIn("Vehicle Only | Medium", html)
            self.assertIn("Pedestrians | Medium", html)
            self.assertIn("Pedestrian waiting time", html)

            with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["variant"] for row in rows}, {"vehicle", "peds"})
            by_key = {(row["algorithm"], row["variant"]): row for row in rows}
            self.assertAlmostEqual(
                float(by_key[("shared_ppo", "vehicle")]["mean_average_waiting_time_improvement_pct"]),
                30.0,
            )
            self.assertAlmostEqual(
                float(by_key[("shared_ppo", "peds")]["mean_average_waiting_time_improvement_pct"]),
                (14.0 - 9.0) / 14.0 * 100.0,
            )


if __name__ == "__main__":
    unittest.main()
