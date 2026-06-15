from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.config import load_experiment_config, load_yaml


class PPOConfigTest(unittest.TestCase):
    def test_ppo_approaches_share_training_budget(self) -> None:
        common_budget = load_yaml("configs/ppo_common.yaml")["train"]["total_timesteps"]
        approach_configs = [
            "configs/ppo_centralized.yaml",
            "configs/ppo_independent.yaml",
            "configs/ppo_shared.yaml",
            "configs/ppo_mappo.yaml",
        ]

        for path in approach_configs:
            with self.subTest(path=path):
                self.assertNotIn("total_timesteps", load_yaml(path).get("train", {}))
                config = load_experiment_config("configs/ppo_common.yaml", path)
                self.assertEqual(config["train"]["total_timesteps"], common_budget)


if __name__ == "__main__":
    unittest.main()
