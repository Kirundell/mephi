"""Проверки механики среды, от которой зависит корректность эксперимента."""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from warehouse_env import WarehouseEnv


class EnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        check_env(WarehouseEnv(), skip_render_check=True)

    def test_hazard_and_goal(self):
        env = WarehouseEnv()
        env.reset(seed=1, options={"start": (6, 0)})
        _, reward, terminated, truncated, info = env.step(1)
        self.assertEqual((reward, terminated, truncated, info["hazard"]),
                         (-51.0, True, False, True))
        env.reset(seed=1, options={"start": (5, 10)})
        _, reward, terminated, truncated, info = env.step(2)
        self.assertEqual((reward, terminated, truncated, info["success"]),
                         (29.0, True, False, True))

    def test_boundary_and_time_limit(self):
        env = WarehouseEnv(max_steps=1)
        obs, _ = env.reset(seed=1, options={"start": (0, 0)})
        next_obs, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual((obs, next_obs, reward, terminated, truncated),
                         (0, 0, -1.0, False, True))

    def test_seed_replays_noise(self):
        envs = [WarehouseEnv(slip=0.4), WarehouseEnv(slip=0.4)]
        for env in envs:
            env.reset(seed=123, options={"start": (0, 0)})
        for action in [0, 1, 2, 3] * 8:
            results = [env.step(action) for env in envs]
            self.assertEqual(results[0], results[1])
            if results[0][2] or results[0][3]:
                for env in envs:
                    env.reset(options={"start": (0, 0)})

    def test_slip_probabilities(self):
        env = WarehouseEnv(slip=0.4)
        env.reset(seed=321, options={"start": (0, 0)})
        actual = []
        for _ in range(10000):
            env.reset(options={"start": (0, 0)})
            actual.append(env.step(1)[4]["actual_action"])
        rates = np.bincount(actual, minlength=4) / len(actual)
        np.testing.assert_allclose(rates, [0.2, 0.6, 0.2, 0], atol=0.02)

    def test_render_and_invalid_start(self):
        env = WarehouseEnv(render_mode="rgb_array")
        env.reset(seed=1)
        self.assertEqual(env.render().shape, (308, 484, 3))
        with self.assertRaises(ValueError):
            env.reset(options={"start": (6, 4)})


if __name__ == "__main__":
    unittest.main()
