"""Небольшая среда маршрутизации с опасной зоной и сбоями привода."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw


class WarehouseEnv(gym.Env):
    """Робот на сетке 7 x 11: координаты -> действие -> новая клетка."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}
    rows, cols = 7, 11
    goal = (6, 10)
    train_starts = ((6, 0), (5, 0), (4, 0))
    shifted_starts = ((3, 0), (2, 0), (2, 1))
    # Вверх, вправо, вниз, влево; при сбое поворот на +/-90 градусов.
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, slip=0.0, max_steps=80, render_mode=None):
        super().__init__()
        if not 0 <= slip <= 1:
            raise ValueError("Вероятность проскальзывания должна быть в [0, 1].")
        self.slip = float(slip)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(self.rows * self.cols)
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def is_hazard(position):
        return position[0] == 6 and 1 <= position[1] <= 9

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        starts = self.train_starts
        start = (options or {}).get("start")
        if start is None:
            start = starts[int(self.np_random.integers(len(starts)))]
        self.position = tuple(start)
        if (not 0 <= self.position[0] < self.rows
                or not 0 <= self.position[1] < self.cols
                or self.is_hazard(self.position) or self.position == self.goal):
            raise ValueError("Начальная клетка должна быть свободной и не целевой.")
        self.steps = 0
        return self.position[0] * self.cols + self.position[1], {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Неизвестное действие.")
        # Один равномерный шум на шаг: одинаковые seed дают общие случайные числа.
        noise = self.np_random.random()
        actual = int(action)
        if noise < self.slip / 2:
            actual = (actual - 1) % 4
        elif noise < self.slip:
            actual = (actual + 1) % 4
        dr, dc = self.moves[actual]
        row, col = self.position
        self.position = (
            min(max(row + dr, 0), self.rows - 1),
            min(max(col + dc, 0), self.cols - 1),
        )
        self.steps += 1
        success = self.position == self.goal
        hazard = self.is_hazard(self.position)
        terminated = success or hazard
        truncated = self.steps >= self.max_steps and not terminated
        # Каждый шаг стоит 1; доставка даёт +30, авария -50 дополнительно.
        reward = -1.0 + 30.0 * success - 50.0 * hazard
        info = {"success": success, "hazard": hazard, "actual_action": actual}
        observation = self.position[0] * self.cols + self.position[1]
        return observation, reward, terminated, truncated, info

    def render(self):
        size = 44
        canvas = Image.new("RGB", (self.cols * size, self.rows * size), "#edf2f7")
        draw = ImageDraw.Draw(canvas)
        for row in range(self.rows):
            for col in range(self.cols):
                box = (col * size, row * size, (col + 1) * size, (row + 1) * size)
                color = "#ee7777" if self.is_hazard((row, col)) else "#edf2f7"
                if (row, col) == self.goal:
                    color = "#66c5a3"
                draw.rectangle(box, fill=color, outline="#adb5bd")
        row, col = self.position
        draw.ellipse((col * size + 9, row * size + 9,
                      (col + 1) * size - 9, (row + 1) * size - 9), fill="#315cb4")
        return np.asarray(canvas)
