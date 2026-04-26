import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RocketLandingEnv(gym.Env):
    def __init__(self):
        super(RocketLandingEnv, self).__init__()

        # State: [x, y, vx, vy]
        self.observation_space = spaces.Box(
            low=np.array([-100, 0, -10, -10], dtype=np.float32),
            high=np.array([100, 100, 10, 10], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: thrust (0 to 1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.gravity = -9.8
        self.dt = 0.1

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = np.random.uniform(-10, 10)
        self.y = np.random.uniform(50, 100)
        self.vx = 0.0
        self.vy = 0.0

        state = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        return state, {}

    def step(self, action):
        thrust = float(action[0]) * 20.0

        # Physics update
        self.vy += (thrust + self.gravity) * self.dt
        self.y += self.vy * self.dt

        # Horizontal drift
        self.vx += np.random.uniform(-0.1, 0.1)
        self.x += self.vx * self.dt

        # Distance to landing pad (0, 0)
        distance = np.sqrt(self.x**2 + self.y**2)

        # Reward V3:
        # encourage closeness to pad, soft vertical landing, and reduced side drift
        reward = (
            -distance
            - 0.5 * abs(self.vy)
            - 0.3 * abs(self.x)
            - 0.2 * abs(self.vx)
        )

        done = False

        # Terminate when rocket hits ground
        if self.y <= 0:
            done = True

            # Successful landing condition
            if abs(self.x) < 2 and abs(self.vy) < 2 and self.y <= 0:
                reward += 100
            else:
                reward -= 100

        state = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        return state, reward, done, False, {}

    def render(self):
        print(f"x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f}")