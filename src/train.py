from stable_baselines3 import PPO
from env_v3 import RocketLandingEnv
import os

env = RocketLandingEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

os.makedirs("../models", exist_ok=True)
model.save("../models/ppo_rocket_v3")

print("Training complete. Model saved.")