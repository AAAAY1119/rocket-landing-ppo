import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

NUM_EPISODES = 20
SAVE_DIR = "../figures"

os.makedirs(SAVE_DIR, exist_ok=True)

# Choose version from terminal
# Example: python evaluate.py v1
if len(sys.argv) < 2:
    print("Usage: python evaluate.py v1 OR v2 OR v3")
    sys.exit()

version = sys.argv[1].lower()

if version == "v1":
    from env_v1 import RocketLandingEnv
    model_path = "../models/ppo_v1.zip"
elif version == "v2":
    from env_v2 import RocketLandingEnv
    model_path = "../models/ppo_v2.zip"
elif version == "v3":
    from env_v3 import RocketLandingEnv
    model_path = "../models/ppo_v3.zip"
else:
    print("Invalid version. Use v1, v2, or v3.")
    sys.exit()

env = RocketLandingEnv()
model = PPO.load(model_path)

episode_rewards = []
successes = 0
all_trajectories = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    trajectory = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        total_reward += reward
        trajectory.append((env.x, env.y))

    episode_rewards.append(total_reward)
    all_trajectories.append(trajectory)

    if abs(env.x) < 2 and abs(env.vy) < 2 and env.y <= 0:
        successes += 1

success_rate = successes / NUM_EPISODES
avg_reward = np.mean(episode_rewards)

print(f"Evaluation complete for {version.upper()} over {NUM_EPISODES} episodes")
print(f"Success rate: {success_rate:.2%}")
print(f"Average reward: {avg_reward:.2f}")

# Plot 1: Evaluation rewards
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_EPISODES + 1), episode_rewards, marker="o")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"PPO Rocket Landing Evaluation Rewards ({version.upper()})")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/evaluation_rewards_{version}.png", dpi=300)
plt.show()

# Plot 2: Sample trajectories
plt.figure(figsize=(8, 6))

for i, trajectory in enumerate(all_trajectories[:10]):
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    plt.plot(xs, ys, label=f"Ep {i + 1}", alpha=0.8)

plt.axhline(0, linestyle="--")
plt.scatter(0, 0, marker="x", s=100, label="Landing Pad")
plt.xlabel("Horizontal Position (x)")
plt.ylabel("Vertical Position (y)")
plt.title(f"Sample Rocket Landing Trajectories ({version.upper()})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/sample_trajectories_{version}.png", dpi=300)
plt.show()