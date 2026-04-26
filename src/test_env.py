from env import RocketLandingEnv

env = RocketLandingEnv()

state, _ = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    env.render()

    if done:
        print("Episode finished")
        break