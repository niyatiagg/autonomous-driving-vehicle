import gymnasium as gym
from stable_baselines3 import DQN
import highway_env
import sys

env = gym.make("intersection-v0", render_mode="human")

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "dqn_intersection"

model = DQN.load(filename)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()