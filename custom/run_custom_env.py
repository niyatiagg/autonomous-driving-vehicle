import gymnasium as gym
import highway_env
import time
from custom_env import AccidentEnv

gym.register("accident-v0", "custom_env:AccidentEnv")

env = gym.make("accident-v0", render_mode="human", config={
    "manual_control": True,
    "observation": {"type": "LidarObservation"},
    "duration": 20
})
env.reset()
terminated = truncated = False
while not terminated and not truncated:
    time.sleep(0.2)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"observation: {obs}")
    print(f"reward: {reward}")
    print(f"info: {info}")