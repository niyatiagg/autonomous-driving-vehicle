import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import highway_env
from matplotlib import pyplot as plt

# Parallel environments
vec_env = make_vec_env('intersection-v0', n_envs=4)
model = PPO("MlpPolicy", vec_env, verbose=2, device="cpu", n_steps=256)
model.learn(total_timesteps=5000, progress_bar=True)
model.save("ppo_intersection")