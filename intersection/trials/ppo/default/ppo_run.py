import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import highway_env
from matplotlib import pyplot as plt

vec_env = make_vec_env('intersection-v0', n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=2, device="cpu")
model.load("ppo_intersection")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")