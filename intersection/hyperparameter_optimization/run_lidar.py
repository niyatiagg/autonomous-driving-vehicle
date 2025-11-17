import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import highway_env
from matplotlib import pyplot as plt
import sys

vec_env = make_vec_env('intersection-v1', n_envs=1,env_kwargs={"config": {"observation": {"type": "LidarObservation"}}})
model = PPO("MlpPolicy", vec_env, verbose=2, device="cpu")

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "best/best_model.zip"

model.load(filename)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")