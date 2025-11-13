import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import highway_env
from matplotlib import pyplot as plt

vec_env = make_vec_env('intersection-v0', n_envs=1, env_kwargs={"config": {"observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    }}})
model = PPO("MlpPolicy", vec_env, verbose=2, device="cpu")
model.load("ppo_intersection_grayscale")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")