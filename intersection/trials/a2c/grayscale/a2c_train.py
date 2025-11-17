import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import highway_env

# Parallel environments
# GrayscaleObservation config taken from the documentation's Example Configuration
vec_env = make_vec_env('intersection-v0', n_envs=4, env_kwargs={"config": {"observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    }}})
model = A2C("MlpPolicy", vec_env, verbose=2, device="cpu")
model.learn(total_timesteps=5000, progress_bar=True, log_interval=50)
model.save("a2c_intersection_grayscale")