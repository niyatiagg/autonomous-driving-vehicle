import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import highway_env
import sys

vec_env = make_vec_env('intersection-v0', n_envs=1)
model = A2C("MlpPolicy", vec_env, verbose=2, device="cpu")

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "a2c_intersection_grayscale"

model.load(filename)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")