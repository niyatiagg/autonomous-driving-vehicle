import gymnasium as gym
from stable_baselines3 import DQN
import highway_env

env = gym.make("intersection-v0", config={"observation": {"type": "LidarObservation"}})

model = DQN("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=5000, log_interval=50, progress_bar=True)
model.save("dqn_intersection_lidar")