import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
import sys

env_config = {
        "action": {"type": "ContinuousAction"},
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": 1,
            "weights": [0.2989, 0.5870, 0.1140]  # weights for RGB conversion
        }}

def test(model: PPO):
    test_env = DummyVecEnv([lambda: gym.make("intersection-v1", config=env_config, render_mode="human")])
    test_env = VecFrameStack(test_env, 4)
    obs = test_env.reset()

    runs = 0
    while runs < 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render("human")
        if done:
            runs += 1
            obs = test_env.reset()

if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "grayscale_continuous"

    model = PPO.load(filename)
    test(model)