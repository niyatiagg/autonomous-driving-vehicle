import gymnasium as gym
import highway_env
import time

env = gym.make("intersection-v1", render_mode="human", config={
    "manual_control": False,
    "observation": {"type": "LidarObservation"},
    "duration": 20,
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 4.5, 9],}
})
env.reset()
terminated = truncated = False
while not terminated and not truncated:
    time.sleep(0.2)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"observation: {obs}")
    print(f"reward: {reward}")
    print(f"info: {info}")