
import gymnasium as gym
import highway_env

def make_merge_lidar_env(render_mode=None):
    config = {
        "observation": {
            "type": "LidarObservation",
            "cells": 64,
            "maximum_range": 60,
            "normalize": True,
        }
    }
    return gym.make("merge-v0", render_mode=render_mode, config=config)

def test_environment():
    env = make_merge_lidar_env(render_mode="human") 
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")  
    print(f"Action space: {env.action_space}")
    
    for _ in range(100):  
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    test_environment()
