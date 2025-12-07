# run_trained_custom_simple.py
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "MyCustomEnv-v0"
MODEL_PATH = "./models/ppo_custom_manual_tuned.zip"  # <- fixed as requested

# Register your custom env once
try:
    gym.register(id=ENV_ID, entry_point="custom_env:AccidentEnv")
except gym.error.RegistrationError:
    pass

def run_video(episodes: int, video_dir: str, seed: int = 42, deterministic: bool = True):
    """Record MP4(s) of evaluation episodes (works on Colab)."""
    os.makedirs(video_dir, exist_ok=True)
    # Must support rgb_array rendering for video recording
    env = gym.make(ENV_ID, render_mode="rgb_array")
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda e: True,  # record every episode
        name_prefix="ppo_eval"
    )

    model = PPO.load(MODEL_PATH)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated
        print(f"[VIDEO] Episode {ep+1}/{episodes} — return: {ep_ret:.2f}, length: {ep_len}")

    env.close()
    print(f"Saved video(s) to: {os.path.abspath(video_dir)}")

def run_live(episodes: int, seed: int = 42, deterministic: bool = True):
    """Live demo with a window (works locally; NOT visible on Colab)."""
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(MODEL_PATH)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated
            # human mode usually renders automatically; call render() just in case:
            env.render()
        print(f"[LIVE] Episode {ep+1}/{episodes} — return: {ep_ret:.2f}, length: {ep_len}")

    env.close()

def main():
    parser = argparse.ArgumentParser(description="Run PPO on MyCustomEnv-v0")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-dir", type=str, help="Record MP4(s) to this folder (Colab-friendly).")
    group.add_argument("--live", action="store_true", help="Open a native window (not visible on Colab).")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.video_dir:
        run_video(episodes=args.episodes, video_dir=args.video_dir,
                  seed=args.seed, deterministic=args.deterministic)
    else:
        run_live(episodes=args.episodes, seed=args.seed,
                 deterministic=args.deterministic)

if __name__ == "__main__":
    main()
