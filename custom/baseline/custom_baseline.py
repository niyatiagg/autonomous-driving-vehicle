import os
import gymnasium as gym
from custom_env import AccidentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

"""
custom_baseline.py

Train a PPO baseline (stable-baselines3) on a local custom environment file.
The loader will try to import custom-env.py or custom_env.py from the same directory
and instantiate the first gym.Env subclass it finds, or a callable named `make_env`
or `CustomEnv`.

Usage:
    python custom_baseline.py
"""

# Register the custom environment with Gym
try:
    gym.register(
        id="MyCustomEnv-v0",
        entry_point="custom_env:AccidentEnv",
    )
except gym.error.RegistrationError:
    # Handle case where it might already be registered (e.g. during re-runs)
    print("Environment already registered.") 

def main():
        BASE_DIR = os.getcwd()
        # create vectorized training env
        vec_env = make_vec_env("MyCustomEnv-v0", n_envs=2, seed=42)

        # instantiate PPO (tune hyperparams as needed)
        model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                learning_rate=3e-4,
                ent_coef=0.0,
        )

        # train
        total_timesteps = int(200)
        model.learn(total_timesteps=total_timesteps)

        # save
        save_path = os.path.join(BASE_DIR, "ppo_custom_env")
        model.save(save_path)
        print(f"Model saved to: {save_path}.zip")

        log_dir = "./monitor_logs/" # Directory to save monitor logs
        wrapped_env = Monitor(AccidentEnv(), filename=log_dir, allow_early_resets=True)

        # evaluation (use a fresh env instance not wrapped in Vec)
        eval_env = wrapped_env
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
        print(f"Evaluation over 10 episodes: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

        eval_env.close()
        vec_env.close()


if __name__ == "__main__":
        main()