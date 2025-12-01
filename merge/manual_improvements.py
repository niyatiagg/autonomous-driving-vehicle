
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import make_merge_lidar_env

configs = [
    {
        "name": "baseline_200k",
        "lr": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "net_arch": [64, 64],
        "log_name": "manual_baseline_200k"
    },
    {
        "name": "large_net",
        "lr": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "net_arch": [256, 256],
        "log_name": "manual_large_net_200k"
    },
    {
        "name": "fast_learning",
        "lr": 5e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "net_arch": [128, 128],
        "log_name": "manual_fast_lr_200k"
    }
]

TIMESTEPS_PER_CONFIG = 200_000

for config in configs:
    print(f"\n=== Training {config['name']} ===")
    
    env = Monitor(make_merge_lidar_env(render_mode=None))
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": config["net_arch"]},
        verbose=1,
        tensorboard_log="./ppo_merge_tensorboard",  
    )
    
    model.learn(
        total_timesteps=TIMESTEPS_PER_CONFIG,
        tb_log_name=config["log_name"]  
    )
    
    # Save model
    model.save(f"./models/{config['name']}.zip")
    print(f"Saved {config['name']} to ./models/{config['name']}.zip")

print("\nDone! Check TensorBoard:")
print("tensorboard --logdir ./ppo_merge_tensorboard")
