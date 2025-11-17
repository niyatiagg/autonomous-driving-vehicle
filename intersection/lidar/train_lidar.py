import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

env_config = {
        "action": {"type": "ContinuousAction"},
        "observation": {"type": "LidarObservation"},
        "reward_speed_range":[1, 9],
        "high_speed_reward": 0,
        "collision_reward": -100,
        "arrived_reward": 50,
        "offroad_terminal": True}

def train(n_envs, n_steps, timesteps, load=False):
    env = SubprocVecEnv([lambda: gym.make("intersection-v1", config=env_config) for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("intersection-v1", config=env_config))])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    evaluation = EvalCallback(eval_env, eval_freq=2048 // n_envs,
                                log_path=f"./logs", 
                                best_model_save_path=f"./best_models")
    checkpoint = CheckpointCallback(10000 // n_envs, "./checkpoints", verbose=2)

    if load:
        model = PPO.load("lidar_continuous.zip")
        model.set_env(env)
    else:
        model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=n_envs * n_steps // 16, device="cpu", verbose=2,
                    ent_coef=0.01)

    try:
        model.learn(total_timesteps=timesteps, callback=[evaluation, checkpoint], progress_bar=True, reset_num_timesteps=(not load))
    except Exception as e:
        print(e)
    finally:
        model.save(f"lidar_continuous_{evaluation.num_timesteps}")

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
    train(8, 256, 32000)
    # model = PPO.load("best_models/best_model")
    # test(model)