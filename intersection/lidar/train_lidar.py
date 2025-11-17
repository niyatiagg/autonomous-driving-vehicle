import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

env_config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 4.5, 9],},
        "observation": {"type": "LidarObservation"},
        "offroad_terminal": True}

def train(n_envs, n_steps, timesteps, load=False):
    env = SubprocVecEnv([lambda: gym.make("intersection-v1", config=env_config) for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("intersection-v1", config=env_config))])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    evaluation = EvalCallback(eval_env, eval_freq=4096 // n_envs, n_eval_episodes=10,
                                log_path=f"./logs", 
                                best_model_save_path=f"./best_models")
    checkpoint = CheckpointCallback(10000 // n_envs, "./checkpoints", verbose=2)

    if load:
        model = PPO.load("lidar_discrete_32768.zip")
        model.set_env(env)
    else:
        model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=n_envs * n_steps // 16, device="cpu", verbose=2,
                    ent_coef=0.001)

    try:
        model.learn(total_timesteps=timesteps, callback=[evaluation, checkpoint], progress_bar=True, reset_num_timesteps=(not load))
    except Exception as e:
        print(e)
    finally:
        model.save(f"lidar_discrete_{evaluation.num_timesteps}")

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
    train(8, 256, 64000, load=True)
    # model = PPO.load("best_models/best_model")
    # test(model)