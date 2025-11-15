import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

env_config = {
        "action": {"type": "ContinuousAction"},
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": 1,
            "weights": [0.2989, 0.5870, 0.1140]  # weights for RGB conversion
        }}

def train(n_envs, n_steps, timesteps, load=False):
    env = SubprocVecEnv([lambda: gym.make("intersection-v1", config=env_config) for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([lambda: Monitor(gym.make("intersection-v1", config=env_config))])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    evaluate = EvalCallback(eval_env, eval_freq=n_steps // n_envs,
                                log_path=f"./logs", 
                                best_model_save_path=f"./best_models")
    checkpoint = CheckpointCallback(10000 // n_envs, "./checkpoints", verbose=2)

    if load:
        model = PPO.load("grayscale_continuous.zip")
        model.set_env(env)
    else:
        model = PPO("CnnPolicy", env, n_steps=n_steps, batch_size=n_envs * n_steps // 16, verbose=2)

    try:
        model.learn(total_timesteps=timesteps, callback=[evaluate, checkpoint], progress_bar=True, reset_num_timesteps=(not load))
    except Exception as e:
        print(e)
    finally:
        model.save(f"grayscale_continuous_{callback.num_timesteps}")


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
    train(8, 1024, 320000, load=True)