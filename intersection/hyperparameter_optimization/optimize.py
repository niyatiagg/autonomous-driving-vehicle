import optuna
from optuna.trial import Trial
from optuna.pruners import PatientPruner, HyperbandPruner
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium
import highway_env
import pickle
import os

# Custom Callback class that includes pruning
class Evaluator(EvalCallback):

    def __init__(self, eval_env, trial, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trial = trial

    def _on_step(self):
        super()._on_step()

        if self.n_calls % self.eval_freq == 0:
            # Decide whether or not to prune the trial
            self.trial.report(self.best_mean_reward, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            
        return True

# The objective function
def make_objective(studyname, env_name, env_kwargs, timesteps):
    
    def objective(trial: Trial):
        # All the hyperparameters to optimize
        n_envs = trial.suggest_categorical("n_envs", [1, 2, 4, 8, 16])
        # policy = trial.suggest_categorical("policy", ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"])
        policy = "MlpPolicy"
        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, log=True)
        n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_epochs = trial.suggest_int("n_epochs", 5, 10)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        ent_coef = trial.suggest_float("ent_coef", 0, 0.01)
        vf_coef = trial.suggest_float("vf_coef", 0.5, 1)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2, log=True)

        # Create the environment and the model using the hyperparameters
        env = make_vec_env(env_name, n_envs=n_envs, env_kwargs=env_kwargs)
        model = PPO(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range,
                    verbose=1, device="cpu", ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
        
        eval_env = make_vec_env(env_name, n_envs=1, env_kwargs=env_kwargs)
        callback = Evaluator(eval_env, trial, eval_freq=max(100 // n_envs, 1),
                                log_path=f"./logs_{studyname}/trial{trial.number}", 
                                best_model_save_path=f"./best_models_{studyname}/trial{trial.number}")

        try:
            # Train the model
            model.learn(timesteps, callback)

            # Cleanup
            assert model.env is not None
            model.env.close()
            eval_env.close()
            del model.env, eval_env
            del model
        except (AssertionError, ValueError, optuna.TrialPruned) as e:
            # Cleanup
            if model.env is not None:
                model.env.close()
                eval_env.close()
            del model.env, eval_env
            del model
            raise optuna.TrialPruned() from e

        return callback.best_mean_reward
    
    return objective

def run_study(studyname, trials, env_name, env_kwargs, timesteps):
    '''
    Runs an Optuna study. Will load and continue a previous study if a file 
    named {studyname}.pkl exists in the same directory. Otherwise, makes a new study.

    * **studyname** - The name of this study
    * **trials** - The number of trials to run
    * **env_name** - The name of the gymnasium environment
    * **env_kwargs** - Keyword arguments to pass to the environment constructor
    * **timesteps** - The number of timesteps per trial
    '''
    if os.path.isfile(f"{studyname}.pkl"):
        with open(f"{studyname}.pkl", "rb") as file:
            study = pickle.load(file)
    else:
        study = optuna.create_study(study_name=studyname, direction="maximize")

    try:
        study.optimize(make_objective(studyname, env_name, env_kwargs, timesteps), n_trials=trials, show_progress_bar=True)
    except Exception as e:
        print(e)
    finally:
        print("Saving study, do not interrupt...")
        with open(f"{study.study_name}.pkl","wb") as file:
            pickle.dump(study, file)
        print(f"Best parameters: {study.best_params}")
        print(f"Best trial: {study.best_trial}")

if __name__=="__main__":

    env_name = "intersection-v1"
    # env_kwargs = {"config": {"observation": {"type": "LidarObservation"}, "action": {"type": "DiscreteMetaAction"}}}
    env_kwargs = {"config": {"observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    }, "action": {"type": "ContinuousAction"}}}
    timesteps = 8000
    studyname = "intersection_grayscale_continuous_patient"
    trials = 20

    run_study(studyname, trials, env_name, env_kwargs, timesteps)