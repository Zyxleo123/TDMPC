import os
import sys

from collections import deque
import numpy as np
import gymnasium as gym

from tdmpc2.envs.wrappers.time_limit import TimeLimit


### TorchDriveEnv ###
os.environ["IAI_API_KEY"] = ""
#####################


class DriveenvWrapper(gym.Wrapper):
    # NOTE: currently rendering do not rotate with the ego vehicle torchdriveenv-master/torchdriveenv/gym_env.py
    def __init__(self, env, cfg, frame_stack=3):
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.frame_stack = frame_stack
        self.observation_stack = deque([], maxlen=self.frame_stack)
        # Observation space is (C*frame_stack, H, W) where C=3 for RGB
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(3 * self.frame_stack, 64, 64), 
            dtype=np.float32
        )

    def _get_stacked_obs(self):
        """Stack frames along the channel dimension."""
        assert len(self.observation_stack) == self.frame_stack
        # Stack along channel dimension: (frame_stack, 3, 64, 64) -> (frame_stack*3, 64, 64)
        return np.concatenate(list(self.observation_stack), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs.astype(np.float32)
        # Fill the stack with the initial observation
        for _ in range(self.frame_stack):
            self.observation_stack.append(obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        self.observation_stack.append(obs)
        info["success"] = info["is_success"]
        return self._get_stacked_obs(), reward, done, truncated, info
    
    def render(self, *args, **kwargs):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_env(cfg):
    """
    Make Humanoid environment.
    """
    print('\ncfg.task', cfg.task)
    if not cfg.task.startswith("driveenv"):
        raise ValueError("Unknown task:", cfg.task)
    with open(os.path.join(os.path.dirname(__file__), 'iai_api_key.txt'), 'r') as f:
        iai_api_key = f.read().strip()
    os.environ["IAI_API_KEY"] = iai_api_key
    
    import torchdriveenv
    from torchdriveenv.env_utils import load_default_train_data, load_default_validation_data
    from torchdriveenv.env_utils import construct_env_config
    import invertedai as iai
    training_data = load_default_train_data()
    validation_data = load_default_validation_data()

    ego_only = False if "multi_agent" in cfg.task else True
    frame_stack = 3
    env_config = {
        "ego_only": ego_only,
        "frame_stack": frame_stack,
        "waypoint_bonus": cfg.waypoint_bonus,
        "heading_penalty": cfg.heading_penalty,
        "distance_bonus": cfg.distance_bonus,
        "distance_cutoff": cfg.distance_cutoff,
    }
    env_config = construct_env_config(env_config)
    print("env_config", env_config)

    env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': training_data})
    env = DriveenvWrapper(env, cfg, frame_stack=frame_stack)
    env.max_episode_steps = 1000#env.get_wrapper_attr("_max_episode_steps")
    return env
