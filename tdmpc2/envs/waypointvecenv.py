import os
import sys

import numpy as np
import gymnasium as gym

from tdmpc2.envs.wrappers.time_limit import TimeLimit


### WaypointSuiteEnvVecRep ###
# API key for invertedai - you may need to set this based on your setup
# os.environ["IAI_API_KEY"] = "YOUR_API_KEY_HERE"
#####################


class WaypointVecWrapper(gym.Wrapper):
    """
    Wrapper for WaypointSuiteEnvVecRep to make it compatible with TDMPC2.
    The environment provides vectorized scene representation for autonomous driving.
    """
    def __init__(self, env, cfg):
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
        self._default_scenario = getattr(cfg, 'scenario', 0)
        
        # WaypointSuiteEnvVecRep has observation_space of shape (num_objects, num_past_steps-1, n_dim)
        # Default: (151, 9, 10) where 151 = 50 agents + 50 lanes + 50 traffic_lights + 1 waypoint
        # Keep the original observation space from the environment
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        # Set scenario on EVERY reset (BaseEnv sets it to None after episode ends)
        self.env.unwrapped.set_scenario(self._default_scenario)
        obs, info = self.env.reset(**kwargs)
        # Ensure observation is numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        obs = obs.astype(np.float32)
        return obs, info
    
    def set_scenario(self, scenario):
        """Set the driving scenario for subsequent resets."""
        self._default_scenario = scenario

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        # Ensure observation is numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        obs = obs.astype(np.float32)
        info["success"] = info.get("is_success", False)
        return obs, reward, done, truncated, info
    
    def render(self, *args, **kwargs):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_env(cfg):
    """
    Make WaypointSuiteEnvVecRep environment for autonomous driving.
    
    Expected task format: waypointvec-v0 or similar
    """
    print('\ncfg.task', cfg.task)
    if not cfg.task.startswith("waypointvec"):
        raise ValueError("Unknown task:", cfg.task)
    
    # CRITICAL: Set API key BEFORE any imports!
    # InvertedAI verifies the key immediately upon import
    os.environ["IAI_API_KEY"] = "rtH7QGSLDd8mVfqeQRu2t3cBYcm51hgw4o6jdYQG"
    
    # Import the necessary modules from vlm-clean
    # Adjust the path based on where vlm-clean is located relative to tdmpc2
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../dependency/vlm-clean"))
    
    from torchdriveenv.gym_env import EnvConfig
    from torchdriveenv.env_utils import construct_env_config
    from scenarios.waypoint import WaypointSuiteEnvVecRep, SingleAgentWrapper

    os.environ["IAI_API_KEY"] = "rtH7QGSLDd8mVfqeQRu2t3cBYcm51hgw4o6jdYQG" # "BOTALVxftVU51EcBYXo14F2S6cF6xpcEFe0YQb70"
    
    # Set up environment configuration
    # You can adjust these parameters based on your needs
    num_past_steps = 10  # number of past steps to consider for history (1 sec at 10Hz)
    n_max_agent = 50
    n_max_lane = 50
    n_max_traffic_light = 50
    n_dim = 10
    penalty = 100.0
    
    # Create EnvConfig
    # This should match the configuration expected by WaypointSuiteEnvVecRep
    env_config_dict = {
        "ego_only": True,  # single agent control
        "frame_stack": 1,  # vectorized representation doesn't need frame stacking
        "waypoint_bonus": 100.,
        "heading_penalty": 25.,
        "distance_bonus": 1.,
        "distance_cutoff": 0.25,
        "max_environment_steps": 1000,
        "seed": getattr(cfg, 'seed', 0),
        "device": None,  # Will auto-detect CUDA
        "video_res": getattr(cfg, 'video_res', 1024),
        "video_fov": getattr(cfg, 'video_fov', 60),
    }
    env_config = construct_env_config(env_config_dict)
    print("env_config", env_config)
    
    # Create the environment
    env = WaypointSuiteEnvVecRep(
        cfg=env_config,
        penalty=penalty,
        num_past_steps=num_past_steps,
        n_max_agent=n_max_agent,
        n_max_lane=n_max_lane,
        n_max_traffic_light=n_max_traffic_light,
        n_dim=n_dim
    )
    
    # Wrap with SingleAgentWrapper to remove batch/agent dimensions
    env = SingleAgentWrapper(env)
    
    # Wrap with our custom wrapper
    env = WaypointVecWrapper(env, cfg)
    
    # Set max episode steps
    env.max_episode_steps = 1000
    
    return env

