import os
import sys

from collections import deque
import numpy as np
import torch
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
    frame_stack = cfg.frame_stack
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


class _EnvFactory:
    """Picklable factory for creating torchdriveenv instances in subprocesses."""

    def __init__(self, env_config, training_data, frame_stack, iai_api_key):
        self.env_config = env_config
        self.training_data = training_data
        self.frame_stack = frame_stack
        self.iai_api_key = iai_api_key

    def __call__(self):
        import os
        import gymnasium as gym
        os.environ["IAI_API_KEY"] = self.iai_api_key
        env = gym.make('torchdriveenv-v0', args={'cfg': self.env_config, 'data': self.training_data})
        env = DriveenvWrapper(env, None, frame_stack=self.frame_stack)
        env.max_episode_steps = 1000
        return env


class VecDriveEnvWrapper:
    """
    Vectorized wrapper for N parallel torchdriveenv instances via gymnasium AsyncVectorEnv.
    Provides a torch-tensor I/O interface compatible with OnlineTrainer.
    """

    def __init__(self, env_fns, eval_env, cfg):
        import gymnasium.vector as gvec
        self.vec_env = gvec.AsyncVectorEnv(env_fns)
        self.n_envs = len(env_fns)
        self.cfg = cfg
        self.eval_env = eval_env  # single TensorWrapper-wrapped env for evaluation
        self.observation_space = self.vec_env.single_observation_space
        self.action_space = self.vec_env.single_action_space
        self.max_episode_steps = eval_env.max_episode_steps
        self._current_obs = None  # (n_envs, *obs_shape) torch tensor, updated after each step

    def rand_act(self):
        """Return (n_envs, action_dim) random actions as a torch tensor."""
        acts = np.array([self.action_space.sample() for _ in range(self.n_envs)], dtype=np.float32)
        return torch.from_numpy(acts)

    def reset(self):
        """Reset all envs. Returns (n_envs, *obs_shape) torch tensor."""
        obs_np, info = self.vec_env.reset()
        self._current_obs = torch.from_numpy(obs_np.astype(np.float32))
        return self._current_obs, info

    def step(self, actions):
        """
        Step all envs with (n_envs, action_dim) torch actions.

        Returns (terminal_obs, rewards, dones, truncateds, infos) where:
          - terminal_obs[i] is the TERMINAL observation for done envs (pre-reset),
            and the normal next-obs for running envs.
          - self._current_obs is updated to the post-reset obs (ready for next step).
        """
        obs_np, rewards_np, dones, truncateds, infos = self.vec_env.step(actions.cpu().numpy())
        # obs_np[i] = auto-reset (new episode) obs for done[i]; step obs otherwise.
        # Recover terminal obs for done envs from infos["final_observation"].
        terminal_obs_np = obs_np.copy()
        final_obs = infos.get("final_observation", None)
        final_mask = infos.get("_final_observation", None)
        if final_obs is not None and final_mask is not None:
            for i in range(self.n_envs):
                if final_mask[i] and final_obs[i] is not None:
                    terminal_obs_np[i] = final_obs[i]
        self._current_obs = torch.from_numpy(obs_np.astype(np.float32))
        terminal_obs = torch.from_numpy(terminal_obs_np.astype(np.float32))
        rewards = torch.from_numpy(rewards_np.astype(np.float32))
        return terminal_obs, rewards, dones, truncateds, infos

    def get_success(self, infos, env_idx, done):
        """Extract the success flag for env_idx from vectorized infos."""
        if done:
            final_info = infos.get("final_info", None)
            if final_info is not None and final_info[env_idx] is not None:
                return float(final_info[env_idx].get("success", False))
        # Fall back to per-step info (may not be present for running envs)
        success_arr = infos.get("success", None)
        if success_arr is not None:
            return float(success_arr[env_idx])
        return 0.0

    def close(self):
        self.vec_env.close()
        if self.eval_env is not None:
            self.eval_env.close()


def make_parallel_env(cfg, n_envs, eval_env):
    """
    Create a VecDriveEnvWrapper with n_envs parallel torchdriveenv instances.
    eval_env should be a single TensorWrapper-wrapped env for evaluation.
    """
    if not cfg.task.startswith("driveenv"):
        raise ValueError("Unknown task:", cfg.task)

    with open(os.path.join(os.path.dirname(__file__), 'iai_api_key.txt'), 'r') as f:
        iai_api_key = f.read().strip()
    os.environ["IAI_API_KEY"] = iai_api_key

    import torchdriveenv
    from torchdriveenv.env_utils import load_default_train_data, construct_env_config

    ego_only = False if "multi_agent" in cfg.task else True
    frame_stack = cfg.frame_stack
    env_config_raw = {
        "ego_only": ego_only,
        "frame_stack": frame_stack,
        "waypoint_bonus": cfg.waypoint_bonus,
        "heading_penalty": cfg.heading_penalty,
        "distance_bonus": cfg.distance_bonus,
        "distance_cutoff": cfg.distance_cutoff,
    }
    env_config = construct_env_config(env_config_raw)
    training_data = load_default_train_data()

    factory = _EnvFactory(env_config, training_data, frame_stack, iai_api_key)
    env_fns = [factory] * n_envs

    return VecDriveEnvWrapper(env_fns, eval_env, cfg)
