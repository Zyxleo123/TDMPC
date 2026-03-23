# TODO: modify TDMPC2 using offline RL policy update.
import numpy as np
import torch
import torch.nn.functional as F

from tdmpc2.common import math
from tdmpc2.common.scale import RunningScale
from tdmpc2.common.world_model import WorldModel

from copy import deepcopy


class TDMPC2:
	"""
	Modified TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	Add-ons: Using AWAC for actor learning, lower distributional lower distributional shift.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam(
			[
				{
					"params": self.model._encoder.parameters(),
					"lr": self.cfg.lr * self.cfg.enc_lr_scale,
				},
				{"params": self.model._dynamics.parameters()},
				{"params": self.model._reward.parameters()},
				{"params": self.model._Qs.parameters()},
				{
					"params": self.model._task_emb.parameters()
					if self.cfg.multitask
					else []
				},
			],
			lr=self.cfg.lr,
		)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.log_pi_scale = RunningScale(cfg)	
		self.cfg.iterations += 2 * int(
			cfg.action_dim >= 20
		)  # Heuristic for large action spaces
		self.discount = (
			torch.tensor(
				[self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
				device="cuda",
			)
			if self.cfg.multitask
			else self._get_discount(cfg.episode_length)
		)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
				episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
				float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(
			max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
		)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
				fp (str): Filepath to save state dict to.
		"""
		torch.save({
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"pi_optim": self.pi_optim.state_dict(),
		}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
				fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])
		if "optim" in state_dict:
			self.optim.load_state_dict(state_dict["optim"])
		if "pi_optim" in state_dict:
			self.pi_optim.load_state_dict(state_dict["pi_optim"])

	@torch.no_grad()
	def predict_reward(self, obs, action, task=None, info_latent=None):
		"""
		Predict reward for a given observation and action.
		"""
		if self.cfg.get('use_info_latent', False) and info_latent is not None:
			z = info_latent.to(self.device).unsqueeze(0)
		else:
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
			if task is not None:
				task = torch.tensor([task], device=self.device)
			z = self.model.encode(obs, task)
		
		# Ensure action is a tensor on the correct device
		if not isinstance(action, torch.Tensor):
			action = torch.tensor(action, device=self.device)
		else:
			action = action.to(self.device)
			
		if action.ndim == 1:
			action = action.unsqueeze(0)

		if self.cfg.num_bins > 1:
			reward = math.two_hot_inv(self.model.reward(z, action, task), self.cfg)
		else:
			reward = self.model.reward(z, action, task)
		
		return reward.item()
	
	@torch.no_grad()
	def predict_value(self, obs, action, task=None, info_latent=None):
		"""
		Predict value for a given observation and action.
		"""
		if self.cfg.get('use_info_latent', False) and info_latent is not None:
			z = info_latent.to(self.device).unsqueeze(0)
		else:
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
			if task is not None:
				task = torch.tensor([task], device=self.device)
			z = self.model.encode(obs, task)
		
		# Ensure action is a tensor on the correct device
		if not isinstance(action, torch.Tensor):
			action = torch.tensor(action, device=self.device)
		else:
			action = action.to(self.device)
			
		if action.ndim == 1:
			action = action.unsqueeze(0)
            
		q = self.model.Q(z, action, task, return_type="avg")
		
		return q.item()

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_pi=False, return_plan=False, info_latent=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
				obs (torch.Tensor): Observation from the environment.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (int): Task index (only used for multi-task experiments).
				info_latent (torch.Tensor): 4-dim info latent (used when use_info_latent=True).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		if self.cfg.get('use_info_latent', False) and info_latent is not None:
			z = info_latent.to(self.device).unsqueeze(0)
		else:
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
			if task is not None:
				task = torch.tensor([task], device=self.device)
			z = self.model.encode(obs, task)
		# When use_info_latent is on, dynamics are bypassed so MPC is not viable
		force_pi = self.cfg.get('use_info_latent', False)
		if self.cfg.mpc and not use_pi and not force_pi:
			if return_plan:
				a, mu, std, plan_info = self.plan(z, t0=t0, eval_mode=eval_mode, task=task, return_plan=True)
			else:
				a, mu, std = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		else:
			mu, pi, log_pi, log_std = self.model.pi(z, task)
			if eval_mode:
				a = mu[0]
			else:
				a = pi[0]
			mu, std = mu[0], log_std.exp()[0]
			plan_info = None

		if return_plan:
			return a.cpu(), mu.cpu(), std.cpu(), plan_info

		if eval_mode:
			return a.cpu()
		else:	
			return a.cpu(), mu.cpu(), std.cpu()

	@torch.no_grad()
	def act_vec(self, obs_batch, t0_flags, eval_mode=False, task=None, use_pi_flags=None, info_latents=None):
		"""Batched action selection for n_envs environments.

		Fuses the encoder forward pass into a single GPU call, then runs
		planning (or policy) per-env using the pre-computed latents.

		Args:
			obs_batch (torch.Tensor): shape (n_envs, *obs_shape)
			t0_flags (list[bool]): per-env first-step flag
			eval_mode (bool): deterministic actions
			task: task index (multi-task only)
			use_pi_flags (list[bool]): per-env flag to use policy instead of MPC
			info_latents (torch.Tensor): shape (n_envs, 4), used when use_info_latent=True

		Returns:
			actions (n_envs, action_dim), mus (n_envs, action_dim), stds (n_envs, action_dim)
		"""
		n = obs_batch.shape[0]
		if use_pi_flags is None:
			use_pi_flags = [False] * n

		# --- single batched encode (or use info latents directly) ---
		if self.cfg.get('use_info_latent', False) and info_latents is not None:
			z_batch = info_latents.to(self.device)  # (n, latent_dim=4)
			task_tensor = torch.tensor([task] * n, device=self.device) if task is not None else None
		else:
			obs_batch = obs_batch.to(self.device, non_blocking=True)
			if task is not None:
				task_tensor = torch.tensor([task] * n, device=self.device)
			else:
				task_tensor = None
			z_batch = self.model.encode(obs_batch, task_tensor)  # (n, latent_dim)

		actions = torch.empty(n, self.cfg.action_dim)
		mus     = torch.empty(n, self.cfg.action_dim)
		stds    = torch.empty(n, self.cfg.action_dim)

		# When use_info_latent is on, dynamics are bypassed so all envs use pi
		force_pi = self.cfg.get('use_info_latent', False)
		mpc_idxs = [i for i in range(n) if self.cfg.mpc and not use_pi_flags[i] and not force_pi]
		pi_idxs  = [i for i in range(n) if not (self.cfg.mpc and not use_pi_flags[i] and not force_pi)]

		# --- batch pi envs (single model.pi forward) ---
		if pi_idxs:
			z_pi = z_batch[pi_idxs]
			task_pi = task_tensor[pi_idxs] if task_tensor is not None else None
			mu_all, pi_all, _, log_std_all = self.model.pi(z_pi, task_pi)
			for local_j, global_i in enumerate(pi_idxs):
				a = mu_all[local_j] if eval_mode else pi_all[local_j]
				actions[global_i] = a.cpu()
				mus[global_i]     = mu_all[local_j].cpu()
				stds[global_i]    = log_std_all.exp()[local_j].cpu()

		# --- MPC envs: fully batched planning ---
		if mpc_idxs:
			mpc_idx_t = torch.tensor(mpc_idxs, device=self.device)
			# Lazily init or resize persistent prev_mean (n_envs, H, action_dim)
			if not hasattr(self, '_prev_mean_vec') or self._prev_mean_vec.shape[0] != n:
				self._prev_mean_vec = torch.zeros(
					n, self.cfg.horizon, self.cfg.action_dim, device=self.device
				)

			z_mpc       = z_batch[mpc_idx_t]
			task_mpc    = task_tensor[mpc_idx_t] if task_tensor is not None else None
			prev_mean   = self._prev_mean_vec[mpc_idx_t]
			t0_mpc      = [t0_flags[i] for i in mpc_idxs]

			a_mpc, mu_mpc, std_mpc, new_mean = self._plan(
				z_mpc, t0_mpc, eval_mode=eval_mode, task=task_mpc, prev_mean=prev_mean
			)
			self._prev_mean_vec[mpc_idx_t] = new_mean

			for local_j, global_i in enumerate(mpc_idxs):
				actions[global_i] = a_mpc[local_j].cpu()
				mus[global_i]     = mu_mpc[local_j].cpu()
				stds[global_i]    = std_mpc[local_j].cpu()

		return actions, mus, stds

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, horizon, eval_mode=False, return_details=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		rewards = []
		for t in range(horizon):
			if self.cfg.num_bins > 1:
				reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			else:
				reward = self.model.reward(z, actions[t], task)
			z = self.model.next(z, actions[t], task)
			if return_details:
				rewards.append(reward)
			G += discount * reward
			discount *= (
				self.discount[torch.tensor(task)]
				if self.cfg.multitask
				else self.discount
			)
		q = self.model.Q(
			z, self.model.pi(z, task)[1], task, return_type="avg"
		)
		if return_details:
			return G + discount * q, torch.stack(rewards), q
		return G + discount * q

	@torch.no_grad()
	def _estimate_value_parallel(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			if self.cfg.num_bins > 1:
				reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
			else:
				reward = self.model.reward(z, actions[:, t], task)
			z = self.model.next(z, actions[:, t], task)
			G = G + discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None, return_plan=False):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
				z (torch.Tensor): Latent state from which to plan.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		num_pi_trajs = self.cfg.num_pi_trajs
		
		# Define effective horizon based on frame skipping
		if self.cfg.horizon % self.cfg.time_chunk_size != 0:
			raise ValueError(f"Horizon ({self.cfg.horizon}) must be divisible by time_chunk_size ({self.cfg.time_chunk_size})")
		time_chunk_size = self.cfg.time_chunk_size
		H_eff = self.cfg.horizon // time_chunk_size


		if num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon,
				num_pi_trajs,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]
		else:
			pi_actions = None

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(H_eff, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(
			H_eff, self.cfg.action_dim, device=self.device
		)
		if not t0:
			# Shift mean by 1 unit in the compressed space approx., or reset? 
			# Standard shifting logic applies to raw time steps. 
			# With skipping, shifting by 1 index means shifting by 'skip' time steps.
			# For now, simplistic reuse:
			mean[:-1] = self._prev_mean[1:]
			
		actions = torch.empty(
			self.cfg.horizon,
			self.cfg.num_samples,
			self.cfg.action_dim,
			device=self.device,
		)
		if num_pi_trajs > 0:
			actions[:, : num_pi_trajs] = pi_actions

		# Iterate MPPI
		plan_metrics = []
		for _ in range(self.cfg.iterations):
			# Sample unique actions (effective horizon)
			actions_eff = (
				mean.unsqueeze(1)
				+ std.unsqueeze(1)
				* torch.randn(
					H_eff,
					self.cfg.num_samples - num_pi_trajs,
					self.cfg.action_dim,
					device=std.device,
				)
			).clamp(-1, 1)
			
			# Expand actions to full horizon by repeating each action 'skip' times
			actions_expanded = actions_eff.repeat_interleave(time_chunk_size, dim=0)

			# Store in actions buffer
			actions[:, num_pi_trajs :] = actions_expanded

			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			if return_plan:
				value, rewards, q_values = self._estimate_value(z, actions, task, self.cfg.horizon, return_details=True)
			else:
				value = self._estimate_value(z, actions, task, self.cfg.horizon)

			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0
			).indices
			elite_value = value[elite_idxs]
			
			# Get elite actions and compress back to effective horizon for update
			elite_actions_full = actions[:, elite_idxs]
			elite_actions_eff = elite_actions_full[::time_chunk_size] # Take every k-th action

			# Update parameters using compressed elite actions
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= score.sum(0)

			if return_plan:
				plan_metrics.append({
					"mean": mean.cpu(),
					"std": std.cpu(),
					"values": value.cpu(),
					"elite_idxs": elite_idxs.cpu(),
					"elite_values": elite_value.cpu(),
					"elite_actions": elite_actions_full.cpu(),
					"score": score.cpu(),
					"actions": actions.cpu(),
					"rewards": rewards.cpu(),
					"q_values": q_values.cpu(),
				})

			mean = torch.sum(score.unsqueeze(0) * elite_actions_eff, dim=1) / (
				score.sum(0) + 1e-9
			)
			std = torch.sqrt(
				torch.sum(
					score.unsqueeze(0) * (elite_actions_eff - mean.unsqueeze(1)) ** 2, dim=1
				)
				/ (score.sum(0) + 1e-9)
			).clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(1).cpu().numpy()
		# Sample from full horizon elites, then take first action
		actions_full = elite_actions_full[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		
		# Store compressed mean for next iteration
		self._prev_mean = mean
		
		# Return first action from the full sequence (which corresponds to first action of first block)
		mu, std = actions_full[0], std[0] 
		# Note: std here is from compressed 'std', but roughly applicable for exploration logic
		
		if not eval_mode:
			a = mu + std * torch.randn(self.cfg.action_dim, device=std.device)
		else:
			a = mu
		if return_plan:
			return a.clamp_(-1, 1), mu, std, plan_metrics
		return a.clamp_(-1, 1), mu, std
	
	# @torch.no_grad()
	# def plan_arc(self, z, t0=False, eval_mode=False, task=None):
	# 	"""
	# 	Plan a sequence of actions using the learned world model.
	# 	Actuator Regularized Control (ARC) adapted from LOOP: https://arxiv.org/abs/2008.10066

	# 	Args:
	# 			z (torch.Tensor): Latent state from which to plan.
	# 			t0 (bool): Whether this is the first observation in the episode.
	# 			eval_mode (bool): Whether to use the mean of the action distribution.
	# 			task (Torch.Tensor): Task index (only used for multi-task experiments).

	# 	Returns:
	# 			torch.Tensor: Action to take in the environment.
	# 	"""
	# 	if self.cfg.num_pi_trajs > 0:
	# 		pi_actions = torch.empty(
	# 			self.cfg.horizon,
	# 			self.cfg.num_pi_trajs,
	# 			self.cfg.action_dim,
	# 			device=self.device,
	# 		)
	# 		_z = z.repeat(self.cfg.num_pi_trajs, 1)
	# 		for t in range(self.cfg.horizon - 1):
	# 			pi_actions[t] = self.model.pi(_z, task)[1]
	# 			_z = self.model.next(_z, pi_actions[t], task)
	# 		pi_actions[-1] = self.model.pi(_z, task)[1]

	# 	# Initialize state and parameters
	# 	z = z.repeat(self.cfg.num_samples, 1)
	# 	mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
	# 	std = self.cfg.max_std * torch.ones(
	# 		self.cfg.horizon, self.cfg.action_dim, device=self.device
	# 	)
	# 	if not t0:
	# 		mean[:-1] = self._prev_mean[1:]
	# 	actions = torch.empty(
	# 		self.cfg.horizon,
	# 		self.cfg.num_samples,
	# 		self.cfg.action_dim,
	# 		device=self.device,
	# 	)
	# 	if self.cfg.num_pi_trajs > 0:
	# 		actions[:, : self.cfg.num_pi_trajs] = pi_actions

	# 	# Iterate MPPI
	# 	for _ in range(self.cfg.iterations):
	# 		# Sample actions
	# 		actions[:, self.cfg.num_pi_trajs :] = (
	# 			mean.unsqueeze(1)
	# 			+ std.unsqueeze(1)
	# 			* torch.randn(
	# 				self.cfg.horizon,
	# 				self.cfg.num_samples - self.cfg.num_pi_trajs,
	# 				self.cfg.action_dim,
	# 				device=std.device,
	# 			)
	# 		).clamp(-1, 1)
	# 		if self.cfg.multitask:
	# 			actions = actions * self.model._action_masks[task]

	# 		# Compute elite actions
	# 		value = self._estimate_value(z, actions, task, self.cfg.horizon).nan_to_num_(0)
	# 		elite_idxs = torch.topk(
	# 			value.squeeze(1), self.cfg.num_elites, dim=0
	# 		).indices
	# 		elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

	# 		# Update parameters
	# 		max_value = elite_value.max(0)[0]
	# 		score = torch.exp(self.cfg.temperature * (elite_value - max_value))
	# 		score /= score.sum(0)
	# 		mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
	# 			score.sum(0) + 1e-9
	# 		)
	# 		std = torch.sqrt(
	# 			torch.sum(
	# 				score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1
	# 			)
	# 			/ (score.sum(0) + 1e-9)
	# 		).clamp_(self.cfg.min_std, self.cfg.max_std)
	# 		if self.cfg.multitask:
	# 			mean = mean * self.model._action_masks[task]
	# 			std = std * self.model._action_masks[task]

	# 	# Select action
	# 	score = score.squeeze(1).cpu().numpy()
	# 	actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
	# 	self._prev_mean = mean
	# 	mu, std = actions[0], std[0]
	# 	if not eval_mode:
	# 		a = mu + std * torch.randn(self.cfg.action_dim, device=std.device)
	# 	else:
	# 		a = mu
	# 	return a.clamp_(-1, 1), mu, std
	
	@torch.no_grad()
	def _plan(self, z, t0_flags, eval_mode=False, task=None, prev_mean=None):
		"""
		Batched MPPI planning for num_envs environments in parallel.

		Args:
			z (torch.Tensor): Latent states, shape (num_envs, latent_dim).
			t0_flags (list[bool] or BoolTensor): Per-env first-step flag.
			eval_mode (bool): Deterministic action selection.
			task: Task index (multi-task only).
			prev_mean (torch.Tensor | None): Previous mean, shape (num_envs, horizon, action_dim).

		Returns:
			action (num_envs, action_dim), mu (num_envs, action_dim),
			std (num_envs, action_dim), mean (num_envs, horizon, action_dim)
		"""
		num_envs = z.size(0)
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[:,t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[:,t], task)
			pi_actions[:,-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
		mean = torch.zeros(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		# Per-env warm-starting: only shift mean for envs that are not at t0
		if prev_mean is not None:
			not_t0 = ~torch.as_tensor(t0_flags, dtype=torch.bool, device=self.device)
			if not_t0.any():
				mean[not_t0, :-1] = prev_mean[not_t0, 1:]
		actions = torch.empty(num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(num_envs, self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, :, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value_parallel(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
			elite_actions = torch.gather(actions, 2, elite_idxs.unsqueeze(1).unsqueeze(3).expand(-1, self.cfg.horizon, -1, self.cfg.action_dim))

			# Update parameters
			max_value = elite_value.max(1).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1)))
			score = (score / score.sum(1, keepdim=True))
			mean = (score.unsqueeze(1) * elite_actions).sum(2) / (score.sum(1, keepdim=True) + 1e-9)
			std = ((score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (score.sum(1, keepdim=True) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action: sample one elite per env proportional to score
		rand_idx = torch.multinomial(score.squeeze(2), num_samples=1).squeeze(1)  # (num_envs,)
		actions = elite_actions[torch.arange(num_envs), :, rand_idx]
		action, mu, std_out = actions[:, 0], mean[:, 0], std[:, 0]
		if not eval_mode:
			action = action + std_out * torch.randn_like(action)
		return action.clamp(-1, 1), mu, std_out, mean

	def update_pi(self, zs, action, mu, std, task, il_flag=None):
		"""
		Update policy using a sequence of latent states.

		Args:
				zs (torch.Tensor): Sequence of latent states.
				action (torch.Tensor): Sequence of actions.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type="avg")
		self.scale.update(qs[0])
		qs = self.scale(qs)
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))

		if self.cfg.actor_mode=="sac":
			# TD-MPC2 baseline setting.
			pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = torch.zeros_like(pi_loss) # Not used
			q_loss = pi_loss.detach().clone()

		elif self.cfg.actor_mode=="awac":
			# Loss for AWAC-MPC
			with torch.no_grad():
				vs = self.model.Q(zs, action, task, return_type="avg")
				vs = self.scale(vs)
			adv = (qs - vs).detach()
			weights = torch.clamp(torch.exp(adv / self.cfg.awac_lambda), self.cfg.exp_adv_min, self.cfg.exp_adv_max)
			log_pis_action = self.model.log_pi_action(zs, action, task)
			pi_loss = (( - weights * log_pis_action).mean(dim=(1, 2)) * rho).mean()
			q_loss = torch.zeros_like(pi_loss)
			prior_loss = torch.zeros_like(pi_loss)

		elif self.cfg.actor_mode=="residual":
			# Loss for TD-M(PC)^2
			action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
			std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
			eps = (pis - mu) / std
			log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
			log_pis_prior = torch.clamp(log_pis_prior, -50000, 0.0)
			self.log_pi_scale.update(log_pis_prior[0]) # Update scale

			if self.scale.value <= self.cfg.scale_threshold:
				log_pis_prior = torch.zeros_like(log_pis_prior)
			else:
				log_pis_prior = self.scale(log_pis_prior)

			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			# if il_flag is not None:
			# 	il_flag = il_flag.squeeze(-1).unsqueeze(0)
			# 	#print(f"il_flag: {il_flag.shape} {log_pis_prior.shape}")##
			# 	prior_loss = - ((log_pis_prior * il_flag).mean(dim=-1) * rho).mean() #TODO: 
			# else:			
			prior_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
			pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss
			
		elif self.cfg.actor_mode=="residual_qw":
			# Q-weight
			with torch.no_grad():
				q_act = self.model.Q(zs, action, task, return_type="avg")
				q_act = self.scale(q_act)
			adv = q_act - q_act.mean(dim=1, keepdim=True)

			weights = torch.clamp(torch.exp(adv / self.cfg.awac_lambda), self.cfg.exp_adv_min, self.cfg.exp_adv_max).squeeze(-1)
			#print(f"weights: {weights.mean()} {weights.min()} {weights.max()}")#

			action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
			std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
			eps = (pis - mu) / std
			log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
			log_pis_prior = torch.clamp(log_pis_prior, -50000, 0.0)
			self.log_pi_scale.update(log_pis_prior[0]) # Update scale

			if self.scale.value <= self.cfg.scale_threshold:
				log_pis_prior = torch.zeros_like(log_pis_prior)
			else:
				log_pis_prior = self.scale(log_pis_prior)
			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = - ((log_pis_prior * weights).mean(dim=-1) * rho).mean()
			#pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss
			pi_loss = q_loss + self.cfg.prior_coef * prior_loss

		elif self.cfg.actor_mode=="bc_sac": 
			# Vanilla BC-SAC loss for policy learning
			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = (((pis - action) ** 2).sum(dim=-1).mean(dim=1) * rho).mean()
			if self.scale.value <= self.cfg.scale_threshold:
				prior_loss = torch.zeros_like(prior_loss)
			else:
				prior_loss = prior_loss #self.scale(prior_loss)
			pi_loss = q_loss + self.cfg.prior_coef * prior_loss

		elif self.cfg.actor_mode=="bc_sac_mean": 
			# BC for the mean for policy learning
			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = (((pis - mu) ** 2).sum(dim=-1).mean(dim=1) * rho).mean()
			if self.scale.value <= self.cfg.scale_threshold:
				prior_loss = torch.zeros_like(prior_loss)
			else:
				prior_loss = prior_loss #self.scale(prior_loss)
			pi_loss = q_loss + self.cfg.prior_coef * prior_loss

		elif self.cfg.actor_mode=="bc":
			# Loss for BC-MPC baseline
			action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
			std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
			eps = (pis - mu) / std
			log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
			log_pis_prior = torch.clamp(log_pis_prior, -50000, 0.0)
			self.log_pi_scale.update(log_pis_prior[0])
			
			log_pis_prior = self.scale(log_pis_prior)
			pi_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
			prior_loss = pi_loss.detach().clone()
			q_loss = torch.zeros_like(pi_loss) # Not used

		else:
			raise NotImplementedError

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model._pi.parameters(), self.cfg.grad_clip_norm
		)
		
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item(), q_loss.item(), prior_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task, n_step=1):
		"""
		Compute the n-step TD-target.

		For n_step=1 (default), equivalent to: reward + γ * Q(next_z).
		For n_step>1: Σ_{k=0}^{n-1} γ^k * reward[t+k] + γ^n * Q(next_z[t+n]).

		Args:
				next_z (torch.Tensor): Latent states at following time steps, shape [T, B, latent_dim].
				reward (torch.Tensor): Rewards at each time step, shape [T, B, 1].
				task (torch.Tensor): Task index (only used for multi-task experiments).
				n_step (int): Number of steps for n-step TD returns.

		Returns:
				torch.Tensor: TD-targets of shape [T-n_step+1, B, 1].
		"""
		discount = (
			self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		)
		T = reward.shape[0]
		num_targets = T - n_step + 1

		# Sum discounted rewards over n steps
		targets = torch.zeros(num_targets, reward.shape[1], reward.shape[2], device=reward.device)
		for k in range(n_step):
			targets = targets + (discount ** k) * reward[k:k + num_targets]

		# Bootstrap from latent state n steps ahead
		bootstrap_z = next_z[n_step - 1:]  # [num_targets, B, latent_dim]
		pi = self.model.pi(bootstrap_z, task)[1]
		targets = targets + (discount ** n_step) * self.model.Q(
			bootstrap_z, pi, task, return_type="min", target=True
		)
		return targets
	
	@torch.no_grad()
	def _td_H_q(self, next_z, reward, task):
		"""
		Compute the TD-H-target from a sequence of rewards and the observation at the final time step
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = (
			self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		)
		traj_reward = torch.sum(reward * discount, dim=0)
		terminal_value = self.model.Q(next_z, pi, task, return_type="avg", target=False)[-1]
		return traj_reward + terminal_value


	def update(self, buffer):	
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
				buffer (common.buffer.Buffer): Replay buffer.

		Returns:
				dict: Dictionary of training statistics.
		"""
		if self.cfg.multitask and self.cfg.task in {"mt30","mt80"}:
			# offline training
			obs, action, reward, task = buffer.sample()
			mu = action.detach().clone()
			std = torch.full_like(action, self.cfg.max_std)
			info_latent = None
		else:
			# online training
			obs, action, mu, std, reward, task, info_latent = buffer.sample()

		# When use_info_latent is enabled, use stored info vectors directly as latents
		use_info_latent = self.cfg.get('use_info_latent', False) and info_latent is not None

		# Compute targets
		n_step = getattr(self.cfg, 'n_step', 1)
		with torch.no_grad():
			next_z = info_latent[1:] if use_info_latent else self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task, n_step=n_step)
			td_H_q = self._td_H_q(next_z, reward, task)
			init_z0 = info_latent[0] if use_info_latent else self.model.encode(obs[0], task)
			init_q = self.model.Q(init_z0, action[0], task, return_type="avg")
			il_flag = td_H_q > init_q

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		if use_info_latent:
			# Use ground-truth info latents directly; dynamics model is bypassed
			zs = info_latent  # (T+1, B, 4) — already on device from _to_device
			consistency_loss = 0
		else:
			z = self.model.encode(obs[0], task)
			actual_batch_size = z.shape[0]
			zs = torch.empty(
				self.cfg.train_horizon + 1,
				actual_batch_size,
				self.cfg.latent_dim,
				device=self.device,
			)
			zs[0] = z
			consistency_loss = 0
			for t in range(self.cfg.train_horizon):
				z = self.model.next(z, action[t], task)
				consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
				zs[t + 1] = z


		#####################
		if self.cfg.record_q_scale:
			# in-distribution q value scale
			with torch.no_grad():
				_qs = self.model.Q(zs[:-1], action, task, return_type="all")
				q_val_sacle = torch.cat([q.view(-1) for q in _qs]).mean().item()
		#####################

		#####################
		if self.cfg.record_pi_q_scale:
			# out-of-distribution q value scale
			with torch.no_grad():
				pi = self.model.pi(zs[:-1], task)[1]
				_qs = self.model.Q(zs[:-1], pi, task, return_type="all")
				q_val_sacle_ood = torch.cat([q.view(-1) for q in _qs]).mean().item()
		#####################

		#####################
		if self.cfg.dyna and not use_info_latent:
			# Rollout with model predictions
			_zs_sim = torch.empty(
				self.cfg.train_horizon + 1,
				self.cfg.batch_size,
				self.cfg.latent_dim,
				device=self.device,
			)
			_as_sim = torch.empty(
				self.cfg.train_horizon,
				self.cfg.batch_size,
				self.cfg.action_dim,
				device=self.device,
			)
			_rs_sim = torch.empty(
				self.cfg.train_horizon, self.cfg.batch_size, 1, device=self.device
			)
			_zs_sim[0] = z
			_td_targets_sim = 0

			with torch.no_grad():
				for t in range(self.cfg.train_horizon):
					# Sample model predi
					_a_sim = self.model.pi(_zs_sim[t], task)[1]
					_as_sim[t] = _a_sim
					if self.cfg.num_bins > 1:
						_rs_sim[t] = math.two_hot_inv(self.model.reward(_zs_sim[t], _a_sim, task), self.cfg)
					else:
						_rs_sim[t] = self.model.reward(_zs_sim[t], _a_sim, task)
					#print(torch.norm(_rs_sim, p=2))
					_zs_sim[t + 1] = self.model.next(_zs_sim[t], _a_sim, task)
					#_td_targets_sim = _td_targets_sim + _rs_sim[t] * (self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount)**t
			
				#_pi_h_sim = self.model.pi(_zs_sim[-1], task)[1]
				# _discount = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
				# _td_targets_sim = _td_targets_sim + _discount**self.cfg.horizon * self.model.Q(
				# 	_zs_sim[-1], _pi_h_sim, task, return_type="min", target=True
				# )
				_td_targets_sim = self._td_target(_zs_sim[:-1], _rs_sim, task)#
			#_q_0 = self.model.Q(_zs_sim[0], action[0], task, return_type="all")
			dyna_q_loss = 0
			# for q in range(self.cfg.num_q):
			# 	dyna_q_loss += (
			# 		math.soft_ce(_q_0[q], _td_targets_sim, self.cfg).mean()
			# 	)
			_dyna_qs = self.model.Q(_zs_sim[:-1], _as_sim, task, return_type="all")
			for t in range(self.cfg.train_horizon):
				for q in range(self.cfg.num_q):
					if self.cfg.num_bins > 1:
						dyna_q_loss += (
							math.soft_ce(_dyna_qs[q][t], _td_targets_sim[t], self.cfg).mean()
							* self.cfg.rho**t
						)
					else:
						dyna_q_loss += (
							F.mse_loss(_dyna_qs[q][t].squeeze(-1), _td_targets_sim[t])
							* self.cfg.rho**t
						)
			dyna_q_loss /= (self.cfg.train_horizon * self.cfg.num_q)
			
		# elif self.cfg.dyna:
		# 	dyna_q_loss = torch.tensor(0.0, device=self.device)
		#####################
		
		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type="all")
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		num_td_targets = td_targets.shape[0]  # T - n_step + 1
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.train_horizon):
			if self.cfg.num_bins > 1:
				reward_loss += (
					math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
					* self.cfg.rho**t
				)
			else:
				reward_loss += (
					F.mse_loss(reward_preds[t].squeeze(-1), reward[t])
					* self.cfg.rho**t
				)
		for t in range(num_td_targets):
			if self.cfg.num_bins > 1:
				for q in range(self.cfg.num_q):
					value_loss += (
						math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
						* self.cfg.rho**t
					)
			else:
				for q in range(self.cfg.num_q):
					value_loss += (
						F.mse_loss(qs[q][t].squeeze(-1), td_targets[t])
						* self.cfg.rho**t
					)

		consistency_loss *= 1 / self.cfg.train_horizon
		reward_loss *= 1 / self.cfg.train_horizon
		value_loss *= 1 / (num_td_targets * self.cfg.num_q)

		if self.cfg.dyna:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss
				+ self.cfg.reward_coef * reward_loss
				+ self.cfg.value_coef * value_loss
				+ self.cfg.dyna_coef * dyna_q_loss
			)
		else:
			total_loss = (
				self.cfg.consistency_coef * consistency_loss
				+ self.cfg.reward_coef * reward_loss
				+ self.cfg.value_coef * value_loss
			)
			
		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm
		)
		self.optim.step()

		# Update policy
		pi_loss, pi_loss_q, pi_loss_prior  = self.update_pi(_zs.detach(), action.detach(), mu.detach(), std.detach(), task, il_flag)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		def _to_float(x):
			return float(x.mean().item()) if hasattr(x, 'mean') else float(x)
		return {
			"consistency_loss": _to_float(consistency_loss),
			"reward_loss": _to_float(reward_loss),
			"value_loss": _to_float(value_loss),
			"pi_loss": pi_loss,
			"pi_loss_q": pi_loss_q,
			"pi_loss_prior": pi_loss_prior,
			"total_loss": _to_float(total_loss),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
			"log_pi_scale": float(self.log_pi_scale.value),	
			"dyna_q_loss": dyna_q_loss.item() if self.cfg.dyna else 0,	
			"q_val_scale": q_val_sacle if self.cfg.record_q_scale else 0,
			"q_val_scale_ood": q_val_sacle_ood if self.cfg.record_pi_q_scale else 0,
		}
