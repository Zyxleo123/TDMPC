from time import time
import os

import math
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from tdmpc2.trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._tds = []
        # For VecDriveEnvWrapper, use the embedded single eval_env for evaluation;
        # otherwise fall back to self.env (single env, already correct).
        self.eval_env = getattr(self.env, 'eval_env', self.env)

    def load_agent(self, log_dir):
        self.agent.load(log_dir)
        # Restore step count from checkpoint filename (e.g. "300000.pt")
        stem = os.path.splitext(os.path.basename(log_dir))[0]
        if stem.isdigit():
            self._step = int(stem)
        print(f"Loaded agent from {log_dir} (step={self._step})")
        # Load buffer if a matching .buffer file exists
        buffer_path = os.path.splitext(log_dir)[0] + '.buffer'
        if os.path.exists(buffer_path):
            self.buffer.load(buffer_path)
            print(f"Loaded buffer from {buffer_path} ({len(self.buffer)} episodes)")

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )
    
    def eval_drive(self):
        """Evaluate a TD-MPC2 agent."""
        # TODO: fix this for rgb input
        ep_rewards, ep_successes = [], []
        psi_smoothness = [] # TODO: add this to the logger

        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.eval_env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.eval_env, enabled=(i == 0))
            while not done:
                action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.eval_env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')

        if self.cfg.eval_pi:
            # Evaluate nominal policy pi
            ep_rewards_pi, ep_successes_pi = [], []
            for i in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.eval_env.reset()[0], False, 0, 0
                while not done:
                    action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards_pi.append(ep_reward)
                ep_successes_pi.append(info["success"])
            
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_reward_pi=np.nanmean(ep_rewards_pi) if self.cfg.eval_pi else np.nan,
            episode_success_pi=np.nanmean(ep_successes_pi) if self.cfg.eval_pi else np.nan,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        ep_rewards_pred = []
        ep_values_pred, ep_values_real = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.eval_env.reset()[0], False, 0, 0
            ep_pred_reward = 0
            ep_vals, ep_real_rewards = [], []

            save_traj = self.cfg.save_trajectories
            traj_plans = []

            if self.cfg.save_video:
                self.logger.video.init(self.eval_env, enabled=True)
            while not done:
                if save_traj:
                    action, _, _, plan_info = self.agent.act(obs, t0=t == 0, eval_mode=True, return_plan=True)
                else:
                    action = self.agent.act(obs, t0=t == 0, eval_mode=True) # TODO: set use pi=False

                pred_reward = self.agent.predict_reward(obs, action)
                pred_value = self.agent.predict_value(obs, action)
                ep_pred_reward += pred_reward
                ep_vals.append(pred_value)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                ep_real_rewards.append(reward)

                if save_traj:
                    traj_plans.append({"plan": plan_info, "reward": reward, "pred_reward": pred_reward, "pred_value": pred_value, "obs": obs})

                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.eval_env)
            
            # Calculate discounted returns
            discount = self.agent._get_discount(self.cfg.episode_length)
            G = 0
            returns = []
            for r in reversed(ep_real_rewards):
                G = r + discount * G
                returns.insert(0, G)
            ep_values_pred.append(np.nanmean(ep_vals))
            # Just incase there are no returns
            if len(returns) > 0:
                ep_values_real.append(np.nanmean(returns))
            else:
                ep_values_real.append(np.nan)

            if save_traj:
                for t_step, return_val in enumerate(returns):
                    if t_step < len(traj_plans):
                         traj_plans[t_step]["value_real"] = return_val

                save_dir = os.path.join(self.cfg.work_dir, "eval_trajectories")
                os.makedirs(save_dir, exist_ok=True)
                
                save_data = {"traj_plans": traj_plans}
                if self.cfg.save_video:
                    save_data["frames"] = np.stack(self.logger.video.frames)
                
                torch.save(save_data, os.path.join(save_dir, f'plans_step_{self._step}_ep_{i}.pt'))

            ep_rewards.append(ep_reward)
            ep_rewards_pred.append(ep_pred_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        
        if self.cfg.eval_pi:
            # Evaluate nominal policy pi
            ep_rewards_pi, ep_successes_pi = [], []
            for i in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.eval_env.reset()[0], False, 0, 0
                while not done:
                    action = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)  #TODO: currently testing arc use_pi=True
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards_pi.append(ep_reward)
                ep_successes_pi.append(info["success"])
            
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_reward_pred=np.nanmean(ep_rewards_pred),
            episode_value_pred=np.nanmean(ep_values_pred),
            episode_value_real=np.nanmean(ep_values_real),
            episode_success=np.nanmean(ep_successes),
            episode_reward_pi=np.nanmean(ep_rewards_pi) if self.cfg.eval_pi else np.nan,
            episode_success_pi=np.nanmean(ep_successes_pi) if self.cfg.eval_pi else np.nan,
        )

    def eval_value(self, n_samples=1):
        """evaluate value approximation."""
        # MC value estimation
        mc_ep_rewards = []
        for i in range(n_samples):
            obs, done, ep_reward, t = self.eval_env.reset()[0], False, 0, 0
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
                ep_reward += reward * self.agent.discount ** t
                t += 1
            mc_ep_rewards.append(ep_reward)

        # Value function approximation
        q_values = []
        for i in range(n_samples):
            obs, done, ep_reward, t = self.eval_env.reset()[0], False, 0, 0
            
            action = self.agent.act(obs, t0=t == 0, eval_mode=True)#, use_pi=True)
            task = None
            #print("action: ", action.shape, ", obs: ", obs.shape)
            # TODO: fix this for rgb input
            # if self.cfg.obs == "state":
                # q_value = self.agent.model.Q(self.agent.model.encode(obs.to(self.agent.device), task), 
                #                             action.to(self.agent.device), 
                #                             task, return_type="avg")
            if self.cfg.obs == "rgb":
                q_value = self.agent.model.Q(self.agent.model.encode(obs.unsqueeze(0).to(self.agent.device), task).squeeze(0), 
                                            action.to(self.agent.device), 
                                            task, return_type="avg")
            else:
                q_value = self.agent.model.Q(self.agent.model.encode(obs.to(self.agent.device), task), 
                                            action.to(self.agent.device), 
                                            task, return_type="avg")
                #raise ValueError("Unknown observation type:", self.cfg.obs)
            
            q_values.append(q_value.detach().cpu().numpy())
        
        return dict(
            mc_value= np.nanmean(mc_ep_rewards),
            q_value= np.nanmean(q_values),
        )

    def to_td(self, obs, action=None, mu=None, std=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            # For VecDriveEnvWrapper rand_act() returns (n_envs, action_dim); use the
            # single eval_env instead so the placeholder has the correct (action_dim,) shape.
            rand_env = getattr(self.env, 'eval_env', self.env)
            action = torch.full_like(rand_env.rand_act(), float("nan"))
        if mu is None:
            mu = torch.full_like(action, float("nan"))
        if std is None:
            std = torch.full_like(action, float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                mu=mu.unsqueeze(0),
                std=std.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent (dispatches to single-env or vec-env variant)."""
        if hasattr(self.env, 'n_envs') and self.env.n_envs > 1:
            self._train_vec()
        else:
            self._train_single()

    def _train_single(self):
        """Train loop for a single (non-vectorized) environment."""
        train_metrics, done, eval_next = {}, True, True
        train_log_return = []
        train_log_success = []  
        train_log_ep_len = [] 

        use_pi_flag = False   

        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                # NOTE: added to sample from the nominal policy
                use_pi_flag = np.random.rand() < self.cfg.use_pi_prob
                
                if eval_next:
                    eval_metrics = self.eval()

                    if self.cfg.eval_value:
                        eval_metrics.update(self.eval_value())
                        
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0 and self._tds:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"], # for TorchDriveEnv
                    )
                    train_metrics.update(self.common_metrics())

                    train_log_return.append(train_metrics['episode_reward'])
                    train_log_success.append(train_metrics['episode_success'])
                    train_log_ep_len.append(len(self._tds[1:]))

                    self.logger.log(train_metrics, "train")

                    if self._ep_idx % self.cfg.log_freq == 0:
                        results_metrics = {'return': np.mean(train_log_return),
                                        'episode_length': np.mean(train_log_ep_len),
                                        'success': np.mean(train_log_success),
                                        'success_subtasks': info['success_subtasks'],
                                        'step': self._step,}
                        #print("result return:", results_metrics["return"])#
                        self.logger.log(results_metrics, "results")

                        train_log_return = []
                        train_log_success = []

                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                t0 = len(self._tds) == 1
                action, mu, std = self.agent.act(obs, t0=t0, use_pi = use_pi_flag) # TODO: don't use pi here  
            else:
                action = self.env.rand_act()
                mu, std = action.detach().clone(), torch.full_like(action, math.exp(self.cfg.log_std_max)) # torch.full_like(action, float('nan')), torch.full_like(action, float('nan')) #  # noqa
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, mu, std, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps and len(self.buffer) > 0:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = self.cfg.n_updates
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            if self._step % self.cfg.save_freq == 0:
                self.logger.save_agent(self.agent, identifier=f"{self._step}")
                self.buffer.save(self.logger.model_dir / f"{self._step}.buffer")
                print(f"Saved agent and buffer at step {self._step}")

            self._step += 1

        self.logger.finish(self.agent)

    def _train_vec(self):
        """Train loop for a vectorized (n_envs > 1) torchdriveenv."""
        n_envs = self.env.n_envs
        train_metrics = {}
        eval_next = True
        _pretrained = False

        train_log_return = []
        train_log_success = []
        train_log_ep_len = []
        _pretrained = self._step >= self.cfg.seed_steps

        # Initialize: reset all envs, build per-env episode buffers.
        obs, _ = self.env.reset()  # (n_envs, *obs_shape)
        _tds_all = [[self.to_td(obs[i])] for i in range(n_envs)]
        use_pi_flags = [False] * n_envs

        while self._step <= self.cfg.steps:
            # Periodic eval trigger.
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            if eval_next:
                eval_metrics = self.eval()
                if self.cfg.eval_value:
                    eval_metrics.update(self.eval_value())
                eval_metrics.update(self.common_metrics())
                self.logger.log(eval_metrics, "eval")
                eval_next = False

            # Collect actions for all envs.
            if self._step > self.cfg.seed_steps:
                actions, mus, stds = [], [], []
                for i in range(n_envs):
                    t0 = len(_tds_all[i]) == 1
                    a, mu, std = self.agent.act(obs[i], t0=t0, use_pi=use_pi_flags[i])
                    actions.append(a)
                    mus.append(mu)
                    stds.append(std)
                actions = torch.stack(actions)   # (n_envs, action_dim)
                mus = torch.stack(mus)
                stds = torch.stack(stds)
            else:
                actions = self.env.rand_act()    # (n_envs, action_dim)
                mus = actions.detach().clone()
                stds = torch.full_like(actions, math.exp(self.cfg.log_std_max))

            # Step all envs simultaneously.
            next_obs, rewards, dones, truncateds, infos = self.env.step(actions)
            # next_obs[i] = terminal obs for done[i], else step obs.
            # self.env._current_obs[i] = auto-reset obs for next step.

            # Process each env's transition.
            for i in range(n_envs):
                _tds_all[i].append(
                    self.to_td(next_obs[i], actions[i], mus[i], stds[i], rewards[i])
                )
                ep_done = bool(dones[i]) or bool(truncateds[i])
                if ep_done:
                    ep_success = self.env.get_success(infos, i, done=True)
                    ep_reward = torch.tensor(
                        [td["reward"] for td in _tds_all[i][1:]]
                    ).sum()

                    train_log_return.append(ep_reward)
                    train_log_success.append(ep_success)
                    train_log_ep_len.append(len(_tds_all[i][1:]))

                    train_metrics.update(
                        episode_reward=ep_reward,
                        episode_success=ep_success,
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")

                    if self._ep_idx % self.cfg.log_freq == 0:
                        results_metrics = {
                            'return': np.mean([r.item() if hasattr(r, 'item') else r for r in train_log_return]),
                            'episode_length': np.mean(train_log_ep_len),
                            'success': np.mean(train_log_success),
                            'step': self._step,
                        }
                        self.logger.log(results_metrics, "results")
                        train_log_return = []
                        train_log_success = []
                        train_log_ep_len = []

                    self._ep_idx = self.buffer.add(torch.cat(_tds_all[i]))

                    # Start fresh episode buffer for this env using the auto-reset obs.
                    reset_obs_i = self.env._current_obs[i]
                    _tds_all[i] = [self.to_td(reset_obs_i)]
                    use_pi_flags[i] = np.random.rand() < self.cfg.use_pi_prob

            # Update obs to the post-step / post-reset observations.
            obs = self.env._current_obs

            # Gradient updates: maintain same update-to-data ratio as single-env.
            if self._step >= self.cfg.seed_steps and len(self.buffer) > 0:
                if not _pretrained:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                    _pretrained = True
                else:
                    num_updates = n_envs * self.cfg.n_updates
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            if self._step % self.cfg.save_freq == 0:
                self.logger.save_agent(self.agent, identifier=f"{self._step}")
                self.buffer.save(self.logger.model_dir / f"{self._step}.buffer")
                print(f"Saved agent and buffer at step {self._step}")

            # Each physical step covers n_envs environment interactions.
            self._step += n_envs

        self.logger.finish(self.agent)
