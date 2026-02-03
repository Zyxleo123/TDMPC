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

    def load_agent(self, log_dir):
        self.agent.load(log_dir)
        print(f"Loaded agent from {log_dir}")

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
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        
        if self.cfg.eval_pi:
            # Evaluate nominal policy pi
            ep_rewards_pi, ep_successes_pi = [], []
            for i in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
                while not done:
                    action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
                    obs, reward, done, truncated, info = self.env.step(action)
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
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            ep_pred_reward = 0
            
            save_traj = self.cfg.save_trajectories
            traj_plans = []

            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=True)
            while not done:
                if save_traj:
                    action, _, _, plan_info = self.agent.act(obs, t0=t == 0, eval_mode=True, return_plan=True)
                else:
                    action = self.agent.act(obs, t0=t == 0, eval_mode=True) # TODO: set use pi=False
                
                pred_reward = self.agent.predict_reward(obs, action)
                ep_pred_reward += pred_reward
                obs, reward, done, truncated, info = self.env.step(action)
                
                if save_traj:
                    traj_plans.append({"plan": plan_info, "reward": reward, "pred_reward": pred_reward})

                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            
            if save_traj:
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
                obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
                while not done:
                    action = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)  #TODO: currently testing arc use_pi=True
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards_pi.append(ep_reward)
                ep_successes_pi.append(info["success"])
            
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_reward_pred=np.nanmean(ep_rewards_pred),
            episode_success=np.nanmean(ep_successes),
            episode_reward_pi=np.nanmean(ep_rewards_pi) if self.cfg.eval_pi else np.nan,
            episode_success_pi=np.nanmean(ep_successes_pi) if self.cfg.eval_pi else np.nan,
        )

    def eval_value(self, n_samples=1):
        """evaluate value approximation."""
        # MC value estimation
        mc_ep_rewards = []
        for i in range(n_samples):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True, use_pi=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward * self.agent.discount ** t
                t += 1
            mc_ep_rewards.append(ep_reward)

        # Value function approximation
        q_values = []
        for i in range(n_samples):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
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
            action = torch.full_like(self.env.rand_act(), float("nan"))
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
        """Train a TD-MPC2 agent."""
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

                if self._step > 0:
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
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            if self._step % self.cfg.save_freq == 0:
                self.logger.save_agent(self.agent, identifier=f"{self._step}")
                print(f"Saved agent at step {self._step}")

            self._step += 1

        self.logger.finish(self.agent)
