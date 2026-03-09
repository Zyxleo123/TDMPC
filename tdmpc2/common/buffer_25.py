import sys

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    """
    Replay buffer for TD-MPC2 training. Based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.

    Optionally maintains a priority buffer that oversamples experience
    near high-reward "peaks", detected via a running Z-score threshold.
    Enable with priority_buffer_ratio > 0 in config.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if sys.platform == "darwin":
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._num_eps = 0

        # Priority buffer setup
        self._use_priority = getattr(cfg, "priority_buffer_ratio", 0.0) > 0
        if self._use_priority:
            self._priority_ratio = cfg.priority_buffer_ratio
            self._priority_k = cfg.priority_buffer_k
            self._priority_window = getattr(cfg, "priority_buffer_window", cfg.train_horizon)
            self._priority_capacity = getattr(cfg, "priority_buffer_size", 10_000)
            self._ema_alpha = cfg.reward_ema_alpha
            # Initialize EMA stats: mean=0, var=1 (gives a reasonable std floor early on)
            self._ema_mean = 0.0
            self._ema_var = 1.0
            self._priority_num_eps = 0
            p_num_slices = max(1, round(cfg.batch_size * self._priority_ratio))
            m_num_slices = cfg.batch_size - p_num_slices
            self._p_batch_size = p_num_slices * (cfg.train_horizon + 1)
            self._m_batch_size = m_num_slices * (cfg.train_horizon + 1)
            self._p_sampler = SliceSampler(
                num_slices=p_num_slices,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
            )
        else:
            m_num_slices = cfg.batch_size

        self._sampler = SliceSampler(
            num_slices=m_num_slices,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
        )
        self._batch_size = m_num_slices * (cfg.train_horizon + 1)

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage, sampler, batch_size):
        """Reserve a ReplayBuffer with the given storage, sampler, and batch size."""
        return ReplayBuffer(
            storage=storage,
            sampler=sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=batch_size,
        )

    def _storage_device(self, tds, capacity, label="Buffer"):
        """Choose storage device and print diagnostics."""
        print(f"{label} capacity: {capacity:,}")
        if sys.platform == "darwin":
            return "cpu"
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * capacity
        print(f"{label} storage required: {total_bytes/1e9:.2f} GB")
        device = "cuda" if 2.5 * total_bytes < mem_free else "cpu"
        print(f"Using {device.upper()} memory for {label.lower()} storage.")
        return device

    def _init(self, tds):
        """Initialize the main replay buffer."""
        device = "cpu"
        storage = LazyTensorStorage(self._capacity, device=torch.device(device))
        return self._reserve_buffer(storage, self._sampler, self._batch_size)

    def _init_priority_buffer(self, tds):
        """Initialize the priority replay buffer."""
        device = "cpu"
        storage = LazyTensorStorage(self._priority_capacity, device=torch.device(device))
        return self._reserve_buffer(storage, self._p_sampler, self._p_batch_size)

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs = td["obs"]
        action = td["action"][1:]
        mu = td["mu"][1:]
        std = td["std"][1:]
        reward = td["reward"][1:].unsqueeze(-1)
        task = td["task"][0] if "task" in td.keys() else None
        return self._to_device(obs, action, mu, std, reward, task)

    def _apply_reward_decay(self, td):
        """
        Apply backward exponential decay to propagate future rewards to earlier steps.
        For a reward sequence (0, 0, 100) with decay=0.9, this produces (81, 90, 100).
        Formula: reward[i] = reward[i] + decay * reward[i+1], applied from end to start.
        td["reward"][0] is NaN (placeholder for the initial obs), so we start from index 1.
        """
        decay = self.cfg.reward_decay
        rewards = td["reward"]  # shape (T,), rewards[0] is NaN
        for i in range(len(rewards) - 2, 0, -1):
            rewards[i] = rewards[i] + decay * rewards[i + 1]
        td["reward"] = rewards

    def _update_ema(self, rewards):
        """Update EMA mean and variance with non-NaN rewards from an episode."""
        alpha = self._ema_alpha
        for r in rewards:
            if torch.isnan(r):
                continue
            r = r.item()
            old_mean = self._ema_mean
            self._ema_mean = alpha * self._ema_mean + (1 - alpha) * r
            self._ema_var = alpha * self._ema_var + (1 - alpha) * (r - old_mean) ** 2

    def _detect_peaks(self, rewards):
        """
        Return indices of timesteps whose reward exceeds mean + k * std.
        Uses current EMA stats. NaN entries (episode start placeholder) are skipped.
        A minimum std of 1e-3 avoids instability when all rewards are near-identical.
        """
        ema_std = max(self._ema_var ** 0.5, 1e-3)
        threshold = self._ema_mean + self._priority_k * ema_std
        peaks = []
        for i, r in enumerate(rewards):
            if not torch.isnan(r) and r.item() > threshold:
                peaks.append(i)
                print(f"[PriorityBuffer] Peak detected: reward={r.item():.4f}, mean={self._ema_mean:.4f}, std={ema_std:.4f}, threshold={threshold:.4f}")
        return peaks

    def _add_peaks_to_priority_buffer(self, td, peak_indices):
        """
        For each detected peak index, extract a window of W steps ending at the peak
        and add it to the priority buffer as a short pseudo-episode.
        The window spans [peak - W, peak] (inclusive), giving W+1 timesteps.
        Requires at least train_horizon+1 steps to be usable by the slice sampler.
        """
        W = self._priority_window
        min_len = self.cfg.train_horizon + 1
        for peak_idx in peak_indices:
            start = max(0, peak_idx - W)
            end = peak_idx + 1  # inclusive of the peak timestep
            if end - start < min_len:
                continue
            window_td = td[start:end].clone()
            window_td["episode"] = (
                torch.ones_like(window_td["reward"], dtype=torch.int64)
                * self._priority_num_eps
            )
            if self._priority_num_eps == 0:
                self._priority_buffer = self._init_priority_buffer(window_td)
            self._priority_buffer.extend(window_td)
            self._priority_num_eps += 1

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = torch.ones_like(td["reward"], dtype=torch.int64) * self._num_eps

        # FIX for HumanoidBench #
        if len(td["episode"]) <= self.cfg.train_horizon + 1:
            return self._num_eps
        ################################

        # Detect peaks on original rewards (before decay alters the signal)
        if self._use_priority:
            peak_indices = self._detect_peaks(td["reward"])
            self._update_ema(td["reward"])

        if self.cfg.reward_decay > 0.0:
            self._apply_reward_decay(td)

        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1

        # Add priority windows (using decayed rewards, matching what training sees)
        if self._use_priority and peak_indices:
            self._add_peaks_to_priority_buffer(td, peak_indices)

        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        if self._use_priority and self._priority_num_eps > 0:
            # Split batch: m slices from main, p slices from priority
            td_main = (
                self._buffer.sample(batch_size=self._m_batch_size)
                .view(-1, self.cfg.train_horizon + 1)
                .permute(1, 0)
            )
            td_prio = (
                self._priority_buffer.sample()
                .view(-1, self.cfg.train_horizon + 1)
                .permute(1, 0)
            )
            td = torch.cat([td_main, td_prio], dim=1)
        else:
            # Priority buffer disabled or not yet populated: use full main batch
            td = self._buffer.sample().view(-1, self.cfg.train_horizon + 1).permute(1, 0)
        return self._prepare_batch(td)
