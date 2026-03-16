"""
Latent collapse diagnostic for TD-MPC2.

Loads a trained TDMPC2 checkpoint, encodes observations from episode files,
and measures whether the latent representations have collapsed to a low-dimensional
subspace.

Effective rank (Roy & Vetterli, 2007):
    srank(Z) = exp(H),  H = -sum(p_i * log(p_i)),  p_i = sigma_i / sum(sigma)

Usage (Hydra-style, like train.py / evaluate.py):

    # From eval_trajectories .pt files (auto-detected by 'traj_plans' key):
    python compute_effective_rank.py \\
        task=dog-run  model_size=5 \\
        checkpoint=/path/to/checkpoint.pt \\
        "data='/path/to/eval_trajectories/*.pt'"

    # From plain observation .npz files:
    python compute_effective_rank.py \\
        task=dog-run  model_size=5 \\
        checkpoint=/path/to/checkpoint.pt \\
        data=/path/to/states.npz

    # Save diagnostic plots:
    python compute_effective_rank.py \\
        task=dog-run  model_size=5 \\
        checkpoint=/path/to/checkpoint.pt \\
        "data='/path/to/eval_trajectories/*.pt'" \\
        plot=true  out_dir=/path/to/plots

See config.yaml for all available config overrides.
"""

import os
import sys
from glob import glob
from pathlib import Path

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.tdmpc25 import TDMPC2


# ─── Collapse metrics ─────────────────────────────────────────────────────────

def effective_rank(Z: torch.Tensor) -> float:
    """
    Roy & Vetterli (2007) effective rank.

    srank(Z) = exp(H),  H = -sum(p_i * log(p_i)),  p_i = sigma_i / sum(sigma).
    Returns a value in [1, D]:
      ~1  → complete collapse
      ~D  → fully expressive
    """
    Z = Z.float()
    Z = Z - Z.mean(dim=0, keepdim=True)
    sigma = torch.linalg.svdvals(Z)
    sigma = sigma[sigma > 1e-10]
    p = sigma / sigma.sum()
    entropy = -(p * p.log()).sum().item()
    return float(np.exp(entropy))


def explained_variance_curve(Z: torch.Tensor) -> np.ndarray:
    """Cumulative fraction of variance explained by top-k PCA components."""
    Z = Z.float()
    Z = Z - Z.mean(dim=0, keepdim=True)
    sigma = torch.linalg.svdvals(Z).cpu().numpy()
    variance = sigma ** 2
    return np.cumsum(variance) / variance.sum()


def per_dimension_stats(Z: torch.Tensor) -> dict:
    """Per-dimension mean, std, and fraction of dead dimensions (std < 0.01)."""
    Z = Z.float().cpu().numpy()
    stds  = Z.std(axis=0)
    means = Z.mean(axis=0)
    return {
        "std_mean":       float(stds.mean()),
        "std_min":        float(stds.min()),
        "std_max":        float(stds.max()),
        "dead_fraction":  float((stds < 0.01).mean()),
        "mean_abs_mean":  float(np.abs(means).mean()),
    }


def pairwise_cosine_stats(Z: torch.Tensor, subsample: int = 500) -> dict:
    """Mean/std of pairwise cosine similarities on a random subsample."""
    Z_np = Z.float().cpu().numpy()
    if len(Z_np) > subsample:
        idx  = np.random.choice(len(Z_np), subsample, replace=False)
        Z_np = Z_np[idx]
    Z_norm  = Z_np / (np.linalg.norm(Z_np, axis=1, keepdims=True) + 1e-10)
    cos_sim = Z_norm @ Z_norm.T
    upper   = cos_sim[np.triu_indices(len(Z_np), k=1)]
    return {
        "cosine_mean": float(upper.mean()),
        "cosine_std":  float(upper.std()),
        "cosine_min":  float(upper.min()),
        "cosine_max":  float(upper.max()),
    }


def interpret(d: int, erank: float, ev_curve: np.ndarray, dim_stats: dict) -> str:
    """Human-readable verdict."""
    lines = []

    erank_ratio = erank / d
    if erank_ratio < 0.05:
        verdict = "SEVERE COLLAPSE"
    elif erank_ratio < 0.15:
        verdict = "MODERATE COLLAPSE"
    elif erank_ratio < 0.35:
        verdict = "MILD COLLAPSE / low utilization"
    else:
        verdict = "HEALTHY (no collapse detected)"

    lines.append(f"Effective rank : {erank:.1f} / {d}  ({100*erank_ratio:.1f}%)  →  {verdict}")

    k1  = float(ev_curve[0])
    k5  = float(ev_curve[4])  if len(ev_curve) > 4  else float(ev_curve[-1])
    k20 = float(ev_curve[19]) if len(ev_curve) > 19 else float(ev_curve[-1])
    lines.append(
        f"Variance explained: top-1={100*k1:.1f}%  top-5={100*k5:.1f}%  top-20={100*k20:.1f}%"
    )
    if k1 > 0.90:
        lines.append("  ↳ >90% variance in a SINGLE dimension — near-complete collapse")
    elif k5 > 0.95:
        lines.append("  ↳ >95% variance in first 5 dimensions — strong dimensional collapse")

    lines.append(
        f"Dead dimensions (std<0.01): {100*dim_stats['dead_fraction']:.1f}%  "
        f"|  overall dim std: mean={dim_stats['std_mean']:.3f}  "
        f"min={dim_stats['std_min']:.4f}  max={dim_stats['std_max']:.3f}"
    )
    return "\n".join(lines)


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_obs_from_traj_pt(path: str) -> torch.Tensor:
    """
    Load observations from an eval_trajectories .pt file produced by online_trainer.py.

    Format: {"traj_plans": [{"obs": ..., "plan": ..., "reward": ..., ...}, ...], ...}
    Each "obs" entry may be a numpy array or a torch Tensor.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    assert "traj_plans" in obj, \
        f"Expected 'traj_plans' key in {path}. Keys found: {list(obj.keys())}"
    steps = obj["traj_plans"]
    assert len(steps) > 0 and "obs" in steps[0], (
        f"'obs' not found in traj_plans entries of {path}. "
        "Re-run evaluation with save_trajectories=true after updating online_trainer.py."
    )
    obs_list = []
    for step in steps:
        o = step["obs"]
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(np.array(o), dtype=torch.float32)
        obs_list.append(o.float())
    # Stack along dim 0; preserve original shape (e.g. [C, H, W] for RGB obs)
    obs = torch.stack(obs_list, dim=0)
    print(f"  Loaded {obs.shape[0]} steps from {path} (traj format)")
    return obs


def load_obs_from_npz(path: str) -> torch.Tensor:
    """Load observations from a .npz file (key: 'states' or first key)."""
    data = np.load(path)
    key  = "states" if "states" in data else list(data.keys())[0]
    obs  = torch.from_numpy(data[key]).float()
    print(f"  Loaded {obs.shape[0]} steps from {path} (key='{key}')")
    return obs


def load_obs_from_plain_pt(path: str, data_key: str = "obs") -> torch.Tensor:
    """Load observations from a plain .pt file (TensorDict or dict with obs key)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    assert hasattr(obj, "keys"), f"Unsupported .pt file format: {type(obj)}"
    assert data_key in obj.keys(), \
        f"Key '{data_key}' not found. Available keys: {list(obj.keys())}"
    obs = obj[data_key]
    if obs.ndim > 2:
        obs = obs.reshape(-1, obs.shape[-1])
    obs = obs.float()
    print(f"  Loaded {obs.shape[0]} steps from {path} (key='{data_key}')")
    return obs


def load_episodes(paths: list, data_key: str = "obs") -> torch.Tensor:
    """Load and concatenate observations from a list of episode files."""
    all_obs = []
    for p in paths:
        if p.endswith(".npz"):
            all_obs.append(load_obs_from_npz(p))
        elif p.endswith(".pt"):
            # Peek at the file to decide which loader to use
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and "traj_plans" in obj:
                all_obs.append(load_obs_from_traj_pt(p))
            else:
                all_obs.append(load_obs_from_plain_pt(p, data_key=data_key))
        else:
            raise ValueError(f"Unsupported file format: {p}. Expected .npz or .pt")
    return torch.cat(all_obs, dim=0)


# ─── Optional plots ────────────────────────────────────────────────────────────

def save_plots(Z: torch.Tensor, erank: float, ev_curve: np.ndarray,
               dim_stats: dict, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    Z_np = Z.float().cpu().numpy()
    d    = Z_np.shape[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Explained variance curve
    ax = axes[0]
    n_components = min(d, 64)
    ax.plot(range(1, n_components + 1), ev_curve[:n_components] * 100)
    ax.axhline(95, color="red",    linestyle="--", linewidth=0.8, label="95%")
    ax.axhline(99, color="orange", linestyle="--", linewidth=0.8, label="99%")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("Explained Variance Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Per-dimension std histogram
    ax = axes[1]
    stds = Z_np.std(axis=0)
    ax.hist(stds, bins=50, edgecolor="black", linewidth=0.5)
    ax.axvline(0.01, color="red", linestyle="--", label="dead threshold (0.01)")
    ax.set_xlabel("Std per latent dimension")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Dimension Std\n"
                 f"Dead: {100*dim_stats['dead_fraction']:.1f}%  "
                 f"Mean: {dim_stats['std_mean']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Latent trajectories (first 8 dims over time)
    ax = axes[2]
    n_show = min(8, d)
    for i in range(n_show):
        ax.plot(Z_np[:, i], alpha=0.7, linewidth=0.8, label=f"dim {i}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Latent value")
    ax.set_title(f"Latent Trajectories (first {n_show} dims)\n"
                 f"Effective rank: {erank:.1f}/{d}")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(out_dir) / "latent_collapse_diagnostic.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_name="config", config_path=".")
def main(cfg):
    """
    Compute effective rank and collapse diagnostics for a TDMPC2 encoder.

    Required overrides:
        checkpoint=<path>   Path to .pt model checkpoint.
        data=<path>         Path to episode file(s). Supports glob patterns.
                            Accepts eval_trajectories .pt files, plain obs .pt,
                            or .npz files.

    Optional overrides:
        data_key=obs        Key for plain .pt files (default: obs).
        batch_size=1024     Batch size for encoding (default: 1024).
        task_id=0           Task index for multi-task models (default: 0).
        plot=true           Save diagnostic plots (default: false).
        out_dir=.           Directory for plots (default: current dir).
    """
    assert cfg.get("checkpoint", None) not in (None, "???"), \
        "Must specify checkpoint=<path>"
    assert cfg.get("data", None) not in (None, "???"), \
        "Must specify data=<path or glob>"

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(colored(f"Device: {device}", "blue"))

    # ── Infer obs_shape / action_dim / episode_length from trajectory file ─────
    # Peek at the first data file to extract these without needing the environment.
    peek_path = cfg.data if "*" not in cfg.data and "?" not in cfg.data \
                else sorted(glob(cfg.data))[0]
    peek = torch.load(peek_path, map_location="cpu", weights_only=False)
    if isinstance(peek, dict) and "traj_plans" in peek:
        steps = peek["traj_plans"]
        # obs shape
        sample_obs = steps[0]["obs"]
        if not isinstance(sample_obs, torch.Tensor):
            sample_obs = torch.tensor(np.array(sample_obs))
        obs_shape = tuple(sample_obs.shape)
        cfg.obs_shape = {cfg.obs: obs_shape}
        # action_dim from plan mean (shape: [horizon, action_dim])
        sample_mean = steps[0]["plan"][0]["mean"]
        cfg.action_dim = int(sample_mean.shape[-1])
        # episode_length from trajectory length
        cfg.episode_length = len(steps)
        print(f"Inferred from data: obs_shape={cfg.obs_shape}, "
              f"action_dim={cfg.action_dim}, episode_length={cfg.episode_length}")
    else:
        raise RuntimeError(
            "Could not infer model config from data file. "
            "Only eval_trajectories .pt files (with 'traj_plans' key) are supported "
            "for auto-inference. Pass a trajectory file or set obs_shape/action_dim manually."
        )

    # ── Load agent ─────────────────────────────────────────────────────────────
    print(colored(f"Loading checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    assert os.path.exists(cfg.checkpoint), f"Checkpoint not found: {cfg.checkpoint}"

    # Infer model architecture from checkpoint state_dict to avoid size mismatches
    ckpt_state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    ckpt_model = ckpt_state["model"]
    # latent_dim: output of dynamics MLP final layer
    cfg.latent_dim = int(ckpt_model["_dynamics.2.bias"].shape[0])
    # mlp_dim: hidden layer width of dynamics MLP
    cfg.mlp_dim = int(ckpt_model["_dynamics.0.bias"].shape[0])
    # num_q: ensemble size
    cfg.num_q = int(ckpt_model["_Qs.params.0"].shape[0])
    # obs type: rgb if CNN encoder keys present, else state
    if any(k.startswith("_encoder.rgb") for k in ckpt_model):
        ckpt_obs = "rgb"
    else:
        ckpt_obs = "state"
    if cfg.obs != ckpt_obs:
        # Re-key obs_shape so the correct encoder type is built
        old_shape = list(cfg.obs_shape.get(cfg.obs, []))
        OmegaConf.update(cfg, "obs_shape", {ckpt_obs: old_shape}, merge=False)
        cfg.obs = ckpt_obs
    print(colored(
        f"Inferred from checkpoint: latent_dim={cfg.latent_dim}, "
        f"mlp_dim={cfg.mlp_dim}, num_q={cfg.num_q}, obs={cfg.obs}",
        "blue",
    ))

    agent = TDMPC2(cfg)
    agent.load(cfg.checkpoint)
    agent.model.eval()
    print(colored("Model loaded.", "green"))

    # ── Resolve episode files ──────────────────────────────────────────────────
    data_path = cfg.data
    if "*" in data_path or "?" in data_path:
        fps = sorted(glob(data_path))
    else:
        fps = [data_path]
    assert len(fps) > 0, f"No data files found at: {data_path}"
    print(colored(f"Found {len(fps)} file(s).", "blue"))

    data_key = cfg.get("data_key", "obs")

    # ── Load observations ──────────────────────────────────────────────────────
    print("Loading observations...")
    obs_all = load_episodes(fps, data_key=data_key)
    print(f"Total observations: {obs_all.shape}")

    # ── Task ID for multi-task models ──────────────────────────────────────────
    task_id = cfg.get("task_id", 0)
    task = torch.tensor([task_id], device=device) if cfg.multitask else None

    # ── Encode in batches ──────────────────────────────────────────────────────
    batch_size = cfg.get("batch_size", 1024)
    latents = []
    print(f"Encoding {obs_all.shape[0]} observations (batch_size={batch_size})...")
    with torch.no_grad():
        for start in range(0, obs_all.shape[0], batch_size):
            batch = obs_all[start : start + batch_size].to(device)
            z = agent.model.encode(batch, task)
            latents.append(z.cpu())
    Z = torch.cat(latents, dim=0)  # (N, latent_dim)
    d = Z.shape[1]
    print(colored(f"Latent matrix shape: {Z.shape}  (N={Z.shape[0]}, D={d})", "blue"))

    # ── Compute metrics ────────────────────────────────────────────────────────
    erank     = effective_rank(Z)
    ev_curve  = explained_variance_curve(Z)
    dim_stats = per_dimension_stats(Z)
    cos_stats = pairwise_cosine_stats(Z)

    norms = Z.float().norm(dim=-1)

    # ── Print verdict ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(colored(interpret(d, erank, ev_curve, dim_stats), "yellow"))
    print("-" * 60)
    print("Cosine similarity (pairwise, random subsample):")
    for k, v in cos_stats.items():
        print(f"  {k:20s}: {v:.4f}")
    print("-" * 60)
    print(f"Latent norms — mean: {norms.mean():.4f}, std: {norms.std():.4f}, "
          f"min: {norms.min():.4f}, max: {norms.max():.4f}")
    print("=" * 60)

    # ── Per-episode delta_t = ||z_{t+1} - z_t||_2 ────────────────────────────
    print()
    print("=" * 60)
    print("Per-episode latent step distances  delta_t = ||z_{t+1} - z_t||_2")
    print("=" * 60)

    import matplotlib.pyplot as plt

    all_episode_deltas = []
    with torch.no_grad():
        for ep_idx, p in enumerate(fps):
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and "traj_plans" in obj:
                steps = obj["traj_plans"]
                obs_list, reward_list = [], []
                for step in steps:
                    o = step["obs"]
                    if not isinstance(o, torch.Tensor):
                        o = torch.tensor(np.array(o), dtype=torch.float32)
                    obs_list.append(o.float())
                    r = step["reward"]
                    if not isinstance(r, torch.Tensor):
                        r = torch.tensor(float(r))
                    reward_list.append(r.float().reshape(()))
                ep_obs = torch.stack(obs_list, dim=0)
                rewards = torch.stack(reward_list).numpy()  # (T,)
            else:
                raise RuntimeError(f"Unsupported format in {p}")

            ep_latents = []
            for start in range(0, ep_obs.shape[0], batch_size):
                batch = ep_obs[start : start + batch_size].to(device)
                z = agent.model.encode(batch, task)
                ep_latents.append(z.cpu())
            Z_ep = torch.cat(ep_latents, dim=0)  # (T, latent_dim)

            deltas = (Z_ep[1:] - Z_ep[:-1]).norm(dim=-1)           # (T-1,)
            norms_ep = Z_ep[:-1].norm(dim=-1).clamp(min=1e-10)
            rel_deltas = (deltas / norms_ep).numpy()
            deltas = deltas.numpy()
            # align rewards with deltas: reward[t] -> transition t -> t+1
            rewards_aligned = rewards[:len(deltas)]
            all_episode_deltas.append((ep_idx, Path(p).name, deltas, rel_deltas, rewards_aligned))

    n_eps = len(all_episode_deltas)
    fig, axes = plt.subplots(n_eps, 2, figsize=(18, 3 * n_eps), squeeze=False)
    for row, (ep_idx, name, deltas, rel_deltas, rewards_aligned) in zip(axes, all_episode_deltas):
        ax_abs, ax_rel = row
        ts = range(len(deltas))

        # Left subplot: abs delta + reward (twin axis)
        color_delta, color_rew = "tab:blue", "tab:green"
        ax_abs.plot(ts, deltas, linewidth=0.9, color=color_delta, label=r"$\delta_t$")
        ax_abs.set_title(f"Episode {ep_idx}  —  {name}")
        ax_abs.set_xlabel("t")
        ax_abs.set_ylabel(r"$\|z_{t+1}-z_t\|_2$", color=color_delta)
        ax_abs.tick_params(axis="y", labelcolor=color_delta)
        ax_abs.grid(True, alpha=0.3)
        ax_rew = ax_abs.twinx()
        ax_rew.plot(ts, rewards_aligned, linewidth=0.9, color=color_rew,
                    alpha=0.7, label="reward")
        ax_rew.set_ylabel("reward", color=color_rew)
        ax_rew.tick_params(axis="y", labelcolor=color_rew)
        lines = ax_abs.get_lines() + ax_rew.get_lines()
        ax_abs.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper left")

        # Right subplot: rel delta + reward (twin axis)
        ax_rel.plot(ts, rel_deltas, linewidth=0.9, color="tab:orange",
                    label=r"$\delta_t / \|z_t\|$")
        ax_rel.set_title(f"Episode {ep_idx}  —  {name}  (relative)")
        ax_rel.set_xlabel("t")
        ax_rel.set_ylabel(r"$\|z_{t+1}-z_t\|_2 / \|z_t\|_2$", color="tab:orange")
        ax_rel.tick_params(axis="y", labelcolor="tab:orange")
        ax_rel.grid(True, alpha=0.3)
        ax_rew2 = ax_rel.twinx()
        ax_rew2.plot(ts, rewards_aligned, linewidth=0.9, color=color_rew,
                     alpha=0.7, label="reward")
        ax_rew2.set_ylabel("reward", color=color_rew)
        ax_rew2.tick_params(axis="y", labelcolor=color_rew)
        lines2 = ax_rel.get_lines() + ax_rew2.get_lines()
        ax_rel.legend(lines2, [l.get_label() for l in lines2], fontsize=7, loc="upper left")

    fig.suptitle(r"Latent step distances $\delta_t$  (abs & relative) with reward", fontsize=13)
    plt.tight_layout()

    out_dir = cfg.get("out_dir", ".")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_stem = cfg.get("plot_stem", "latent_deltas")
    delta_plot_path = Path(out_dir) / f"{plot_stem}.png"
    plt.savefig(delta_plot_path, dpi=150)
    print(f"Delta plot saved to {delta_plot_path}")
    plt.close()

    # ── Save all statistics to JSON ────────────────────────────────────────────
    import json

    ev = ev_curve.tolist()
    stats = {
        "effective_rank": erank,
        "max_rank": d,
        "erank_ratio": erank / d,
        "explained_variance": {
            "top1":  float(ev[0]),
            "top5":  float(ev[4])  if len(ev) > 4  else float(ev[-1]),
            "top20": float(ev[19]) if len(ev) > 19 else float(ev[-1]),
        },
        **{f"dim_{k}": v for k, v in dim_stats.items()},
        **{f"cos_{k}": v for k, v in cos_stats.items()},
        "norm_mean": float(norms.mean()),
        "norm_std":  float(norms.std()),
        "norm_min":  float(norms.min()),
        "norm_max":  float(norms.max()),
        "episodes": [
            {
                "ep_idx": ep_idx,
                "name": name,
                "delta_mean":     float(deltas.mean()),
                "delta_std":      float(deltas.std()),
                "delta_max":      float(deltas.max()),
                "delta_min":      float(deltas.min()),
                "rel_delta_mean": float(rel_deltas.mean()),
                "rel_delta_std":  float(rel_deltas.std()),
                "rel_delta_max":  float(rel_deltas.max()),
                "rel_delta_min":  float(rel_deltas.min()),
                "reward_sum":     float(rewards_aligned.sum()),
                "reward_mean":    float(rewards_aligned.mean()),
                "reward_std":     float(rewards_aligned.std()),
                "reward_max":     float(rewards_aligned.max()),
                "reward_min":     float(rewards_aligned.min()),
                "deltas":         deltas.tolist(),
                "rel_deltas":     rel_deltas.tolist(),
                "rewards":        rewards_aligned.tolist(),
            }
            for ep_idx, name, deltas, rel_deltas, rewards_aligned in all_episode_deltas
        ],
    }
    stats_path = Path(out_dir) / f"{plot_stem}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

    # ── Optional plots ─────────────────────────────────────────────────────────
    if cfg.get("plot", False):
        out_dir = cfg.get("out_dir", ".")
        save_plots(Z, erank, ev_curve, dim_stats, out_dir)

    return {"effective_rank": erank, "max_rank": min(Z.shape)}


if __name__ == "__main__":
    main()
