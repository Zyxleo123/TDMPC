# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Training

All training is launched via SLURM by editing `~/slurm/test.sh` and running `bash ~/slurm/test.sh`. The virtual environment is at `/zfsauton/scratch/yixiz/tdmpc_env`.

```bash
# Single-task online RL (state observations)
python -m tdmpc2.train task=dog-run steps=7000000

# RGB observations
python -m tdmpc2.train task=walker-walk obs=rgb

# DriveEnv (common usage)
python -m tdmpc2.train task=driveenv-single_agent obs=rgb actor_mode=sac n_envs=32

# Resume from latest checkpoint
python -m tdmpc2.train task=dog-run resume=True

# Evaluate
python -m tdmpc2.evaluate task=dog-run checkpoint=/path/to/checkpoint.pt
```

Checkpoints are saved to `{work_dir}/models/{step}.pt`. Work dir defaults to `logs/{task}/{seed}/{wandb_project}/{exp_name}/`.

## Configuration

Uses **Hydra**. Main config: `tdmpc2/config.yaml`. All CLI args are Hydra overrides.

Key config flags:
- `model_size`: {1, 5, 19, 48, 317} — scales `latent_dim`, `mlp_dim`, `enc_dim`, `num_channels`
- `obs`: `"state"` or `"rgb"` — selects encoder type
- `actor_mode`: `"sac"` | `"bc"` | `"awac"` | `"residual"`
- `enc_identity=true` — skips learned encoder; uses raw flattened pixels as latent (auto-sets `latent_dim` to flat pixel size in `train.py` before model init)
- `disable_dynamics=true` — disables learned dynamics model
- `resume=True` — auto-resumes from latest checkpoint in `work_dir`
- `priority_buffer_ratio` — fraction of buffer using Z-score peak-detected high-reward sampling
- `wandb_project`, `disable_wandb` — W&B logging (disabled by default)

Task-specific configs live in `configs/` (e.g., `configs/waypointvec-v0.yaml`).

## Architecture Overview

**Agent**: `tdmpc2/tdmpc25.py` — `TDMPC2` class. Owns planning (`plan()`/`_plan()`), acting (`act()`/`act_vec()`), and training (`update()`).

**World Model**: `tdmpc2/common/world_model.py`
- `_encoder`: `ModuleDict` keyed by obs type (`state`, `rgb`, `waypointvec`). Built by `enc()` in `layers.py`.
- `_dynamics`: Latent transition model
- `_reward`: Predicts reward from (latent, action)
- `_pi`: Policy network
- `_Qs` / `_target_Qs`: Ensemble of 5 distributional critics, vectorized with `functorch.vmap`

**Trainers**: `tdmpc2/trainer/`
- `OnlineTrainer` — single-task online RL (collect → update loop)
- `OfflineTrainer` — multi-task offline RL (batch learning)

**Replay Buffer**: `tdmpc2/common/buffer_25.py` — torchrl-based (`LazyTensorStorage` + `SliceSampler`). Optional priority buffer via Z-score peak detection.

**Env factory**: `tdmpc2/envs/__init__.py` — `make_env()` dispatches by task name to DMControl, Meta-World, ManiSkill2, TorchDriveEnv, WaypointVecEnv, etc. Wraps with `TensorWrapper`, `PixelWrapper`, `MultitaskWrapper` as needed.

**Key layers** (`tdmpc2/common/layers.py`):
- `Ensemble` — vmap-vectorized ensemble
- `NormedLinear` — Linear + LayerNorm + Mish + Dropout
- `SimNorm` — Simplicial normalization (used at encoder output)
- `enc()` — factory returning encoder `ModuleDict`; `enc_identity=true` replaces RGB encoder with `PixelPreprocess() + Flatten()`

**VectorNet encoder** (`tdmpc2/common/vectornet.py`): Used for `waypointvec` observations. Encodes agent histories, lanes, traffic lights, waypoints from (151, 9, 10) tensors.

## Multi-task

Multi-task mode (`cfg.multitask=True`) uses learned per-task embeddings (96-dim by default) concatenated to latent state before Q/policy/dynamics networks. `OfflineTrainer` is used. Task sets: `mt30`, `mt80`.

## Notifications & Run Monitoring

`~/slurm/notify.py` sends email to `yixiz@andrew.cmu.edu` via CMU SMTP and can poll the W&B API for run status.

**Setup** (add to `~/.bashrc`):
```bash
export NOTIFY_EMAIL_PASS="your-andrew-password"
export WANDB_API_KEY="your-wandb-api-key"   # or run `wandb login` once
```

**Usage in `test.sh`** — notify on finish or failure with W&B metrics:
```bash
srun $NODES python -m tdmpc2.train $BASE exp_name=my_exp ... \
  && python ~/slurm/notify.py "my_exp done" --wandb-project 3.20-tests --wandb-exp my_exp \
  || python ~/slurm/notify.py "my_exp FAILED" --wandb-project 3.20-tests --wandb-exp my_exp
```

**Poll until a run finishes**, then email (useful when launched separately):
```bash
python ~/slurm/notify.py --poll --wandb-project 3.20-tests --wandb-exp my_exp &
```

W&B entity is `zyxleo`. The script fetches the most recent run matching `exp_name` and includes state, URL, and key metrics (reward, loss, step) in the email.

## Data Flow (Online Training)

1. Env step → replay buffer (torchrl)
2. Sample batch of trajectories (horizon+1 steps)
3. Encode observations → latent states
4. Predict: forward dynamics, reward, Q-values over horizon
5. Compute losses: TD, reward, consistency, policy
6. Update: encoder, dynamics, reward, Q-networks, policy
7. Periodic eval rollout
