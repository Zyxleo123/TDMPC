from typing import Optional, Union, List
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


def asymmetric_l2_loss(pred, target, tau=0.5):
    """Computes the asymmetric L2 regression loss between predictions and soft targets."""
    u = target - pred
    return torch.abs(tau - (u<0).float()) * u**2


@torch.jit.script
def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


@torch.jit.script
def _gaussian_logprob(residual):
    return residual - 0.5 * torch.log(2 * torch.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


@torch.jit.script
def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


DREG_BINS = None


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    global DREG_BINS
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    if DREG_BINS is None:
        DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
    return symexp(x)


@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    @staticmethod
    # def from_beta(betas: np.ndarray, device:Optional[torch.device("cpu")] = 'cpu'):
    def from_beta(betas: np.ndarray, device: Optional[torch.device] = 'cpu'):
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
                torch.tensor(betas, device=device, dtype=torch.float32),
                torch.tensor(alphas, device=device, dtype=torch.float32),
                torch.tensor(alphas_cumprod, device=device, dtype=torch.float32),
                torch.tensor(alphas_cumprod_prev, device=device, dtype=torch.float32),
                torch.tensor(sqrt_alphas_cumprod, device=device, dtype=torch.float32),
                torch.tensor(sqrt_one_minus_alphas_cumprod, device=device, dtype=torch.float32),
                torch.tensor(log_one_minus_alphas_cumprod, device=device, dtype=torch.float32),
                torch.tensor(sqrt_recip_alphas_cumprod, device=device, dtype=torch.float32),
                torch.tensor(sqrt_recipm1_alphas_cumprod, device=device, dtype=torch.float32),
                torch.tensor(posterior_variance, device=device, dtype=torch.float32),
                torch.tensor(posterior_log_variance_clipped, device=device, dtype=torch.float32),
                torch.tensor(posterior_mean_coef1, device=device, dtype=torch.float32),
                torch.tensor(posterior_mean_coef2, device=device, dtype=torch.float32)
            )

    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas