# -*- encoding: utf-8 -*-
'''
@File    :   mars.py
@Desc    :   MARS (noise-MARginalized negative reinforcement for reaSoning) utilities.

The VLM samples a CoT c ~ pi(.|s); the frozen diffusion planner maps it to a
trajectory tau = D(h(c), eps) with planner noise eps. MARS scores every CoT
under K planner noises and uses the noise-averaged cost as its advantage, so
the CoT is credited with its expected failure, not with a single noise draw.

All tensors here are laid out as [B, G, K]:
    B = scenes in the batch, G = CoT rollouts per scene, K = planner noises per CoT.
'''

from dataclasses import dataclass
from typing import Dict, Optional

import torch


PDM_PERFECT_SCORE = 1.0


@dataclass
class MarsConfig:
    num_planner_samples: int = 4        # K: planner noises per CoT
    common_random_numbers: bool = True  # share the K noises across the G CoTs of a scene
    progress_coef: float = 0.5          # beta: weight of the progress hinge on safe samples
    progress_min: float = 0.6           # EP below this on a safe sample counts as regression
    baseline: str = "none"              # none (pure negative) | group_mean (subtract scene mean cost)
    advantage_eps: float = 1e-6         # skip updates with |A| below this

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "MarsConfig":
        cfg = dict(cfg or {})
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        out = cls(**known)
        if out.num_planner_samples < 1:
            raise ValueError("mars_cfg.num_planner_samples must be >= 1")
        if out.baseline not in ("none", "group_mean"):
            raise ValueError(f"Unknown mars_cfg.baseline: {out.baseline!r}")
        return out


def sample_planner_noise(
    batch_size: int,
    num_samples: int,
    num_steps: int,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Draw x_T and per-step denoising noise for K planner samples of each scene."""
    return {
        "init": torch.randn(batch_size, num_samples, horizon, action_dim, device=device, dtype=dtype),
        "steps": torch.randn(batch_size, num_samples, num_steps, horizon, action_dim, device=device, dtype=dtype),
    }


def compute_failure(nc: torch.Tensor, dac: torch.Tensor) -> torch.Tensor:
    """1.0 if the trajectory collides (NC) or leaves the drivable area (DAC), else 0.0."""
    return ((nc < PDM_PERFECT_SCORE) | (dac < PDM_PERFECT_SCORE)).float()


def compute_mars_cost(
    fail: torch.Tensor,
    ego_progress: torch.Tensor,
    progress_coef: float,
    progress_min: float,
) -> torch.Tensor:
    """
    Per-sample cost in [0, 1 + progress_coef]:
        cost = fail + beta * (1 - fail) * relu(ep_min - EP) / ep_min

    PDM's EP is already normalized by the PDM-Closed reference progress, and is 0
    on unsafe samples, so the hinge is only applied to safe samples. It penalizes
    "safe by stopping" without rewarding progress on scenes that need a stop
    (where the reference progress is < 5 m, PDM sets EP = 1).
    """
    if progress_coef <= 0.0 or progress_min <= 0.0:
        return fail
    hinge = torch.relu(progress_min - ego_progress) / progress_min
    return fail + progress_coef * (1.0 - fail) * hinge


def compute_mars_advantages(cost: torch.Tensor, baseline: str = "none") -> torch.Tensor:
    """
    cost: [B, G, K] -> advantages [B, G].

    none:       A = -E_eps[cost]  (<= 0, only negative signals, NSR-style; unbiased
                for -grad E_c E_eps[cost]).
    group_mean: A = -(E_eps[cost] - mean_g E_eps[cost])  (GRPO-style, without std
                normalization, so scenes where all CoTs are equally bad get no update).
    """
    expected_cost = cost.mean(dim=-1)  # [B, G]
    if baseline == "none":
        return -expected_cost
    if baseline == "group_mean":
        return -(expected_cost - expected_cost.mean(dim=1, keepdim=True))
    raise ValueError(f"Unknown baseline: {baseline!r}")


def decompose_variance(x: torch.Tensor) -> Dict[str, float]:
    """
    Law-of-total-variance split of a per-sample signal x [B, G, K] within each scene:
        Var(x) = E_c[Var_eps(x | c)]  +  Var_c(E_eps[x | c])
                 (planner noise)         (reasoning)

    Both terms use unbiased estimators (needs K >= 2 for the noise term and
    G >= 2 for the reasoning term); the reasoning term subtracts the noise
    contribution to the variance of the K-sample means. Results are averaged
    over scenes.
    """
    B, G, K = x.shape
    x = x.float()
    stats: Dict[str, float] = {}

    cot_mean = x.mean(dim=-1)  # [B, G]
    if K >= 2:
        noise_var = x.var(dim=-1, unbiased=True)  # [B, G]
    else:
        noise_var = torch.zeros_like(cot_mean)
    noise_term = noise_var.mean(dim=1)  # [B]

    if G >= 2:
        between = cot_mean.var(dim=1, unbiased=True)  # [B]
        reasoning_term = (between - noise_term / K).clamp(min=0.0)
    else:
        reasoning_term = torch.zeros_like(noise_term)

    total = noise_term + reasoning_term
    stats["var_noise"] = noise_term.mean().item()
    stats["var_reasoning"] = reasoning_term.mean().item()
    stats["var_total"] = total.mean().item()
    denom = max(stats["var_total"], 1e-12)
    stats["noise_fraction"] = stats["var_noise"] / denom if stats["var_total"] > 0 else 0.0
    return stats


def flip_statistics(fail: torch.Tensor) -> Dict[str, float]:
    """
    fail: [B, G, K] binary.
        flip_rate:          fraction of CoTs whose safety depends on planner noise (0 < p < 1).
        decisive_scene_rate: fraction of scenes with one CoT always safe and another always failing.
        fail_rate:          overall failure rate.
    """
    p = fail.float().mean(dim=-1)  # [B, G]
    flip = ((p > 0) & (p < 1)).float()
    always_safe = (p == 0).any(dim=1)
    always_fail = (p == 1).any(dim=1)
    return {
        "flip_rate": flip.mean().item(),
        "decisive_scene_rate": (always_safe & always_fail).float().mean().item(),
        "fail_rate": fail.float().mean().item(),
        "mean_fail_prob": p.mean().item(),
    }
