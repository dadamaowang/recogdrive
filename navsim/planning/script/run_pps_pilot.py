# -*- encoding: utf-8 -*-
'''
@File    :   run_pps_pilot.py
@Desc    :   PPS pilot, phase A (no training): does the signal PPS relies on exist?

For each high-risk scene: G CoTs x K shared planner noises; every rollout's CoT
prefixes are decoded by the frozen planner (Phi curve). On top of the PPS probe,
two control measurements are taken under the SAME K noises:

  Phi_ref     -- an independent K-noise re-score of the full CoT (fresh noises).
                 Phi_full vs Phi_ref measures how reliable a K-sample Phi is.
  Phi_ablate  -- the full CoT with the most-blamed segment REMOVED (its hidden
                 states dropped). If PPS blame is causal, removing the blamed
                 segment should raise Phi; removing a random other segment
                 (Phi_random) should help less.

Hidden states after a removed segment were computed with that segment in context,
so the ablation is a planner-side intervention (what the planner reads), not a
re-generation; that is exactly the quantity PPS credits.

Output: ${output_dir}/pps_pilot/pilot_rank{RANK}.pt; summarize with
scripts/pps/pilot/summarize_pilot.py, which prints the go / no-go table.
'''

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.negdrive.utils.mars import sample_planner_noise
from navsim.agents.negdrive.utils.pps import segment_deltas
from navsim.planning.training.agent_lightning_module import AgentLightningVLMRL
from navsim.planning.script.run_training_negdrive import build_datasets, negdrive_collate_fn

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def drop_segment(hidden: torch.Tensor, mask: torch.Tensor, prompt_len: int, lo: int, hi: int):
    """Remove response tokens [lo, hi) from hidden [1, L, D] / mask [1, L]."""
    keep = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    keep[prompt_len + lo: prompt_len + hi] = False
    return hidden[:, keep], mask[:, keep]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    torch.cuda.set_device(local_rank)

    split = OmegaConf.select(cfg, "diagnostic.split", default="train")
    max_scenes = OmegaConf.select(cfg, "diagnostic.max_scenes", default=None)
    pl.seed_everything(cfg.seed + rank, workers=True)
    rng = random.Random(cfg.seed + rank)

    agent: AbstractAgent = instantiate(cfg.agent)
    module = AgentLightningVLMRL(agent=agent, cfg=cfg).cuda()
    module.eval()
    K = module.mars_cfg.num_planner_samples
    head = agent.action_head

    train_data, val_data = build_datasets(cfg, agent)
    data = val_data if split == "val" else train_data
    metric_cache_loader = (
        module._get_val_metric_cache_loader() if split == "val" else head.metric_cache_loader
    )
    indices = list(range(len(data)))
    random.Random(cfg.seed).shuffle(indices)  # same scene subset on every rank count
    if max_scenes is not None:
        indices = indices[: int(max_scenes)]
    shard = Subset(data, indices[rank::world_size])
    dataloader = DataLoader(shard, collate_fn=negdrive_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Rank %d/%d: %d scenes (split=%s), G=%d, K=%d, P=%d",
                rank, world_size, len(shard), split, module.G, K, module.pps_cfg.max_segments)

    controls: Dict[Any, Dict[str, Any]] = {}

    def extra_hook(b, g, probe, last_hidden_states, gen_output, noise, diff_input, metric_cache):
        token = current_tokens[b]
        prompt_len = gen_output.response_start_idx
        full_h = last_hidden_states[b:b + 1]
        full_m = gen_output.attention_mask[b:b + 1]
        out: Dict[str, Any] = {}

        # (1) reliability: same CoT, fresh independent noises
        fresh = sample_planner_noise(
            full_h.shape[0], K, module._planner_num_steps(),
            head.config.action_horizon, head.config.action_dim,
            full_h.device, noise["init"].dtype,
        )
        out["cost_ref"] = module.score_hidden_states(
            0, full_h, full_m, fresh, _slice(diff_input, b), token, metric_cache
        )

        # (2) causal check: drop the most-blamed segment vs a random other one
        deltas = segment_deltas(probe["phi"])
        bounds = probe["boundaries"]
        if len(deltas) >= 2 and deltas.min().item() < 0:
            j_blame = int(torch.argmin(deltas).item())
            others = [j for j in range(len(deltas)) if j != j_blame]
            j_rand = rng.choice(others)
            for name, j in (("blame", j_blame), ("random", j_rand)):
                lo, hi = bounds[j], min(bounds[j + 1], probe["num_real"])
                h, m = drop_segment(full_h, full_m, prompt_len, lo, hi)
                out[f"cost_drop_{name}"] = module.score_hidden_states(
                    b, h, m, noise, diff_input, token, metric_cache
                )
                out[f"seg_{name}"] = j
        controls[(token, g)] = out

    records: List[Dict[str, Any]] = []
    current_tokens: List[str] = []
    for batch_idx, (features, _targets, tokens_list) in enumerate(dataloader):
        current_tokens = list(tokens_list)
        bundle, metrics, probes = module.collect_rollouts_with_probes(
            features, current_tokens,
            probe="all",
            metric_cache_loader=metric_cache_loader,
            do_sample=True,
            extra_hook=extra_hook,
        )
        module._cleanup_rollout_bundle(bundle)
        for (b, g), probe in probes.items():
            token = current_tokens[b]
            records.append({
                "token": token,
                "g": g,
                "phi": probe["phi"],
                "costs": probe["costs"],          # [P+1, K]
                "boundaries": probe["boundaries"],
                "num_real": probe["num_real"],
                "segments": probe["segments"],
                "score": metrics["score"][b, g],  # [K]
                "ego_progress": metrics["ego_progress"][b, g],
                **controls.pop((token, g), {}),
            })
        logger.info("Rank %d: batch %d done (%d rollouts)", rank, batch_idx, len(records))

    out_dir = Path(cfg.output_dir) / "pps_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"records": records, "G": module.G, "K": K, "P": module.pps_cfg.max_segments, "split": split},
        out_dir / f"pilot_rank{rank}.pt",
    )
    logger.info("Saved %d rollouts to %s", len(records), out_dir)


def _slice(diff_input, b):
    """diff_input restricted to scene b (as a batch of one)."""
    from transformers.feature_extraction_utils import BatchFeature
    return BatchFeature({k: v[b:b + 1] for k, v in diff_input.items()})


if __name__ == "__main__":
    main()
