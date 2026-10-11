# -*- encoding: utf-8 -*-
'''
@File    :   run_mars_diagnostic.py
@Desc    :   Planner-noise vs. reasoning variance diagnostic for NegDrive (MARS motivation).

For each scene: sample G CoTs from the VLM, decode each with K planner noises
through the frozen diffusion planner, and score all G x K trajectories with PDM.
Each rank writes its shard of per-sample metrics [B, G, K] to
    ${output_dir}/mars_diagnostic/metrics_rank{RANK}.pt
Merge and summarize the shards with scripts/mars/summarize_mars_diagnostic.py.

Uses the same hydra config as run_training_negdrive.py; extra options:
    +diagnostic.split=val|train      which dataset to sample scenes from (default: val)
    +diagnostic.max_batches=N        stop after N batches per rank (default: all)
    +diagnostic.do_sample=true       sample CoTs (false = greedy, G identical CoTs)
'''

import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningVLMRL
from navsim.planning.script.run_training_negdrive import build_datasets, negdrive_collate_fn

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    torch.cuda.set_device(local_rank)

    split = OmegaConf.select(cfg, "diagnostic.split", default="val")
    max_batches = OmegaConf.select(cfg, "diagnostic.max_batches", default=None)
    do_sample = bool(OmegaConf.select(cfg, "diagnostic.do_sample", default=True))

    # different seed per rank so CoT / planner noises are independent across shards
    pl.seed_everything(cfg.seed + rank, workers=True)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    module = AgentLightningVLMRL(agent=agent, cfg=cfg).cuda()
    module.eval()

    train_data, val_data = build_datasets(cfg, agent)
    data = val_data if split == "val" else train_data
    metric_cache_loader = (
        module._get_val_metric_cache_loader() if split == "val" else agent.action_head.metric_cache_loader
    )
    shard = Subset(data, list(range(rank, len(data), world_size)))
    dataloader = DataLoader(
        shard,
        collate_fn=negdrive_collate_fn,
        **cfg.dataloader.params,
        shuffle=False,
    )
    logger.info(
        "Rank %d/%d: %d scenes (split=%s), G=%d, K=%d, CRN=%s",
        rank, world_size, len(shard), split, module.G,
        module.mars_cfg.num_planner_samples, module.mars_cfg.common_random_numbers,
    )

    tokens: List[str] = []
    metrics: Dict[str, List[torch.Tensor]] = {}
    for batch_idx, (features, _targets, tokens_list) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        bundle, batch_metrics = module._collect_rollouts_mars(
            features, list(tokens_list),
            metric_cache_loader=metric_cache_loader,
            do_sample=do_sample,
        )
        module._cleanup_rollout_bundle(bundle)

        tokens.extend(tokens_list)
        for key, value in batch_metrics.items():
            metrics.setdefault(key, []).append(value)
        logger.info("Rank %d: batch %d done (%d scenes)", rank, batch_idx, len(tokens))

    out_dir = Path(cfg.output_dir) / "mars_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_rank{rank}.pt"
    torch.save(
        {
            "tokens": tokens,
            "metrics": {k: torch.cat(v, dim=0) for k, v in metrics.items()},  # [N, G, K]
            "G": module.G,
            "K": module.mars_cfg.num_planner_samples,
            "common_random_numbers": module.mars_cfg.common_random_numbers,
            "do_sample": do_sample,
            "split": split,
        },
        out_path,
    )
    logger.info("Saved %d scenes to %s", len(tokens), out_path)


if __name__ == "__main__":
    main()
