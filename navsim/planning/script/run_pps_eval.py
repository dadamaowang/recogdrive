# -*- encoding: utf-8 -*-
'''
@File    :   run_pps_eval.py
@Desc    :   Reasoning-gain / blame evaluation for PPS (planner-potential shaping).

For each scene: sample G CoTs, decode each with K planner noises, and probe
every rollout's CoT prefixes with the frozen planner. Saves per-rollout
potentials Phi at the segment boundaries plus the segment texts:

    Phi(prompt only)  -> the plan without reasoning (original ReCogDrive behaviour)
    Phi(full CoT)     -> the plan the policy actually executes
    reasoning gain    =  Phi(full) - Phi(prompt)   (> 0: the CoT made the plan safer)

Each rank writes ${output_dir}/pps_eval/probes_rank{RANK}.pt; summarize with
scripts/pps/summarize_pps_eval.py. Use with a trained LoRA (agent.vlm_lora_path)
to compare base / NSR / MARS / PPS policies.

Uses the same hydra config as run_training_negdrive.py; extra options:
    +diagnostic.split=val|train
    +diagnostic.max_batches=N
    +diagnostic.do_sample=true
'''

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

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

    pl.seed_everything(cfg.seed + rank, workers=True)

    agent: AbstractAgent = instantiate(cfg.agent)
    module = AgentLightningVLMRL(agent=agent, cfg=cfg).cuda()
    module.eval()

    train_data, val_data = build_datasets(cfg, agent)
    data = val_data if split == "val" else train_data
    metric_cache_loader = (
        module._get_val_metric_cache_loader() if split == "val" else agent.action_head.metric_cache_loader
    )
    shard = Subset(data, list(range(rank, len(data), world_size)))
    dataloader = DataLoader(shard, collate_fn=negdrive_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info(
        "Rank %d/%d: %d scenes (split=%s), G=%d, K=%d, P=%d",
        rank, world_size, len(shard), split, module.G,
        module.mars_cfg.num_planner_samples, module.pps_cfg.max_segments,
    )

    records: List[Dict[str, Any]] = []
    for batch_idx, (features, _targets, tokens_list) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        tokens_list = list(tokens_list)
        bundle, metrics, probes = module.collect_rollouts_with_probes(
            features, tokens_list,
            probe="all",
            metric_cache_loader=metric_cache_loader,
            do_sample=do_sample,
        )
        module._cleanup_rollout_bundle(bundle)

        for (b, g), probe in probes.items():
            records.append({
                "token": tokens_list[b],
                "g": g,
                "phi": probe["phi"],
                "boundaries": probe["boundaries"],
                "segments": probe["segments"],
                "nc": metrics["no_at_fault_collisions"][b, g],
                "dac": metrics["drivable_area_compliance"][b, g],
                "ego_progress": metrics["ego_progress"][b, g],
                "score": metrics["score"][b, g],
            })
        logger.info("Rank %d: batch %d done (%d rollouts)", rank, batch_idx, len(records))

    out_dir = Path(cfg.output_dir) / "pps_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probes_rank{rank}.pt"
    torch.save(
        {
            "records": records,
            "G": module.G,
            "K": module.mars_cfg.num_planner_samples,
            "P": module.pps_cfg.max_segments,
            "split": split,
            "vlm_lora_path": OmegaConf.select(cfg, "agent.vlm_lora_path", default=None),
        },
        out_path,
    )
    logger.info("Saved %d rollouts to %s", len(records), out_path)


if __name__ == "__main__":
    main()
