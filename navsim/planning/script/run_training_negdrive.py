# -*- encoding: utf-8 -*-
'''
@File    :   run_training_negdrive.py
@Time    :   2026/01/03 19:47:39
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   runnable, developing, draft finished 
@Desc    :   【正在优化】
'''

from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
 
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from swanlab.integration.pytorch_lightning import SwanLabLogger


from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.planning.training.dataset import Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningVLMRL, VRAMMonitor, OptimizerHealthMonitor, IterLoRAModelCheckpoint



import sys

logger = logging.getLogger(__name__)


CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"   


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')  
torch.cuda.set_per_process_memory_fraction(0.95, 0) 


def negdrive_collate_fn(
        batch: List[
            Tuple[
                Dict[str, torch.Tensor],    # features 
                Dict[str, torch.Tensor],    # targets 
                str     # tokens/prompt
                ]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:

    features_list, targets_list, tokens_list = zip(*batch)

    # extract features 
    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()   

    # extrace image_path_tensor 
    image_path_tensor = torch.stack([features['image_path_tensor'] for features in features_list], dim=0).cpu() 

    # extract targets
    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        'image_path_tensor': image_path_tensor
    }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list


def load_token_list(tokens_path: Path) -> List[str]:
    """Load scene tokens from a newline-delimited text file."""
    if not tokens_path.exists():
        raise FileNotFoundError(f"High-risk tokens file not found: {tokens_path}")

    tokens: List[str] = []
    with tokens_path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)

    if not tokens:
        raise ValueError(f"No tokens found in {tokens_path}")

    return tokens


def apply_train_log_names(scene_filter: SceneFilter, train_logs: List[str]) -> SceneFilter:
    """Restrict scene filter to official training logs."""
    if scene_filter.log_names is not None:
        scene_filter.log_names = [
            log_name for log_name in scene_filter.log_names if log_name in train_logs
        ]
    else:
        scene_filter.log_names = list(train_logs)
    return scene_filter


def warn_missing_metric_cache(
    tokens: List[str],
    metric_cache_path: str,
    rank: int = 0,
) -> None:
    """Warn if high-risk tokens are missing from the metric cache index."""
    if rank != 0:
        return
    if not metric_cache_path:
        return

    cache_path = Path(metric_cache_path)
    if not cache_path.exists():
        logger.warning("Metric cache path does not exist: %s", metric_cache_path)
        return

    loader = MetricCacheLoader(cache_path)
    cache_tokens = set(loader.metric_cache_paths.keys())
    missing = set(tokens) - cache_tokens
    if missing:
        logger.warning(
            "High-risk tokens missing from metric cache: %d / %d (path: %s)",
            len(missing),
            len(tokens),
            metric_cache_path,
        )
    else:
        logger.info("All %d high-risk tokens found in metric cache.", len(tokens))


def build_experiment_config(cfg: DictConfig) -> Dict[str, Any]:
    """Build a compact config dict for SwanLab (avoid huge log_names / Hydra dumps)."""
    scene_filter = OmegaConf.select(cfg, "train_test_split.scene_filter", default={})
    log_names = scene_filter.get("log_names") if scene_filter else None
    tokens = scene_filter.get("tokens") if scene_filter else None

    return {
        "experiment_name": cfg.experiment_name,
        "split": cfg.get("split", None),
        "seed": cfg.seed,
        "train_test_split": {
            "data_split": str(OmegaConf.select(cfg, "train_test_split.data_split", default="")),
            "scene_filter": {
                "num_history_frames": OmegaConf.select(
                    cfg, "train_test_split.scene_filter.num_history_frames", default=None
                ),
                "num_future_frames": OmegaConf.select(
                    cfg, "train_test_split.scene_filter.num_future_frames", default=None
                ),
                "num_log_names": len(log_names) if log_names else 0,
                "num_tokens": len(tokens) if tokens else 0,
            },
        },
        "train_scene_subset": cfg.get("train_scene_subset", None),
        "val_scene_subset": cfg.get("val_scene_subset", None),
        "high_risk": {
            "tokens_path": str(OmegaConf.select(cfg, "high_risk.tokens_path", default="")),
            "val_tokens_path": str(OmegaConf.select(cfg, "high_risk.val_tokens_path", default="")),
            "include_val_logs": bool(OmegaConf.select(cfg, "high_risk.include_val_logs", default=True)),
        },
        "dataloader": OmegaConf.to_container(cfg.dataloader.params, resolve=True),
        "trainer": OmegaConf.to_container(cfg.trainer.params, resolve=True),
        "agent": {
            "rl_algorithm": OmegaConf.select(cfg, "agent.rl_algorithm", default=None),
            "vlm_type": OmegaConf.select(cfg, "agent.vlm_type", default=None),
            "vlm_size": OmegaConf.select(cfg, "agent.vlm_size", default=None),
            "vlm_lr": OmegaConf.select(cfg, "agent.vlm_lr", default=None),
            "lora_r": OmegaConf.select(cfg, "agent.lora_r", default=None),
            "lora_alpha": OmegaConf.select(cfg, "agent.lora_alpha", default=None),
            "lora_dropout": OmegaConf.select(cfg, "agent.lora_dropout", default=None),
            "per_sample_rollout": OmegaConf.select(cfg, "agent.per_sample_rollout", default=None),
            "bag_g": OmegaConf.select(cfg, "agent.bag_g", default=None),
            "max_text_tokens": OmegaConf.select(cfg, "agent.max_text_tokens", default=None),
            "max_padding_len": OmegaConf.select(cfg, "agent.max_padding_len", default=None),
            "rollout_temperature": OmegaConf.select(cfg, "agent.rollout_temperature", default=None),
            "val_do_sample": OmegaConf.select(cfg, "agent.val_do_sample", default=None),
            "pass_cot_token_only": OmegaConf.select(cfg, "agent.pass_cot_token_only", default=None),
            "grpo_cfg": OmegaConf.to_container(cfg.agent.grpo_cfg, resolve=True),
        },
        "log_split_stats": {
            "num_train_logs": len(getattr(cfg, "train_logs", []) or []),
            "num_val_logs": len(getattr(cfg, "val_logs", []) or []),
        },
    }


def build_train_scene_filter(cfg: DictConfig) -> SceneFilter:
    """
    Build training SceneFilter according to train_scene_subset.
    :param cfg: omegaconf dictionary
    :return: scene filter for training set
    """
    base_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    subset = cfg.get("train_scene_subset", "full")

    if subset == "full":
        return apply_train_log_names(base_filter, cfg.train_logs)

    if subset == "high_risk":
        tokens_path = Path(cfg.high_risk.tokens_path)
        tokens = load_token_list(tokens_path)
        logger.info(
            "Using high-risk training subset: %d tokens from %s",
            len(tokens),
            tokens_path,
        )

        metric_cache_path = OmegaConf.select(cfg, "agent.metric_cache_path", default=None)
        warn_missing_metric_cache(
            tokens,
            metric_cache_path,
            rank=int(os.getenv("RANK", 0)),
        )

        include_val_logs = bool(
            OmegaConf.select(cfg, "high_risk.include_val_logs", default=True)
        )
        if include_val_logs:
            base_filter.log_names = list(cfg.train_logs) + list(cfg.val_logs)
            logger.info(
                "High-risk training searches train_logs + val_logs (%d logs).",
                len(base_filter.log_names),
            )
        else:
            base_filter.log_names = list(cfg.train_logs)
        base_filter.tokens = tokens
        return base_filter

    raise ValueError(
        f"Unknown train_scene_subset: {subset!r}. Expected 'full' or 'high_risk'."
    )


def resolve_validation_paths(cfg: DictConfig) -> Tuple[Path, Path, Optional[str]]:
    """
    Resolve navsim log, sensor blob, and metric cache paths for validation.
    """
    subset = cfg.get("val_scene_subset", "full")
    open_scene = Path(os.environ.get("OPENSCENE_DATA_ROOT", "/root/navsim_workspace/dataset"))

    if subset == "full":
        metric_cache_path = OmegaConf.select(cfg, "agent.metric_cache_path", default=None)
        return Path(cfg.navsim_log_path), Path(cfg.sensor_blobs_path), metric_cache_path

    if subset == "high_risk_test":
        metric_cache_path = OmegaConf.select(
            cfg, "validation.metric_cache_path", default=None
        )
        return (
            open_scene / "navsim_logs" / "test",
            open_scene / "sensor_blobs" / "test",
            metric_cache_path,
        )

    raise ValueError(
        f"Unknown val_scene_subset: {subset!r}. Expected 'full' or 'high_risk_test'."
    )


def build_val_scene_filter(cfg: DictConfig) -> SceneFilter:
    """Build validation SceneFilter according to val_scene_subset."""
    subset = cfg.get("val_scene_subset", "full")

    if subset == "full":
        val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        if val_scene_filter.log_names is not None:
            val_scene_filter.log_names = [
                log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs
            ]
        else:
            val_scene_filter.log_names = list(cfg.val_logs)
        return val_scene_filter

    if subset == "high_risk_test":
        tokens_path = Path(cfg.high_risk.val_tokens_path)
        tokens = load_token_list(tokens_path)
        logger.info(
            "Using high-risk navtest validation subset: %d tokens from %s",
            len(tokens),
            tokens_path,
        )

        metric_cache_path = OmegaConf.select(
            cfg, "validation.metric_cache_path", default=None
        )
        warn_missing_metric_cache(
            tokens,
            metric_cache_path,
            rank=int(os.getenv("RANK", 0)),
        )

        val_scene_filter: SceneFilter = instantiate(cfg.validation.scene_filter)
        val_scene_filter.log_names = None
        val_scene_filter.tokens = tokens
        return val_scene_filter

    raise ValueError(
        f"Unknown val_scene_subset: {subset!r}. Expected 'full' or 'high_risk_test'."
    )


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset

    SceneFilter cfg: config/common/train_test_split/scene_filter/navtrain.yaml
    train_logs / val_logs: config/training/default_train_val_test_log_split.yaml

    train_scene_subset:
      - full: all scenes in train_logs (default)
      - high_risk: (train_logs + val_logs if high_risk.include_val_logs) intersect
        high_risk.tokens_path whitelist

    val_scene_subset:
      - full: all scenes in val_logs on trainval (default)
      - high_risk_test: navtest test split intersect high_risk.val_tokens_path
    """
    train_scene_filter: SceneFilter = build_train_scene_filter(cfg)
    val_scene_filter: SceneFilter = build_val_scene_filter(cfg)

    train_data_path = Path(cfg.navsim_log_path)
    train_sensor_blobs_path = Path(cfg.sensor_blobs_path)
    val_data_path, val_sensor_blobs_path, _ = resolve_validation_paths(cfg)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=train_sensor_blobs_path,
        data_path=train_data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True       # NOTE 如果训 VLM, 这里需要设置成 True; 
    )

    if cfg.get("train_scene_subset", "full") == "high_risk" and int(os.getenv("RANK", 0)) == 0:
        requested = len(train_scene_filter.tokens or [])
        loaded = len(train_scene_loader.tokens)
        if loaded < requested:
            logger.warning(
                "Loaded %d high-risk scenes from dataset, but %d tokens were listed "
                "(%d not found in trainval logs).",
                loaded,
                requested,
                requested - loaded,
            )
        else:
            logger.info("Loaded %d high-risk training scenes.", loaded)

    val_scene_loader = SceneLoader(
        sensor_blobs_path=val_sensor_blobs_path,
        data_path=val_data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )

    if cfg.get("val_scene_subset", "full") == "high_risk_test" and int(os.getenv("RANK", 0)) == 0:
        requested = len(val_scene_filter.tokens or [])
        loaded = len(val_scene_loader.tokens)
        if loaded < requested:
            logger.warning(
                "Loaded %d high-risk navtest scenes for validation, but %d tokens were listed "
                "(%d not found in test logs).",
                loaded,
                requested,
                requested - loaded,
            )
        else:
            logger.info("Loaded %d high-risk navtest validation scenes.", loaded)

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,    # VLM fine-tine must None(null)
        force_cache_computation=False,  # VLM fine-tine must False
    )   

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=False,
    )

    return train_data, val_data



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training NegDrive Agent 
    :param cfg: omegaconf dictionary
    """

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))    
    rank = int(os.getenv('RANK', 0))
    logger.info("Initialized distributed training with local_rank=%d, world_size=%d, rank=%d", local_rank, world_size, rank)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)


    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")


    logger.info(f"Path where all results are stored: {cfg.output_dir}")


    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent) 


    logger.info("Building Lightning Module")    
    lightning_module = AgentLightningVLMRL(
        agent=agent,
        cfg=cfg
    )

    logger.info("Building SceneLoader and Dataset...")
    train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building DataLoader")
    train_dataloader = DataLoader(
        train_data,
        collate_fn=negdrive_collate_fn, 
        **cfg.dataloader.params,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_data,
        collate_fn=negdrive_collate_fn,
        **cfg.dataloader.params,
        shuffle=False
    )

    experiment_config = build_experiment_config(cfg)
    swanlab_mode = os.environ.get("SWANLAB_MODE", "online")
    swanlab_logger = SwanLabLogger(
        project="negdrive",
        experiment_name=cfg.experiment_name,
        config=experiment_config,
        log_dir=cfg.output_dir,
        mode=swanlab_mode,
    )

    logger.info("Building Trainer (SwanLab mode=%s)", swanlab_mode)
    trainer = pl.Trainer(
        **cfg.trainer.params,
        enable_checkpointing=False,
        logger=swanlab_logger,
        callbacks=[
            VRAMMonitor(), 
            LearningRateMonitor(logging_interval="step"),
            OptimizerHealthMonitor(),
            IterLoRAModelCheckpoint(    # TODO 改成监控 rewards 的
                dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            ),
        ]
        )


    if cfg.get("run_baseline_validation", True):
        logger.info("Running baseline validation at iter 0 (before training)...")
        trainer.validate(model=lightning_module, dataloaders=val_dataloader)
        logger.info("Baseline validation complete (metrics logged at global_step=0).")

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
