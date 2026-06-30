# -*- encoding: utf-8 -*-
'''
@File    :   view_dataset.py
@Time    :   2026/06/25 10:28:08
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   None
@Desc    :   None
'''



# check_one_train_item.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset


from PIL import Image






CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def negdrive_collate_fn(
    batch: List[
        Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            str,
        ]
    ]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack(
        [features["history_trajectory"] for features in features_list], dim=0
    ).cpu()
    high_command_one_hot = torch.stack(
        [features["high_command_one_hot"] for features in features_list], dim=0
    ).cpu()
    status_feature = torch.stack(
        [features["status_feature"] for features in features_list], dim=0
    ).cpu()
    image_path_tensor = torch.stack(
        [features["image_path_tensor"] for features in features_list], dim=0
    ).cpu()

    trajectory = torch.stack(
        [targets["trajectory"] for targets in targets_list], dim=0
    ).cpu()

    return (
        {
            "history_trajectory": history_trajectory,
            "high_command_one_hot": high_command_one_hot,
            "status_feature": status_feature,
            "image_path_tensor": image_path_tensor,
        },
        {"trajectory": trajectory},
        list(tokens_list),
    )


def decode_image_path_tensor(path_tensor: torch.Tensor) -> str:
    if path_tensor.ndim == 2:
        path_tensor = path_tensor[0]
    chars = []
    for code in path_tensor:
        code_item = int(code.item())
        if code_item == 0:
            break
        chars.append(chr(code_item))
    return "".join(chars)

def save_image_from_path_tensor(path_tensor: torch.Tensor, save_path: Path) -> None:
    image_path = decode_image_path_tensor(path_tensor)
    image = Image.open(image_path).convert("RGB")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    print(f"Saved image to: {save_path}")



def load_token_list(tokens_path: Path) -> List[str]:
    if not tokens_path.exists():
        raise FileNotFoundError(f"High-risk tokens file not found: {tokens_path}")
    with tokens_path.open("r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    if not tokens:
        raise ValueError(f"No tokens found in {tokens_path}")
    return tokens


def apply_train_log_names(scene_filter: SceneFilter, train_logs: List[str]) -> SceneFilter:
    if scene_filter.log_names is not None:
        scene_filter.log_names = [
            log_name for log_name in scene_filter.log_names if log_name in train_logs
        ]
    else:
        scene_filter.log_names = list(train_logs)
    return scene_filter


def build_train_scene_filter(cfg: DictConfig) -> SceneFilter:
    base_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    subset = cfg.get("train_scene_subset", "full")

    if subset == "full":
        return apply_train_log_names(base_filter, cfg.train_logs)

    if subset == "high_risk":
        tokens_path = Path(cfg.high_risk.tokens_path)
        tokens = load_token_list(tokens_path)

        include_val_logs = bool(
            OmegaConf.select(cfg, "high_risk.include_val_logs", default=True)
        )
        if include_val_logs:
            base_filter.log_names = list(cfg.train_logs) + list(cfg.val_logs)
        else:
            base_filter.log_names = list(cfg.train_logs)
        base_filter.tokens = tokens
        return base_filter

    raise ValueError(
        f"Unknown train_scene_subset: {subset!r}. Expected 'full' or 'high_risk'."
    )


def resolve_validation_paths(cfg: DictConfig) -> Tuple[Path, Path, Optional[str]]:
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
        val_scene_filter: SceneFilter = instantiate(cfg.validation.scene_filter)
        val_scene_filter.log_names = None
        val_scene_filter.tokens = tokens
        return val_scene_filter

    raise ValueError(
        f"Unknown val_scene_subset: {subset!r}. Expected 'full' or 'high_risk_test'."
    )


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter = build_train_scene_filter(cfg)
    val_scene_filter = build_val_scene_filter(cfg)

    train_data_path = Path(cfg.navsim_log_path)
    train_sensor_blobs_path = Path(cfg.sensor_blobs_path)
    val_data_path, val_sensor_blobs_path, _ = resolve_validation_paths(cfg)

    train_scene_loader = SceneLoader(
        data_path=train_data_path,
        sensor_blobs_path=train_sensor_blobs_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    val_scene_loader = SceneLoader(
        data_path=val_data_path,
        sensor_blobs_path=val_sensor_blobs_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=False,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=False,
    )

    return train_data, val_data


def decode_image_path_tensor(path_tensor: torch.Tensor) -> str:
    if path_tensor.ndim == 2:
        path_tensor = path_tensor[0]
    return "".join(chr(int(x)) for x in path_tensor.tolist())


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    agent: AbstractAgent = instantiate(cfg.agent)

    train_data, _ = build_datasets(cfg, agent)

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        num_workers=0,
        collate_fn=negdrive_collate_fn,
        shuffle=False,
    )

    batch = next(iter(train_loader))
    features, targets, tokens = batch

    print("token:", tokens[0])
    print("history_trajectory shape:", features["history_trajectory"].shape)
    print("high_command_one_hot shape:", features["high_command_one_hot"].shape)
    print("status_feature shape:", features["status_feature"].shape)
    print("image_path_tensor shape:", features["image_path_tensor"].shape)
    print("decoded image path:", decode_image_path_tensor(features["image_path_tensor"]))
    print("trajectory shape:", targets["trajectory"].shape)
    print("trajectory:", targets["trajectory"][0])

    # You can now visualize:
    # history = features["history_trajectory"][0].numpy()
    # future = targets["trajectory"][0].numpy()
    # image_path = decode_image_path_tensor(features["image_path_tensor"])
    # ...
    image_path = decode_image_path_tensor(features["image_path_tensor"])
    print("decoded image path:", image_path)

    save_image_from_path_tensor(
        features["image_path_tensor"],
        Path(cfg.output_dir) / "sample_image.png",
    )


if __name__ == "__main__":
    main()
