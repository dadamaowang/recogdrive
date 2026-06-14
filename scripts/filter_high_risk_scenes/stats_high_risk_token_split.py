#!/usr/bin/env python3
"""
统计高风险 token 在 train_logs / val_logs 中的分布。

与 run_training_negdrive.py 中 high_risk 训练逻辑对齐：
- 数据路径: OPENSCENE_DATA_ROOT/navsim_logs/trainval
- SceneFilter: navtrain 默认 (4 history, 10 future, has_route=True)
- train 侧: log_names = train_logs, tokens = 高风险白名单
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Set

from omegaconf import OmegaConf

RECOGDRIVE_ROOT = Path(__file__).resolve().parents[2]
if str(RECOGDRIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(RECOGDRIVE_ROOT))

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import filter_scenes


def load_token_list(path: Path) -> list[str]:
    tokens: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                tokens.append(token)
    if not tokens:
        raise ValueError(f"No tokens in {path}")
    return tokens


def tokens_in_logs(
    data_path: Path,
    log_names: list[str],
    token_whitelist: Set[str],
    scene_filter_kwargs: dict,
) -> Set[str]:
    """在指定 log 列表中，能按 SceneFilter 规则加载到的 token 集合。"""
    scene_filter = SceneFilter(
        log_names=log_names,
        tokens=list(token_whitelist),
        **scene_filter_kwargs,
    )
    scenes = filter_scenes(data_path, scene_filter)
    return set(scenes.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stats high-risk tokens vs train/val logs.")
    parser.add_argument(
        "--tokens_path",
        type=Path,
        default=Path("scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt"),
        help="高风险 token 列表（一行一个 token）",
    )
    parser.add_argument(
        "--log_split_yaml",
        type=Path,
        default=Path("navsim/planning/script/config/training/default_train_val_test_log_split.yaml"),
        help="train_logs / val_logs 划分文件",
    )
    parser.add_argument(
        "--navsim_log_path",
        type=Path,
        default=None,
        help="navsim_logs 目录，默认 OPENSCENE_DATA_ROOT/navsim_logs/trainval",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="可选：将统计结果写入 JSON",
    )
    args = parser.parse_args()

    tokens_path = args.tokens_path
    if not tokens_path.is_absolute():
        tokens_path = RECOGDRIVE_ROOT / tokens_path

    log_split_yaml = args.log_split_yaml
    if not log_split_yaml.is_absolute():
        log_split_yaml = RECOGDRIVE_ROOT / log_split_yaml

    if args.navsim_log_path is not None:
        data_path = args.navsim_log_path
    else:
        open_scene = os.environ.get("OPENSCENE_DATA_ROOT", "/root/navsim_workspace/dataset")
        data_path = Path(open_scene) / "navsim_logs" / "trainval"

    if not tokens_path.exists():
        raise FileNotFoundError(f"tokens_path not found: {tokens_path}")
    if not log_split_yaml.exists():
        raise FileNotFoundError(f"log_split_yaml not found: {log_split_yaml}")
    if not data_path.exists():
        raise FileNotFoundError(f"navsim_log_path not found: {data_path}")

    print("=" * 60)
    print("High-risk token split statistics")
    print("=" * 60)
    print(f"tokens_path:     {tokens_path}")
    print(f"log_split_yaml:  {log_split_yaml}")
    print(f"navsim_log_path: {data_path}")
    print()

    high_risk_tokens = load_token_list(tokens_path)
    high_risk_set = set(high_risk_tokens)
    print(f"High-risk tokens loaded: {len(high_risk_set)}")

    split_cfg = OmegaConf.load(str(log_split_yaml))
    train_logs = list(split_cfg.train_logs)
    val_logs = list(split_cfg.val_logs)
    train_log_set = set(train_logs)
    val_log_set = set(val_logs)
    overlap_logs = train_log_set & val_log_set
    print(f"train_logs: {len(train_logs)}, val_logs: {len(val_logs)}")
    if overlap_logs:
        print(f"WARNING: train/val log overlap: {len(overlap_logs)} logs")
    print()

    scene_filter_kwargs = {
        "num_history_frames": 4,
        "num_future_frames": 10,
        "frame_interval": 1,
        "has_route": True,
        "max_scenes": None,
    }

    t0 = time.time()
    print("Scanning train_logs (may take a few minutes)...")
    train_found = tokens_in_logs(data_path, train_logs, high_risk_set, scene_filter_kwargs)
    t1 = time.time()
    print(f"  Found in train_logs: {len(train_found)}  ({t1 - t0:.1f}s)")

    missing_from_train = high_risk_set - train_found
    print("Scanning val_logs for tokens missing from train...")
    val_found_missing = tokens_in_logs(
        data_path, val_logs, missing_from_train, scene_filter_kwargs
    )
    t2 = time.time()
    print(f"  Found in val_logs:   {len(val_found_missing)}  ({t2 - t1:.1f}s)")
    print()

    in_val_only = missing_from_train & val_found_missing
    not_found_anywhere = missing_from_train - val_found_missing
    in_both = train_found & val_found_missing

    print("=" * 60)
    print("Summary (aligned with training warning)")
    print("=" * 60)
    print(f"Total high-risk tokens:              {len(high_risk_set)}")
    print(f"Found in train_logs (可参与训练):    {len(train_found)}")
    print(f"Missing from train_logs:             {len(missing_from_train)}")
    print()
    print("Breakdown of missing-from-train:")
    print(f"  -> in val_logs only:                {len(in_val_only)}")
    print(f"  -> not found in train nor val:      {len(not_found_anywhere)}")
    if in_both:
        print(f"  (also in both train & val:         {len(in_both)} tokens)")
    print()
    print(f"Found in val_logs (among missing):   {len(val_found_missing)}")
    print(f"Elapsed: {t2 - t0:.1f}s")
    print("=" * 60)

    result = {
        "tokens_path": str(tokens_path),
        "navsim_log_path": str(data_path),
        "total_high_risk": len(high_risk_set),
        "found_in_train_logs": len(train_found),
        "missing_from_train": len(missing_from_train),
        "missing_from_train_in_val_logs": len(in_val_only),
        "missing_from_train_not_found_anywhere": len(not_found_anywhere),
        "found_in_val_logs_for_missing": len(val_found_missing),
        "in_both_train_and_val": len(in_both),
        "train_found_tokens": sorted(train_found),
        "val_only_tokens": sorted(in_val_only),
        "not_found_tokens": sorted(not_found_anywhere),
    }

    if args.output_json:
        out_path = args.output_json
        if not out_path.is_absolute():
            out_path = RECOGDRIVE_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"JSON written to: {out_path}")

    out_dir = tokens_path.parent
    (out_dir / "stats_in_train_tokens.txt").write_text(
        "\n".join(sorted(train_found)) + "\n", encoding="utf-8"
    )
    (out_dir / "stats_in_val_only_tokens.txt").write_text(
        "\n".join(sorted(in_val_only)) + "\n", encoding="utf-8"
    )
    (out_dir / "stats_not_found_tokens.txt").write_text(
        "\n".join(sorted(not_found_anywhere)) + "\n", encoding="utf-8"
    )
    print(f"Token lists written under: {out_dir}")


if __name__ == "__main__":
    main()
