#!/usr/bin/env python3
"""
Filter high-risk scenes from NAVSIM PDM evaluation CSV results.

A scene is high-risk if either no_at_fault_collisions or drivable_area_compliance
is below the perfect-score threshold (default 1.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_INPUT_CSV = (
    "/root/navsim_workspace/exps/0612_yrs_navtrain/"
    "2026.06.12.17.33.41/2026.06.12.22.51.17.csv"
)
DEFAULT_OUTPUT_DIR = "/root/recogdrive_yrs/scripts/filter_high_risk_scenes/high_risk_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter high-risk navtrain scenes from PDM evaluation CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(DEFAULT_INPUT_CSV),
        help="Path to PDM evaluation CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory to write filtered outputs.",
    )
    parser.add_argument(
        "--collision_threshold",
        type=float,
        default=1.0,
        help="Perfect score threshold for no_at_fault_collisions.",
    )
    parser.add_argument(
        "--drivable_threshold",
        type=float,
        default=1.0,
        help="Perfect score threshold for drivable_area_compliance.",
    )
    parser.add_argument(
        "--include_invalid",
        action="store_true",
        help="Treat valid=False rows as high-risk candidates instead of excluding them.",
    )
    return parser.parse_args()


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map({"True": True, "False": False, True: True, False: False})


def assign_risk_type(row: pd.Series) -> str:
    collision_fail = row["no_at_fault_collisions"] < 1.0
    drivable_fail = row["drivable_area_compliance"] < 1.0

    if collision_fail and drivable_fail:
        return "both"
    if collision_fail:
        return "collision_only"
    if drivable_fail:
        return "drivable_only"
    return "none"


def load_and_clean(
    input_csv: Path,
    include_invalid: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(input_csv)

    if "token" not in df.columns:
        raise ValueError(f"Missing 'token' column in {input_csv}")

    df = df[df["token"] != "average"].copy()

    duplicate_tokens = df["token"][df["token"].duplicated()].unique().tolist()
    if duplicate_tokens:
        print(
            f"Warning: found {len(duplicate_tokens)} duplicated token(s); "
            "keeping the first occurrence for each token."
        )
        df = df.drop_duplicates(subset="token", keep="first")

    if "valid" in df.columns:
        df["valid"] = _to_bool(df["valid"])
        invalid_df = df[df["valid"] == False].copy()  # noqa: E712
        valid_df = df[df["valid"] == True].copy()  # noqa: E712
    else:
        invalid_df = pd.DataFrame(columns=df.columns)
        valid_df = df.copy()

    if include_invalid:
        candidate_df = df.copy()
    else:
        candidate_df = valid_df.copy()

    return df, candidate_df, invalid_df, duplicate_tokens


def filter_high_risk(
    candidate_df: pd.DataFrame,
    collision_threshold: float,
    drivable_threshold: float,
) -> pd.DataFrame:
    collision_fail = candidate_df["no_at_fault_collisions"] < collision_threshold
    drivable_fail = candidate_df["drivable_area_compliance"] < drivable_threshold

    high_risk = candidate_df[collision_fail | drivable_fail].copy()
    high_risk["risk_type"] = high_risk.apply(assign_risk_type, axis=1)
    return high_risk.sort_values("token").reset_index(drop=True)


def build_summary(
    input_csv: Path,
    total_rows: int,
    candidate_count: int,
    high_risk: pd.DataFrame,
    invalid_df: pd.DataFrame,
    duplicate_tokens: list[str],
    collision_threshold: float,
    drivable_threshold: float,
    include_invalid: bool,
) -> dict[str, Any]:
    collision_only = int((high_risk["risk_type"] == "collision_only").sum())
    drivable_only = int((high_risk["risk_type"] == "drivable_only").sum())
    both = int((high_risk["risk_type"] == "both").sum())

    collision_0 = int((high_risk["no_at_fault_collisions"] == 0.0).sum())
    collision_05 = int((high_risk["no_at_fault_collisions"] == 0.5).sum())

    high_risk_count = len(high_risk)
    ratio = high_risk_count / candidate_count if candidate_count else 0.0

    return {
        "input_csv": str(input_csv),
        "collision_threshold": collision_threshold,
        "drivable_threshold": drivable_threshold,
        "include_invalid": include_invalid,
        "total_rows_excluding_average": total_rows,
        "candidate_scenes": candidate_count,
        "high_risk_count": high_risk_count,
        "high_risk_ratio": ratio,
        "breakdown": {
            "collision_only": collision_only,
            "drivable_only": drivable_only,
            "both": both,
            "collision_score_0.0": collision_0,
            "collision_score_0.5": collision_05,
        },
        "invalid_excluded": int(len(invalid_df)),
        "duplicate_tokens_found": len(duplicate_tokens),
    }


def write_outputs(
    output_dir: Path,
    high_risk: pd.DataFrame,
    invalid_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = output_dir / "high_risk_tokens.txt"
    with tokens_path.open("w", encoding="utf-8") as f:
        for token in high_risk["token"]:
            f.write(f"{token}\n")

    high_risk.to_csv(output_dir / "high_risk_scenes.csv", index=False)

    if len(invalid_df) > 0:
        invalid_df.to_csv(output_dir / "invalid_scenes.csv", index=False)

    with (output_dir / "high_risk_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    scene_filter = {
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "num_history_frames": 4,
        "num_future_frames": 10,
        "frame_interval": 1,
        "has_route": True,
        "max_scenes": None,
        "log_names": None,
        "tokens": high_risk["token"].tolist(),
    }
    with (output_dir / "scene_filter_high_risk.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(scene_filter, f, sort_keys=False, allow_unicode=True)


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    breakdown = summary["breakdown"]
    print(f"Input CSV:                {summary['input_csv']}")
    print(f"Candidate scenes:         {summary['candidate_scenes']:,}")
    print(
        f"High-risk scenes:         {summary['high_risk_count']:,} "
        f"({summary['high_risk_ratio'] * 100:.2f}%)"
    )
    print(f"  - collision only:       {breakdown['collision_only']:,}")
    print(f"  - drivable only:        {breakdown['drivable_only']:,}")
    print(f"  - both:                 {breakdown['both']:,}")
    print(f"  - collision=0.0:        {breakdown['collision_score_0.0']:,}")
    print(f"  - collision=0.5:        {breakdown['collision_score_0.5']:,}")
    print(f"Invalid scenes excluded:  {summary['invalid_excluded']:,}")
    if summary["duplicate_tokens_found"]:
        print(f"Duplicate tokens found:   {summary['duplicate_tokens_found']:,}")
    print(f"\nOutputs written to:       {output_dir}")


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    raw_df, candidate_df, invalid_df, duplicate_tokens = load_and_clean(
        args.input_csv,
        include_invalid=args.include_invalid,
    )

    high_risk = filter_high_risk(
        candidate_df,
        collision_threshold=args.collision_threshold,
        drivable_threshold=args.drivable_threshold,
    )

    summary = build_summary(
        input_csv=args.input_csv,
        total_rows=len(raw_df),
        candidate_count=len(candidate_df),
        high_risk=high_risk,
        invalid_df=invalid_df,
        duplicate_tokens=duplicate_tokens,
        collision_threshold=args.collision_threshold,
        drivable_threshold=args.drivable_threshold,
        include_invalid=args.include_invalid,
    )

    write_outputs(args.output_dir, high_risk, invalid_df, summary)
    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
