#!/usr/bin/env python3
"""
Merge per-rank MARS diagnostic shards and print the variance decomposition.

Usage:
    python scripts/mars/summarize_mars_diagnostic.py <output_dir>/mars_diagnostic [--json out.json]

Reports, per signal (failure indicator, PDM score, ego progress):
    var_noise      E_c[Var_eps]   variance due to diffusion-planner noise
    var_reasoning  Var_c[E_eps]   variance due to the sampled CoT
    noise_fraction var_noise / (var_noise + var_reasoning)
and flip statistics of the failure indicator (fraction of CoTs whose safety
depends on planner noise).
"""

import argparse
import json
from pathlib import Path

import torch

from navsim.agents.negdrive.utils.mars import compute_failure, decompose_variance, flip_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diag_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    shards = sorted(args.diag_dir.glob("metrics_rank*.pt"))
    if not shards:
        raise FileNotFoundError(f"No metrics_rank*.pt in {args.diag_dir}")

    tokens, metrics, meta = [], {}, None
    for path in shards:
        data = torch.load(path, map_location="cpu")
        meta = meta or {k: data[k] for k in ("G", "K", "common_random_numbers", "do_sample", "split")}
        tokens.extend(data["tokens"])
        for key, value in data["metrics"].items():
            metrics.setdefault(key, []).append(value)
    metrics = {k: torch.cat(v, dim=0) for k, v in metrics.items()}

    fail = compute_failure(metrics["no_at_fault_collisions"], metrics["drivable_area_compliance"])
    summary = {
        **meta,
        "num_scenes": len(tokens),
        "flip": flip_statistics(fail),
        "fail": decompose_variance(fail),
        "score": decompose_variance(metrics["score"]),
        "ego_progress": decompose_variance(metrics["ego_progress"]),
        "means": {k: v.mean().item() for k, v in metrics.items()},
    }

    print(f"scenes={summary['num_scenes']}  G={meta['G']}  K={meta['K']}  "
          f"CRN={meta['common_random_numbers']}  split={meta['split']}")
    print("flip stats:", {k: round(v, 4) for k, v in summary["flip"].items()})
    for name in ("fail", "score", "ego_progress"):
        s = summary[name]
        print(f"{name:>13}: var_noise={s['var_noise']:.4f}  var_reasoning={s['var_reasoning']:.4f}  "
              f"noise_fraction={s['noise_fraction']:.3f}")
    print("means:", {k: round(v, 4) for k, v in summary["means"].items()})

    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
