#!/usr/bin/env python3
"""
Summarize PPS probe shards: reasoning gain, harmful-CoT rate, per-segment blame,
and the most-blamed segment texts (for qualitative figures).

Usage:
    python scripts/pps/summarize_pps_eval.py <output_dir>/pps_eval [--json out.json] [--top 10]
"""

import argparse
import json
from pathlib import Path

import torch

from navsim.agents.negdrive.utils.pps import blame_statistics, segment_deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    shards = sorted(args.eval_dir.glob("probes_rank*.pt"))
    if not shards:
        raise FileNotFoundError(f"No probes_rank*.pt in {args.eval_dir}")

    records, meta = [], None
    for path in shards:
        data = torch.load(path, map_location="cpu")
        meta = meta or {k: data[k] for k in ("G", "K", "P", "split", "vlm_lora_path")}
        records.extend(data["records"])

    phis = [r["phi"] for r in records]
    fail_mask = [(r["nc"] < 1).any().item() or (r["dac"] < 1).any().item() for r in records]
    summary = {
        **meta,
        "num_rollouts": len(records),
        "all": blame_statistics(phis),
        "failed_rollouts": blame_statistics([p for p, f in zip(phis, fail_mask) if f]),
        "pdm_score": torch.stack([r["score"].float().mean() for r in records]).mean().item(),
        "ego_progress": torch.stack([r["ego_progress"].float().mean() for r in records]).mean().item(),
    }

    blamed = []
    for r in records:
        d = segment_deltas(r["phi"])
        for j, delta in enumerate(d.tolist()):
            if delta < 0:
                blamed.append((delta, r["token"], r["g"], j, r["segments"][j]))
    blamed.sort(key=lambda x: x[0])
    summary["most_blamed"] = [
        {"delta": d, "token": t, "g": g, "segment_idx": j, "text": txt}
        for d, t, g, j, txt in blamed[: args.top]
    ]

    print(f"rollouts={summary['num_rollouts']}  G={meta['G']}  K={meta['K']}  P={meta['P']}  "
          f"split={meta['split']}  lora={meta['vlm_lora_path']}")
    for name in ("all", "failed_rollouts"):
        print(f"[{name}]", {k: round(v, 4) for k, v in summary[name].items()})
    print(f"pdm_score={summary['pdm_score']:.4f}  ego_progress={summary['ego_progress']:.4f}")
    print(f"\nTop {args.top} blamed segments:")
    for item in summary["most_blamed"]:
        text = item["text"].replace("\n", " ")[:160]
        print(f"  d={item['delta']:+.3f}  {item['token']} g={item['g']} seg={item['segment_idx']}: {text}")

    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
