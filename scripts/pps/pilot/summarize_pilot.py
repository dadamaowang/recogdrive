#!/usr/bin/env python3
"""
PPS pilot, phase A summary: go / no-go table for the four signals PPS needs.

  S1 steer       the CoT changes the plan:            P(Phi_full != Phi_prompt)
  S2 localize    failures have a blamable segment,    P(min_j d_j < 0 | failed)
                 and blame is concentrated:            share of negative mass on the worst segment
  S3 reliable    Phi is stable under fresh noises:     Spearman(Phi_full, Phi_ref); blame vs noise floor
  S4 causal      removing the blamed segment helps     E[Phi_drop_blame - Phi_full] - E[Phi_drop_random - Phi_full]
                 more than removing a random one:      (paired bootstrap 95% CI must exclude 0)

Usage:
    python scripts/pps/pilot/summarize_pilot.py <output_dir>/pps_pilot [--json out.json]
"""

import argparse
import json
from pathlib import Path

import torch

from navsim.agents.negdrive.utils.pps import segment_deltas

THRESHOLDS = {
    "S1_steer_rate": 0.20,
    "S2_localizable_rate": 0.30,
    "S2_concentration": 0.60,
    "S3_spearman": 0.50,
    "S3_signal_to_noise": 1.50,
    "S4_causal_gap_ci_low": 0.0,
}


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    def rank(v):
        order = v.argsort()
        r = torch.empty_like(v)
        r[order] = torch.arange(len(v), dtype=v.dtype)
        # average ties
        for val in v.unique():
            idx = (v == val).nonzero().flatten()
            r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x.double()), rank(y.double())
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = (rx.norm() * ry.norm()).item()
    return (rx @ ry).item() / denom if denom > 0 else 0.0


def bootstrap_ci(x: torch.Tensor, n: int = 2000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x), (n, len(x)), generator=g)
    means = x[idx].mean(dim=1)
    return x.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    shards = sorted(args.pilot_dir.glob("pilot_rank*.pt"))
    if not shards:
        raise FileNotFoundError(f"No pilot_rank*.pt in {args.pilot_dir}")
    records, meta = [], None
    for p in shards:
        d = torch.load(p, map_location="cpu")
        meta = meta or {k: d[k] for k in ("G", "K", "P", "split")}
        records.extend(d["records"])
    if not records:
        raise RuntimeError("No rollouts recorded")

    phi = [r["phi"].float() for r in records]
    phi_prompt = torch.stack([p[0] for p in phi])
    phi_full = torch.stack([p[-1] for p in phi])
    failed = phi_full < 0
    out = {**meta, "num_rollouts": len(records), "num_scenes": len({r["token"] for r in records}),
           "num_failed": int(failed.sum())}

    # S1 steer
    out["S1_steer_rate"] = (phi_full != phi_prompt).float().mean().item()
    out["reasoning_gain"] = (phi_full - phi_prompt).mean().item()
    out["harmful_cot_rate"] = (phi_full < phi_prompt).float().mean().item()
    out["prompt_fail_rate"] = (phi_prompt < 0).float().mean().item()
    out["failed_but_prompt_safe_rate"] = (failed & (phi_prompt == 0)).float().sum().item() / max(int(failed.sum()), 1)

    # S2 localize
    deltas = [segment_deltas(p) for p in phi]
    fail_deltas = [d for d, f in zip(deltas, failed.tolist()) if f]
    if fail_deltas:
        has_neg = torch.tensor([float((d < 0).any()) for d in fail_deltas])
        out["S2_localizable_rate"] = has_neg.mean().item()
        conc = [(d.clamp(max=0).min() / d.clamp(max=0).sum()).item() for d in fail_deltas if (d < 0).any()]
        out["S2_concentration"] = sum(conc) / len(conc) if conc else 0.0
        P = max(len(d) for d in fail_deltas)
        out["blame_by_segment"] = [
            sum(d[j].clamp(max=0).item() for d in fail_deltas if len(d) > j) / len(fail_deltas) for j in range(P)
        ]
        out["argmax_blame_hist"] = [
            sum(1 for d in fail_deltas if (d < 0).any() and int(d.argmin()) == j) for j in range(P)
        ]
    else:
        out["S2_localizable_rate"] = out["S2_concentration"] = 0.0

    # S3 reliability
    ref = [(r, -r["cost_ref"].float().mean()) for r in records if "cost_ref" in r]
    if ref:
        pf = torch.stack([r["phi"][-1].float() for r, _ in ref])
        pr = torch.stack([v for _, v in ref])
        out["S3_spearman"] = spearman(pf, pr)
        noise_floor = (pf - pr).abs().mean().item()
        blame_mag = [(-d.min()).item() for d in fail_deltas if (d < 0).any()]
        out["S3_noise_floor"] = noise_floor
        out["S3_blame_magnitude"] = sum(blame_mag) / len(blame_mag) if blame_mag else 0.0
        out["S3_signal_to_noise"] = out["S3_blame_magnitude"] / max(noise_floor, 1e-6)

    # S4 causal
    paired = [r for r in records if "cost_drop_blame" in r]
    out["S4_num_pairs"] = len(paired)
    if paired:
        full = torch.stack([r["phi"][-1].float() for r in paired])
        gain_blame = torch.stack([-r["cost_drop_blame"].float().mean() for r in paired]) - full
        gain_rand = torch.stack([-r["cost_drop_random"].float().mean() for r in paired]) - full
        out["S4_gain_drop_blame"] = gain_blame.mean().item()
        out["S4_gain_drop_random"] = gain_rand.mean().item()
        m, lo, hi = bootstrap_ci(gain_blame - gain_rand)
        out["S4_causal_gap"], out["S4_causal_gap_ci_low"], out["S4_causal_gap_ci_high"] = m, lo, hi
        out["S4_fix_rate"] = torch.stack([(r["cost_drop_blame"] == 0).all() for r in paired]).float().mean().item()

    checks = {
        "S1 CoT steers the planner": out["S1_steer_rate"] >= THRESHOLDS["S1_steer_rate"],
        "S2 failures are localizable": out["S2_localizable_rate"] >= THRESHOLDS["S2_localizable_rate"],
        "S2 blame is concentrated": out["S2_concentration"] >= THRESHOLDS["S2_concentration"],
        "S3 Phi is reliable": out.get("S3_spearman", 0) >= THRESHOLDS["S3_spearman"],
        "S3 blame above noise floor": out.get("S3_signal_to_noise", 0) >= THRESHOLDS["S3_signal_to_noise"],
        "S4 blame is causal": out.get("S4_causal_gap_ci_low", -1) > THRESHOLDS["S4_causal_gap_ci_low"],
    }
    out["checks"] = checks
    hard = [checks["S1 CoT steers the planner"], checks["S2 failures are localizable"], checks["S4 blame is causal"]]
    out["verdict"] = ("GO" if all(checks.values()) else
                      "GO (weak: tune K/segmentation)" if all(hard) else "NO-GO")

    print(f"scenes={out['num_scenes']} rollouts={out['num_rollouts']} failed={out['num_failed']} "
          f"G={meta['G']} K={meta['K']} P={meta['P']} split={meta['split']}")
    keys = ["S1_steer_rate", "reasoning_gain", "harmful_cot_rate", "prompt_fail_rate",
            "failed_but_prompt_safe_rate", "S2_localizable_rate", "S2_concentration",
            "S3_spearman", "S3_noise_floor", "S3_blame_magnitude", "S3_signal_to_noise",
            "S4_num_pairs", "S4_gain_drop_blame", "S4_gain_drop_random", "S4_causal_gap",
            "S4_causal_gap_ci_low", "S4_causal_gap_ci_high", "S4_fix_rate"]
    for k in keys:
        if k in out:
            thr = THRESHOLDS.get(k)
            print(f"  {k:<24} {out[k]:>8.4f}" + (f"   (threshold {thr})" if thr is not None else ""))
    if "blame_by_segment" in out:
        print("  blame_by_segment        ", [round(v, 4) for v in out["blame_by_segment"]])
        print("  argmax_blame_hist       ", out["argmax_blame_hist"])
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nVERDICT: {out['verdict']}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
