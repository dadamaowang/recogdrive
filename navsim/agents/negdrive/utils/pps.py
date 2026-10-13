# -*- encoding: utf-8 -*-
'''
@File    :   pps.py
@Desc    :   PPS (Planner-Potential Shaping) utilities: step-level negative credit for CoT.

The VLM is causal, so the hidden states of a CoT prefix c_{<=t} are the first
prompt_len + t positions of the full forward pass. Decoding them with the frozen
planner (under the same K planner noises) and scoring with PDM gives a potential

    Phi(c_{<=t}) = -E_eps[ cost(planner(h(c_{<=t}), eps)) ]

Phi(prompt only) is the no-reasoning plan; Phi(full CoT) is the outcome that the
rollout is rewarded with. Splitting the CoT into segments at boundaries
0 = b_0 < b_1 < ... < b_P = L, segment j gets the potential difference

    d_j = Phi(c_{<b_{j+1}}) - Phi(c_{<b_j}),      sum_j d_j = Phi(full) - Phi(prompt)

i.e. a one-step TD advantage with the frozen planner as a free critic. With
negative_only, only segments that increased the risk receive (negative) credit.

Pure functions only (no model / simulator), so they can be unit-tested on CPU.
'''

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


DEFAULT_STAGE_MARKERS = (
    "Dynamic Understanding",
    "Planning & Reasoning",
    "Advanced Reasoning",
    "[PT",
)


@dataclass
class PPSConfig:
    max_segments: int = 4            # P: number of CoT segments probed per rollout
    shaping_coef: float = 1.0        # lambda: weight of the segment potential differences
    negative_only: bool = True       # keep only d_j < 0 (NSR-style: blame, never praise)
    probe: str = "failures"          # failures: probe rollouts with any planner failure | all
    use_stage_markers: bool = True   # prefer the prompt's stage headings as boundaries
    advantage_eps: float = 1e-6

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "PPSConfig":
        cfg = dict(cfg or {})
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        out = cls(**known)
        if out.max_segments < 1:
            raise ValueError("pps_cfg.max_segments must be >= 1")
        if out.probe not in ("failures", "all"):
            raise ValueError(f"Unknown pps_cfg.probe: {out.probe!r}")
        return out


def find_stop_index(response_ids: Sequence[int], stop_ids: Sequence[int]) -> int:
    """Number of real response tokens: index of the first stop token + 1, else len."""
    stop = set(stop_ids)
    for i, tok in enumerate(response_ids):
        if tok in stop:
            return i + 1
    return len(response_ids)


def _char_to_token(pieces: Sequence[str]) -> List[int]:
    """Start character offset of every token in the concatenated text."""
    starts, pos = [], 0
    for piece in pieces:
        starts.append(pos)
        pos += len(piece)
    return starts


def segment_response(
    pieces: Sequence[str],
    num_real: int,
    num_total: int,
    max_segments: int,
    markers: Sequence[str] = DEFAULT_STAGE_MARKERS,
    use_markers: bool = True,
) -> List[int]:
    """
    Choose prefix boundaries (in response-token indices) for a generated CoT.

    Args:
        pieces:       decoded text of each response token.
        num_real:     tokens up to and including the stop token.
        num_total:    all generated tokens (incl. padding after stop); the last
                      boundary, so the final potential equals the rollout outcome.
        max_segments: at most this many segments.

    Returns sorted boundaries [0, b_1, ..., num_total] with 0 < b_j < num_real.
    A boundary b means the prefix contains response tokens [0, b).
    Stage markers (the prompt's headings, "[PT" for the trajectory text) are
    used first; remaining slots are filled with sentence/line ends spaced as
    evenly as possible.
    """
    if num_real <= 1 or max_segments <= 1:
        return [0, num_total]

    pieces = list(pieces[:num_real])
    starts = _char_to_token(pieces)
    text = "".join(pieces)

    def token_at(char_pos: int) -> int:
        tok = 0
        for i, s in enumerate(starts):
            if s <= char_pos:
                tok = i
            else:
                break
        return tok

    chosen = set()
    if use_markers:
        for marker in markers:
            pos = text.find(marker)
            if pos > 0:
                b = token_at(pos)
                if 0 < b < num_real:
                    chosen.add(b)

    candidates = [
        i + 1
        for i, piece in enumerate(pieces[:-1])
        if "\n" in piece or piece.rstrip().endswith((".", ":", ";"))
    ]
    candidates = [c for c in candidates if 0 < c < num_real and c not in chosen]

    budget = max_segments - 1
    if len(chosen) > budget:
        # keep the markers closest to an even split
        targets = [num_real * (j + 1) / max_segments for j in range(budget)]
        ordered = sorted(chosen)
        picked = []
        for t in targets:
            best = min((b for b in ordered if b not in picked), key=lambda b: abs(b - t))
            picked.append(best)
        chosen = set(picked)

    while len(chosen) < budget and candidates:
        # fill the largest gap with the candidate closest to its middle
        edges = sorted(chosen | {0, num_real})
        gaps = sorted(zip(edges[:-1], edges[1:]), key=lambda e: e[1] - e[0], reverse=True)
        placed = False
        for lo, hi in gaps:
            inside = [c for c in candidates if lo < c < hi]
            if inside:
                mid = (lo + hi) / 2
                best = min(inside, key=lambda c: abs(c - mid))
                chosen.add(best)
                candidates.remove(best)
                placed = True
                break
        if not placed:
            break

    return [0] + sorted(chosen) + [num_total]


def segment_deltas(phi: torch.Tensor) -> torch.Tensor:
    """phi: [P+1] potentials at the boundaries -> d: [P] potential differences."""
    return phi[1:] - phi[:-1]


def build_token_advantages(
    boundaries: Sequence[int],
    phi: torch.Tensor,
    num_real: int,
    num_total: int,
    outcome_advantage: float,
    shaping_coef: float,
    negative_only: bool,
) -> torch.Tensor:
    """
    Per-token advantages [num_total] for one rollout:
        A_t = outcome_advantage + lambda * d_{seg(t)}     for t < num_real
        A_t = 0                                         for padding after stop
    """
    deltas = segment_deltas(phi.float())
    if negative_only:
        deltas = deltas.clamp(max=0.0)
    adv = torch.zeros(num_total, dtype=torch.float32)
    for j in range(len(boundaries) - 1):
        lo, hi = boundaries[j], min(boundaries[j + 1], num_real)
        if hi > lo:
            adv[lo:hi] = outcome_advantage + shaping_coef * deltas[j].item()
    return adv


def blame_statistics(phis: List[torch.Tensor]) -> Dict[str, float]:
    """
    Aggregate over probed rollouts (each phi: [P+1]).
        reasoning_gain:   mean Phi(full) - Phi(prompt)   (> 0: CoT made plans safer)
        harmful_cot_rate: fraction of rollouts with Phi(full) < Phi(prompt)
        neg_segment_rate: fraction of segments with d_j < 0
        blame_seg{j}:     mean min(d_j, 0) at segment position j (positions beyond
                          a rollout's segment count are skipped)
    """
    if not phis:
        return {}
    gains = torch.stack([p[-1] - p[0] for p in phis]).float()
    deltas = [segment_deltas(p.float()) for p in phis]
    all_d = torch.cat(deltas)
    stats = {
        "phi_prompt": torch.stack([p[0] for p in phis]).float().mean().item(),
        "phi_full": torch.stack([p[-1] for p in phis]).float().mean().item(),
        "reasoning_gain": gains.mean().item(),
        "harmful_cot_rate": (gains < 0).float().mean().item(),
        "neg_segment_rate": (all_d < 0).float().mean().item(),
    }
    max_p = max(len(d) for d in deltas)
    for j in range(max_p):
        vals = [d[j].clamp(max=0.0) for d in deltas if len(d) > j]
        stats[f"blame_seg{j}"] = torch.stack(vals).mean().item()
    return stats
