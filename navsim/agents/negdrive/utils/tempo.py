# -*- encoding: utf-8 -*-
'''
@File    :   tempo.py
@Desc    :   TEMPO advantage over a segment-level prefix tree (arXiv 2509.18314, adapted).

GRPO's group-relative outcome term plus a TD correction from a prefix tree built
over the semantic CoT sections of the G responses sampled for the same prompt:

    A_{i,t} = ( (r_i - mean(r)) + td_coef * (V(node_j) - V(node_{j-1})) ) / (std(r) + eps)

for token t in section j of response i. Nodes are section prefixes (the paper uses
tokens); V(node) is the mean raw reward of the responses passing through it.
No critic, no PRM, no extra rollouts. CPU only, no autograd.

Pure module (torch + numpy); no model / simulator imports, so it is unit-tested on CPU
(tests/test_tempo.py).
'''

import importlib
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Structured CoT prompt (VLM side)
# ---------------------------------------------------------------------------
DEFAULT_SEGMENT_TAGS = ["<perception>", "<prediction>", "<decision>", "<check>", "<trajectory>"]

META_ACTIONS = (
    "KEEP LANE, ACCELERATE, DECELERATE, STOP, TURN LEFT, TURN RIGHT, "
    "CHANGE LANE LEFT, CHANGE LANE RIGHT"
)

SECTION_INSTRUCTIONS = {
    "<perception>": "critical static and dynamic elements visible (traffic lights, lanes, vehicles, pedestrians, obstacles).",
    "<prediction>": "movement and likely intent of the agents that matter for your path.",
    "<decision>": f"start with exactly one meta-action from [{META_ACTIONS}], then give the causal reason in one sentence.",
    "<check>": "one counterfactual safety check (e.g. 'If the lead car brakes, I will ...').",
    "<trajectory>": "8 future waypoints as [PT, (x, y, heading), ...] with 2 decimal places.",
}


def closing_tag(tag: str) -> Optional[str]:
    """'<name>' -> '</name>'; None for tags that are not of that form."""
    m = re.fullmatch(r"<([^/<>][^<>]*)>", tag)
    return f"</{m.group(1)}>" if m else None


def tag_name(tag: str) -> str:
    return tag.strip("<>/ ") or tag


def build_structured_cot_instructions(tags: Sequence[str]) -> str:
    """Instruction block asking for one tagged section per tag, in order."""
    lines = [
        f"Write your answer as exactly {len(tags)} sections, in this order, each wrapped in its tags. "
        f"Output nothing before {tags[0]} and nothing after the last closing tag. "
        "Use each tag exactly once and never write the tag names inside a section.\n"
    ]
    for tag in tags:
        desc = SECTION_INSTRUCTIONS.get(tag, f"the {tag_name(tag)} part of your reasoning.")
        close = closing_tag(tag) or ""
        lines.append(f"{tag} {desc} {close}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Token -> byte strings (offsets for tag search)
# ---------------------------------------------------------------------------
def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 / Qwen2 byte-level BPE byte <-> unicode map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_SP_BYTE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def build_token_bytes(tokenizer, ids: Sequence[int]) -> Dict[int, bytes]:
    """
    Byte string of every distinct id in `ids` (one tokenizer call per batch).
    Special tokens map to b"" (they occupy a position but no text).

    Byte-level BPE (has `byte_decoder`): exact bytes, so multi-byte characters split
    across tokens keep correct cumulative offsets. SentencePiece: '<0xNN>' byte-fallback
    pieces and '▁' -> ' '. Anything else: the token string encoded as UTF-8.
    """
    uniq = sorted({int(i) for i in ids})
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    toks = tokenizer.convert_ids_to_tokens(uniq)
    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    out: Dict[int, bytes] = {}
    for tid, tok in zip(uniq, toks):
        if tid in special or tok is None:
            out[tid] = b""
        elif byte_decoder is not None:
            try:
                out[tid] = bytes(byte_decoder[c] for c in tok)
            except KeyError:  # added (non byte-level) token: literal text
                out[tid] = tok.encode("utf-8")
        else:
            m = _SP_BYTE.fullmatch(tok)
            out[tid] = bytes([int(m.group(1), 16)]) if m else tok.replace("▁", " ").encode("utf-8")
    return out


# ---------------------------------------------------------------------------
# 3.1 Segment split
# ---------------------------------------------------------------------------
def _find_all(haystack: bytes, needle: bytes) -> List[int]:
    out, start = [], 0
    while needle:
        p = haystack.find(needle, start)
        if p < 0:
            break
        out.append(p)
        start = p + 1
    return out


def split_response_into_segments(
    response_ids,
    response_mask_row,
    tokenizer,
    tags: Sequence[str],
    token_bytes: Optional[Dict[int, bytes]] = None,
    strict_order: bool = True,
) -> Tuple[List[str], List[Tuple[int, int]], bool]:
    """
    -> (items, spans, compliant)

    items[j]  inner text of section j (opening/closing tags stripped).
    spans[j]  (start, end) in tensor coordinates of `response_ids` (end exclusive):
              from the token containing the first byte of the j-th opening tag to the
              token before the (j+1)-th opening tag; the last section ends at the last
              valid token. Tokens before the first tag (preamble) belong to no section.

    Compliant (strict_order): the opening tags found are an in-order prefix of `tags`,
    each at most once. strict_order=False tolerates repeats by keeping only the first
    occurrence of each tag (still an in-order prefix). Non-compliant -> ([], [], False).
    """
    ids_t = torch.as_tensor(response_ids)
    mask_t = torch.as_tensor(response_mask_row).bool()
    positions = torch.nonzero(mask_t, as_tuple=False).flatten().tolist()
    if not positions or not tags:
        return [], [], False
    ids = ids_t[positions].tolist()
    if token_bytes is None:
        token_bytes = build_token_bytes(tokenizer, ids)

    pieces = [token_bytes.get(int(t), b"") for t in ids]
    starts, offset = [], 0
    for piece in pieces:
        starts.append(offset)
        offset += len(piece)
    text = b"".join(pieces)

    tag_bytes = [t.encode("utf-8") for t in tags]
    occurrences = sorted((p, k) for k, tb in enumerate(tag_bytes) for p in _find_all(text, tb))
    if not strict_order:
        seen, firsts = set(), []
        for p, k in occurrences:
            if k not in seen:
                seen.add(k)
                firsts.append((p, k))
        occurrences = firsts
    if not occurrences or [k for _, k in occurrences] != list(range(len(occurrences))):
        return [], [], False

    # token index (within valid positions) containing each tag's first byte;
    # zero-length (special) tokens never win because a non-empty token shares their start
    tok_idx = [bisect_right(starts, p) - 1 for p, _ in occurrences]
    items, spans = [], []
    n = len(positions)
    for j, (p, k) in enumerate(occurrences):
        lo = tok_idx[j]
        hi = tok_idx[j + 1] if j + 1 < len(occurrences) else n
        s = positions[lo]
        e = positions[hi - 1] + 1 if hi > lo else s
        spans.append((s, e))

        body_start = p + len(tag_bytes[k])
        body_end = occurrences[j + 1][0] if j + 1 < len(occurrences) else len(text)
        body = text[body_start:body_end]
        close = closing_tag(tags[k])
        if close is not None:
            cut = body.find(close.encode("utf-8"))
            if cut >= 0:
                body = body[:cut]
        items.append(body.decode("utf-8", errors="replace").strip())
    return items, spans, True


def split_batch(
    responses,
    response_mask,
    tokenizer,
    tags: Sequence[str],
    strict_order: bool = True,
) -> Tuple[List[List[str]], List[List[Tuple[int, int]]], Dict[str, float]]:
    """
    -> (seg_items, seg_spans, stats). Token bytes are built once for the whole batch.
    Never raises on a malformed response: errors are counted in stats["vlm/split_errors"].
    """
    responses = torch.as_tensor(responses).cpu()
    mask = torch.as_tensor(response_mask).cpu().bool()
    token_bytes = build_token_bytes(tokenizer, responses[mask].tolist())

    seg_items: List[List[str]] = []
    seg_spans: List[List[Tuple[int, int]]] = []
    n_compliant, n_errors, n_segments = 0, 0, 0
    tag_tokens = [0] * len(tags)
    tag_counts = [0] * len(tags)
    preamble_tokens, compliant_tokens = 0, 0
    for i in range(responses.shape[0]):
        try:
            items, spans, ok = split_response_into_segments(
                responses[i], mask[i], tokenizer, tags, token_bytes=token_bytes, strict_order=strict_order
            )
        except Exception:
            items, spans, ok = [], [], False
            n_errors += 1
        seg_items.append(items)
        seg_spans.append(spans)
        if not ok:
            continue
        n_compliant += 1
        n_segments += len(spans)
        valid = int(mask[i].sum())
        compliant_tokens += valid
        preamble_tokens += int(mask[i, :spans[0][0]].sum())
        for j, (s, e) in enumerate(spans):
            tag_tokens[j] += int(mask[i, s:e].sum())
            tag_counts[j] += 1

    n = max(responses.shape[0], 1)
    stats = {
        "vlm/format_compliance_rate": n_compliant / n,
        "vlm/segments_per_response": n_segments / n,
        "vlm/preamble_token_frac": preamble_tokens / max(compliant_tokens, 1),
        "vlm/split_errors": float(n_errors),
    }
    for j, tag in enumerate(tags):
        stats[f"vlm/seg_tokens/{tag_name(tag)}"] = tag_tokens[j] / max(tag_counts[j], 1)
    return seg_items, seg_spans, stats


# ---------------------------------------------------------------------------
# 3.2 Node matching: cluster_fn(items, depth) -> one canonical label per item
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def exact_normalized_cluster_fn(items: List[Any], depth: int) -> List[Hashable]:
    return [normalize_text(x) for x in items]


def identity_cluster_fn(items: List[Any], depth: int) -> List[Hashable]:
    """Token-level TEMPO: items are token ids."""
    return list(items)


def segment_key_fn(item: str, depth: int) -> Hashable:
    """
    HOOK for node_match=key_fn: return a discrete key for one section's inner text.
    Two sibling sections are the same node iff their keys are equal.

    depth is the section index (0 = first tag). Example idea for the default tags:
    depth 2 (<decision>) -> the leading meta-action ("DECELERATE", "STOP", ...);
    other depths -> a coarse summary key, or normalize_text(item) to fall back to exact.
    Point tempo_cfg.key_fn at your own "module.path:function" instead of editing this.
    """
    raise NotImplementedError(
        "tempo node_match=key_fn: implement segment_key_fn in navsim/agents/negdrive/utils/tempo.py "
        "or set agent.tempo_cfg.key_fn='module.path:function'."
    )


segment_key_fn._is_stub = True


def bag_of_words_embedder(texts: List[str]) -> np.ndarray:
    """Default embedder for node_match=embedding: word-count vectors over the call's vocabulary."""
    docs = [re.findall(r"\w+", str(t).lower()) for t in texts]
    vocab = {w: k for k, w in enumerate(sorted({w for d in docs for w in d}))}
    X = np.zeros((len(texts), max(len(vocab), 1)), dtype=np.float64)
    for i, d in enumerate(docs):
        for w in d:
            X[i, vocab[w]] += 1.0
    return X


def agglomerative_labels(X: np.ndarray, sim_threshold: float) -> List[int]:
    """Average-linkage agglomerative clustering on cosine similarity; merge while avg sim >= threshold."""
    n = X.shape[0]
    if n <= 1:
        return [0] * n
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    S = Xn @ Xn.T
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, pair = -np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                sim = S[np.ix_(clusters[a], clusters[b])].mean()
                if sim > best:
                    best, pair = sim, (a, b)
        if best < sim_threshold:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = [0] * n
    for c, members in enumerate(clusters):
        for i in members:
            labels[i] = c
    return labels


def import_callable(path: str) -> Callable:
    """'package.module:function' -> callable."""
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError(f"Expected 'module.path:function', got {path!r}")
    return getattr(importlib.import_module(module), name)


def make_cluster_fn(
    node_match: str,
    sim_threshold: float = 0.9,
    key_fn: Optional[Callable] = None,
    embedder: Optional[Callable] = None,
) -> Callable[[List[Any], int], List[Hashable]]:
    if node_match == "exact_normalized":
        return exact_normalized_cluster_fn
    if node_match == "key_fn":
        fn = key_fn or segment_key_fn
        if getattr(fn, "_is_stub", False):
            # fail at setup, not mid-training
            fn("", 0)
        return lambda items, depth: [fn(x, depth) for x in items]
    if node_match == "embedding":
        emb = embedder or bag_of_words_embedder
        return lambda items, depth: agglomerative_labels(np.asarray(emb(list(items)), dtype=np.float64), sim_threshold)
    raise ValueError(f"Unknown tempo node_match: {node_match!r}")


# ---------------------------------------------------------------------------
# 3.3 / 3.4 Advantage (reference implementation)
# ---------------------------------------------------------------------------
def compute_tempo_advantage_seg(scores, response_mask, group_ids, seg_items, seg_spans, cluster_fn,
                                td_coef=1.0, eps=1e-6, norm_by_std=True, unbiased_std=True, delta_mode="broadcast"):
    """
    scores [B] raw rewards, response_mask [B, L], group_ids [B] (same id = same prompt),
    seg_items / seg_spans from split_batch. Returns (advantages [B, L] float32 on
    response_mask.device, zero at masked positions; stats dict).

    td_coef=0 gives the GRPO advantage (r_i - mean) / (std + eps) on every valid token.
    """
    dev = response_mask.device
    r_all = scores.detach().double().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores, dtype=np.float64)
    mask = response_mask.detach().cpu().numpy().astype(bool)
    B, L = mask.shape
    adv = np.zeros((B, L), dtype=np.float64)
    groups = defaultdict(list)
    for i, g in enumerate(group_ids): groups[g].append(i)

    n_valid, n_nz, abs_delta, n_branch, depth_sum, n_depth0 = int(mask.sum()), 0, 0.0, 0, 0, 0
    n_nodes_sum, n_resp_sum = 0, 0
    for g, idx in groups.items():
        r = r_all[idx]; G = len(idx)
        if G == 1: mean, std = 0.0, 1.0
        else:      mean, std = r.mean(), r.std(ddof=1 if unbiased_std else 0)
        denom = (std + eps) if norm_by_std else 1.0
        base = (r - mean) / denom
        for m, i in enumerate(idx): adv[i, mask[i]] = base[m]          # GRPO (MC) term on all valid tokens

        active, j = [list(range(G))], 0
        while active:
            nxt = []
            for members in active:
                v_parent = r[members].mean()
                cands = [m for m in members if j < len(seg_items[idx[m]])]
                labels = cluster_fn([seg_items[idx[m]][j] for m in cands], j) if cands else []
                buckets = defaultdict(list)
                for m, lab in zip(cands, labels): buckets[lab].append(m)
                n_nodes_sum += len(buckets); n_resp_sum += len(cands)   # merge-rate: nodes vs responses
                if len(buckets) > 1:
                    n_branch += 1; depth_sum += j; n_depth0 += int(j == 0)
                for sub in buckets.values():
                    delta = 0.0 if len(sub) == len(members) else r[sub].mean() - v_parent
                    if delta != 0.0:
                        for m in sub:
                            s, e = seg_spans[idx[m]][j]
                            if delta_mode == "first_token": e = min(e, s + 1)
                            adv[idx[m], s:e] += td_coef * delta / denom
                            k = int(mask[idx[m], s:e].sum()); n_nz += k; abs_delta += abs(delta) * k
                    if len(sub) > 1: nxt.append(sub)                   # singletons stop: delta = 0 below them
            active, j = nxt, j + 1

    adv *= mask
    out = torch.as_tensor(adv, dtype=torch.float32, device=dev)
    stats = {"tempo/frac_tokens_delta_nonzero": n_nz / max(n_valid, 1),
             "tempo/mean_abs_delta": abs_delta / max(n_valid, 1),
             "tempo/branch_nodes_per_group": n_branch / max(len(groups), 1),
             "tempo/mean_branch_depth": depth_sum / max(n_branch, 1),
             "tempo/frac_branches_at_root": n_depth0 / max(n_branch, 1),
             "tempo/nodes_per_response": n_nodes_sum / max(n_resp_sum, 1)}   # 1.0 => nothing ever merges
    return out, stats


# ---------------------------------------------------------------------------
# Config + rollout helpers (used by the Lightning module)
# ---------------------------------------------------------------------------
@dataclass
class TempoConfig:
    enabled: bool = False
    td_coef: float = 1.0
    delta_mode: str = "broadcast"          # broadcast | first_token
    node_match: str = "exact_normalized"   # exact_normalized | key_fn | embedding
    sim_threshold: float = 0.9             # embedding: cosine threshold (average linkage)
    key_fn: Optional[str] = None           # key_fn: "module.path:function" (default hook: segment_key_fn)
    embedder: Optional[str] = None         # embedding: "module.path:function" (default: bag of words)
    strict_order: bool = True
    warn_compliance_below: float = 0.5
    warn_nodes_per_response_above: float = 0.99

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "TempoConfig":
        cfg = dict(cfg or {})
        out = cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})
        if out.delta_mode not in ("broadcast", "first_token"):
            raise ValueError(f"Unknown tempo delta_mode: {out.delta_mode!r}")
        if out.node_match not in ("exact_normalized", "key_fn", "embedding"):
            raise ValueError(f"Unknown tempo node_match: {out.node_match!r}")
        return out

    def build_cluster_fn(self) -> Callable[[List[Any], int], List[Hashable]]:
        return make_cluster_fn(
            self.node_match,
            sim_threshold=self.sim_threshold,
            key_fn=import_callable(self.key_fn) if self.key_fn else None,
            embedder=import_callable(self.embedder) if self.embedder else None,
        )


def stack_rollout_responses(gen_outputs: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    G generation outputs (each over the same B prompts) -> response tensor [B*G, L_max],
    mask [B*G, L_max] (right-padded with 0) and per-rollout response lengths [G].
    Row b*G + g holds scene b, rollout g. Positions match compute_response_logprobs_tokens:
    response position p is full_ids[:, response_start_idx + p], mask from attention_mask.
    """
    G = len(gen_outputs)
    B = gen_outputs[0].full_ids.shape[0]
    resp = [o.full_ids[:, o.response_start_idx:].detach().cpu() for o in gen_outputs]
    msk = [o.attention_mask[:, o.response_start_idx:].detach().cpu() for o in gen_outputs]
    lengths = [r.shape[1] for r in resp]
    L = max(lengths)
    responses = torch.zeros(B * G, L, dtype=resp[0].dtype)
    mask = torch.zeros(B * G, L, dtype=torch.long)
    for g in range(G):
        rows = torch.arange(B) * G + g
        responses[rows, :lengths[g]] = resp[g]
        mask[rows, :lengths[g]] = msk[g].long()
    return responses, mask, lengths
