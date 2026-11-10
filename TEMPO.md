# Task: Add TEMPO (segment-level prefix tree, branch-gated TD advantage) as an alternative to GRPO advantage in my PyTorch Lightning VLM RL code

Work in the current branch. Ignore the PPS algorithm you implemented previously.

## 1. Idea (paper: "Exploiting Tree Structure for Credit Assignment in RL Training of LLMs", arXiv 2509.18314)
My project already trains a VLM with GRPO using verifiable (binary) rewards. I want to add another advantage estimator, **TEMPO**:
- Keep GRPO's group-relative outcome signal (the "MC" term).
- Add a TD correction derived from a **prefix tree over semantic segments** (4-5 CoT sections per response) built from the G responses
  sampled for the same prompt. Deviation from the paper: the paper's nodes are tokens; mine are segments.
- No critic, no PRM, no extra sampler, no extra rollouts. Fully on-policy. Same loop and cost as GRPO.

Do NOT change the policy loss, the sampling procedure, the reward function, or the optimizer. The ONLY change to generation is the prompt text
(section 3.0). Everything else stays as in my current config. With TEMPO off and `vlm.structured_cot` off, behavior must be bit-for-bit unchanged.


## 2. Pipeline per training step
1. Sample G responses per (image + prompt) from the old policy, using the **structured CoT prompt** (3.0); each gets a binary reward r_i from the existing reward function (unchanged).
2. NEW: split each response into semantic segments + token spans (`split_batch`, 3.1). CPU, no autograd.
3. NEW (replaces the GRPO advantage): build the segment-level prefix tree per group and compute TEMPO advantages (3.2-3.4).
4. Policy update with the existing loss (unchanged).

## 3. Specification

### 3.0 Structured CoT prompt (VLM side)
Goal: the model emits 4-5 ordered sections so segment boundaries are deterministic.
reference existing CoT prompt (SFT, or NegDrive's), design a section-format instruction;
No format reward, no reward-function change, no SFT warm-up in this task.

### 3.1 Segment split (node definition) -- `split_response_into_segments`, `split_batch`
- Segment j = from the first token of the j-th opening tag to the token before the (j+1)-th opening tag (the closing tag and trailing whitespace belong
  to segment j). The last segment ends at the last valid token. Tokens before the first opening tag (preamble) belong to no segment and receive only the GRPO term.
- `seg_items[i][j]` = INNER text of the section (tags stripped); this is what node matching compares.
- `seg_spans[i][j]` = (start, end) positions in the (B, L) response tensor (end exclusive), derived from the rollout token ids and the response mask,
  not from re-tokenizing text.
- Compliance (`tempo.strict_order: true`): compliant = opening tags found are an in-order PREFIX of `vlm.segment_tags`, each at most once (a prefix is allowed so
  truncated responses keep their completed segments; they keep whatever reward the existing reward function gives them). Anything else (repeated tag, out-of-order,
  skipped tag, no tags, tag text quoted inside reasoning) is non-compliant: `seg_items=[]`, `seg_spans=[]`, the response gets only the GRPO term. Never raise on a
  malformed response.
```python
def split_response_into_segments(response_ids, response_mask_row, tokenizer, tags):
    """-> (items: list[str], spans: list[tuple[int, int]], compliant: bool)"""
def split_batch(responses, response_mask, tokenizer, tags):
    """-> (seg_items: list[list[str]], seg_spans: list[list[tuple]], stats: dict)"""
```
- Do the tag search on decoded text, then map each tag start to the token containing its first character. For byte-level BPE tokenizers, compute offsets from
  cumulative byte offsets of the token byte strings (not per-token `len(decode([id]))`) so multi-byte characters near a boundary do not shift spans.
  If a tag merges with adjacent text/newline into one token, the segment starts at the token containing the first character of the tag.
  Batch the decoding (once per batch), not per token. Masks may be right-padded, left-padded or non-contiguous: take positions from the mask and write spans in tensor coordinates.

### 3.2 Segment-level prefix tree per prompt group
- A node = a prefix of segments. The image+prompt is the shared root. Two responses share the node at depth j iff they share the same parent node AND their
  segment j is judged equivalent by `cluster_fn`, decided only among siblings under the same parent.
  **Being under the same tag is NOT sufficient** (that would put all responses in one chain: no branching, TEMPO == GRPO).
- V(node) = mean RAW outcome reward of all responses passing through the node. Responses with no segments (non-compliant) count at the root only.
- `cluster_fn(items: list, depth: int) -> list[hashable]` returns one canonical label per item (labels, not pairwise decisions, so it is transitive).
  Config `tempo.node_match`:
  1. `exact_normalized`: normalized text (strip/lowercase/collapse whitespace). Default. WARNING: free-form text rarely matches exactly; if the merge rate is
     ~0, log a prominent warning that TEMPO is behaving like GRPO.
  2. `key_fn`: user-supplied function extracting a discrete key from a segment (leave a clear hook; I will fill it in).
  3. `embedding`: agglomerative clustering (average linkage, cosine threshold `tempo.sim_threshold`) among siblings, pluggable embedder.
- Token-level TEMPO is the special case seg_items = token ids, single-token spans, identity cluster_fn (used in the tests).

### 3.3 Advantage
For token t in segment j of response i, with delta_{i,j} = V(node_j) - V(node_{j-1}):

    A_{i,t} = ( (r_i - mean(r)) + td_coef * delta_{i,j} ) / (std(r) + eps)

- `tempo.delta_mode`: `broadcast` (default; delta applied to every token of segment j) or `first_token` (ablation).
- delta is nonzero only right after a branching node; set it to exactly 0.0 when the child contains all members of the node.
  Once a response is alone in its subtree, V = r_i below it, so later segments get only the base term.
- `td_coef` (default 1.0, the paper's value) is a NEW ablation knob; td_coef=0 must reproduce my existing GRPO advantage exactly.
- Use the SAME std definition (unbiased vs biased), eps, and "normalize by std" switch as my existing GRPO code (if normalization is off, drop the division in
  both terms). Group size 1: mean=0, std=1 (A = r_i, no NaN). All-equal rewards: A = 0. Do NOT require or compute token entropies.

### 3.4 Reference implementation
```python
import numpy as np, torch
from collections import defaultdict

def compute_tempo_advantage_seg(scores, response_mask, group_ids, seg_items, seg_spans, cluster_fn,
                                td_coef=1.0, eps=1e-6, norm_by_std=True, unbiased_std=True, delta_mode="broadcast"):
    dev = response_mask.device
    r_all = scores.detach().double().cpu().numpy()
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
```

### 3.5 Integration (Lightning)
- **All G responses of a prompt must be in the same call on the same rank.** If rollouts are sharded across DDP ranks, compute advantages where complete groups
  are local (or after gathering). Add an assert.
- Compute advantages on the full rollout batch BEFORE splitting into micro-batches. Run under `torch.no_grad()` on CPU; scores float64 internally; return
  advantages in the dtype/device/shape the loss expects, zero at masked positions.
- Truncated responses: treat exactly as the current reward function does; no special casing.

## 5. Logging (via the LightningModule, `self.log(..., on_step=True, sync_dist=True)`, or return the stats dict)
`tempo/frac_tokens_delta_nonzero`, `tempo/mean_abs_delta`, `tempo/branch_nodes_per_group`, `tempo/mean_branch_depth`, `tempo/frac_branches_at_root`,
`tempo/nodes_per_response`, `vlm/format_compliance_rate`, mean segments per response, mean token length per segment (per tag), fraction of preamble tokens,
plus the usual GRPO metrics. Report prominently if: compliance is low (TEMPO silently degrades to GRPO on non-compliant responses), or
`nodes_per_response` is ~1.0 / `frac_tokens_delta_nonzero` is ~0 (nothing merges or branches: TEMPO == GRPO).

## 6. Paper hyperparameters (reference only; keep mine unless I say otherwise)
G=6, lr 1e-6 (AdamW), mini-batch 64, max response length 1024, clip 0.2/0.28, token-mean loss, no KL, binary reward. Larger G helps GRPO and TEMPO alike.


## 8. Notes
- TEMPO only acts right after a branching node. If all G responses fall into the same node at every depth (or none merge), the TD signal collapses to GRPO.
- Telescoping: TD terms along a response sum (per segment) to r_i - mean(r), so TEMPO redistributes the GRPO signal across branch points instead of adding total signal.
