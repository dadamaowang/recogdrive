"""
CPU tests for TEMPO (navsim/agents/negdrive/utils/tempo.py).

Run without the full navsim stack:
    python tests/test_tempo.py            # or: pytest tests/test_tempo.py
Needs only torch + numpy.
"""

import importlib.util
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "tempo", os.path.join(ROOT, "navsim/agents/negdrive/utils/tempo.py")
)
tempo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tempo)

TAGS = ["<a>", "<b>", "<c>"]


# ---------------------------------------------------------------------------
# Fake tokenizers: the caller chooses the token boundaries (as byte chunks)
# ---------------------------------------------------------------------------
class ByteLevelTok:
    """Qwen2/GPT-2 style: tokens are byte-level unicode strings, has byte_decoder."""

    def __init__(self):
        b2u = tempo._bytes_to_unicode()
        self.byte_encoder = b2u
        self.byte_decoder = {v: k for k, v in b2u.items()}
        self.vocab = {}
        self.inv = {}
        self.all_special_ids = [0]
        self.inv[0] = "<|im_end|>"

    def encode_chunks(self, chunks):
        ids = []
        for c in chunks:
            if c == "<EOS>":
                ids.append(0)
                continue
            b = c if isinstance(c, bytes) else c.encode("utf-8")
            tok = "".join(self.byte_encoder[x] for x in b)
            if tok not in self.vocab:
                tid = len(self.vocab) + 1
                self.vocab[tok] = tid
                self.inv[tid] = tok
            ids.append(self.vocab[tok])
        return ids

    def convert_ids_to_tokens(self, ids):
        return [self.inv[i] for i in ids]


class SentencePieceTok(ByteLevelTok):
    """SentencePiece style: '▁' for spaces, '<0xNN>' byte fallback, no byte_decoder."""

    def __init__(self):
        super().__init__()
        del self.byte_decoder

    def encode_chunks(self, chunks):
        ids = []
        for c in chunks:
            if c == "<EOS>":
                ids.append(0)
                continue
            tok = f"<0x{c[0]:02X}>" if isinstance(c, bytes) else c.replace(" ", "▁")
            if tok not in self.vocab:
                tid = len(self.vocab) + 1
                self.vocab[tok] = tid
                self.inv[tid] = tok
            ids.append(self.vocab[tok])
        return ids


def place(ids, L, start=0, holes=()):
    """Put ids into a length-L row from `start`, skipping masked `holes` positions."""
    row = torch.zeros(L, dtype=torch.long)
    mask = torch.zeros(L, dtype=torch.long)
    pos, k = start, 0
    while k < len(ids):
        if pos in holes:
            pos += 1
            continue
        row[pos] = ids[k]
        mask[pos] = 1
        pos += 1
        k += 1
    return row, mask


def tokens_in(row, mask, span):
    s, e = span
    return [int(row[p]) for p in range(s, e) if mask[p]]


# ---------------------------------------------------------------------------
# 3.1 split
# ---------------------------------------------------------------------------
def test_split_basic_and_preamble():
    tok = ByteLevelTok()
    chunks = ["Sure.", " ", "<a>", " road", " clear", "</a>", "\n", "<b>", " car", " slows", "</b>", "\n",
              "<c>", " brake", "</c>", "<EOS>"]
    ids = tok.encode_chunks(chunks)
    row, mask = place(ids, len(ids) + 3)  # right-padded
    items, spans, ok = tempo.split_response_into_segments(row, mask, tok, TAGS)
    assert ok
    assert items == ["road clear", "car slows", "brake"]
    assert spans == [(2, 7), (7, 12), (12, 16)]  # closing tag + newline in seg j; last ends at last valid
    assert spans[0][0] == 2  # preamble tokens 0..1 belong to no segment


def test_split_padding_layouts():
    tok = ByteLevelTok()
    ids = tok.encode_chunks(["<a>", "x", "<b>", "y", "<c>", "z"])
    for start, holes in [(0, ()), (4, ()), (1, (3, 6))]:  # right-, left-padded, non-contiguous
        row, mask = place(ids, 12, start=start, holes=holes)
        items, spans, ok = tempo.split_response_into_segments(row, mask, tok, TAGS)
        assert ok and items == ["x", "y", "z"]
        got = [tokens_in(row, mask, sp) for sp in spans]
        assert got == [ids[0:2], ids[2:4], ids[4:6]], (start, holes, spans)


def test_split_prefix_is_compliant():
    tok = ByteLevelTok()
    ids = tok.encode_chunks(["<a>", "x", "<b>", "y trunc"])
    row, mask = place(ids, 6)
    items, spans, ok = tempo.split_response_into_segments(row, mask, tok, TAGS)
    assert ok and items == ["x", "y trunc"] and len(spans) == 2


def test_split_noncompliant_cases():
    tok = ByteLevelTok()
    cases = {
        "repeated": ["<a>", "x", "<a>", "y"],
        "out_of_order": ["<b>", "x", "<a>", "y"],
        "skipped": ["<a>", "x", "<c>", "y"],
        "no_tags": ["just", " text"],
        "quoted": ["<a>", " the <b> tag", "<b>", "y"],  # quoting <b> inside <a> repeats <b>
    }
    for name, chunks in cases.items():
        ids = tok.encode_chunks(chunks)
        row, mask = place(ids, len(ids))
        items, spans, ok = tempo.split_response_into_segments(row, mask, tok, TAGS)
        assert (items, spans, ok) == ([], [], False), name


def test_split_tag_merged_with_text_and_multibyte():
    tok = ByteLevelTok()
    e_acute = "é".encode("utf-8")  # 2 bytes, split across two tokens below
    chunks = ["caf", e_acute[:1], e_acute[1:] + b"\n<a", ">ok", "\n<b>", "fin"]
    ids = tok.encode_chunks(chunks)
    row, mask = place(ids, len(ids))
    items, spans, ok = tempo.split_response_into_segments(row, mask, tok, TAGS)
    assert ok, (items, spans)
    # "<a>" starts inside token 2 (which also holds the 2nd byte of é and a newline)
    assert spans[0][0] == 2 and spans[1][0] == 4
    assert items == ["ok", "fin"]


def test_split_sentencepiece():
    tok = SentencePieceTok()
    ids = tok.encode_chunks(["<a>", " road", b"\xc3", b"\xa9", "<b>", " ok"])
    row, mask = place(ids, len(ids))
    items, spans, ok = tempo.split_response_into_segments(row, mask, tok, TAGS)
    assert ok and items == ["roadé", "ok"] and spans == [(0, 4), (4, 6)]


def test_split_batch_stats_and_never_raises():
    tok = ByteLevelTok()
    rows = [["<a>", "x", "<b>", "y"], ["junk"], ["pre", "<a>", "x"]]
    L = 5
    R = torch.zeros(len(rows), L, dtype=torch.long)
    M = torch.zeros(len(rows), L, dtype=torch.long)
    for i, ch in enumerate(rows):
        R[i], M[i] = place(tok.encode_chunks(ch), L)
    items, spans, stats = tempo.split_batch(R, M, tok, TAGS)
    assert [bool(s) for s in spans] == [True, False, True]
    assert math.isclose(stats["vlm/format_compliance_rate"], 2 / 3)
    assert math.isclose(stats["vlm/segments_per_response"], 3 / 3)
    assert math.isclose(stats["vlm/preamble_token_frac"], 1 / 7)
    assert stats["vlm/seg_tokens/a"] == 2.0 and stats["vlm/seg_tokens/b"] == 2.0


# ---------------------------------------------------------------------------
# 3.3 / 3.4 advantage
# ---------------------------------------------------------------------------
def grpo_reference(rewards_bg, eps, unbiased):
    """Same expression as compute_grpo_group_advantages (without skip / clip)."""
    mean = rewards_bg.mean(dim=1, keepdim=True)
    std = rewards_bg.std(dim=1, keepdim=True, unbiased=unbiased)
    return (rewards_bg - mean) / (std + eps)


def test_td0_equals_grpo():
    torch.manual_seed(0)
    B, G, L = 5, 6, 9
    rewards = torch.where(torch.rand(B, G) < 0.5, 1.0, -1.0)
    mask = (torch.rand(B * G, L) < 0.8).long()
    seg_items = [[f"s{np.random.randint(2)}" for _ in range(3)] for _ in range(B * G)]
    seg_spans = [[(0, 3), (3, 6), (6, 9)] for _ in range(B * G)]
    for unbiased in (False, True):
        ref = grpo_reference(rewards, 1e-8, unbiased).reshape(-1, 1) * mask
        adv, _ = tempo.compute_tempo_advantage_seg(
            rewards.reshape(-1), mask, [b for b in range(B) for _ in range(G)], seg_items, seg_spans,
            tempo.exact_normalized_cluster_fn, td_coef=0.0, eps=1e-8, unbiased_std=unbiased,
        )
        assert torch.allclose(adv, ref.float(), atol=1e-6, rtol=0), (adv - ref).abs().max()


def test_group_size_one_and_all_equal():
    mask = torch.ones(3, 4, dtype=torch.long)
    items = [["x"], ["x"], ["y"]]
    spans = [[(0, 4)]] * 3
    adv, _ = tempo.compute_tempo_advantage_seg(torch.tensor([0.7, 1.0, 1.0]), mask, [0, 1, 1], items, spans,
                                               tempo.exact_normalized_cluster_fn)
    assert torch.allclose(adv[0], torch.full((4,), 0.7))  # G = 1: A = r_i
    assert torch.all(adv[1:] == 0) and not torch.isnan(adv).any()  # all-equal rewards


def test_token_level_special_case_and_telescoping():
    # token-level TEMPO: items = token ids, single-token spans, identity cluster_fn
    seqs = [[1, 2, 3], [1, 2, 4], [1, 5, 6], [7, 8, 9]]
    r = torch.tensor([1.0, 0.0, 1.0, 0.0])
    L = 3
    mask = torch.ones(4, L, dtype=torch.long)
    spans = [[(t, t + 1) for t in range(L)] for _ in seqs]
    adv, stats = tempo.compute_tempo_advantage_seg(r, mask, [0] * 4, seqs, spans, tempo.identity_cluster_fn,
                                                   td_coef=1.0, eps=0.0, norm_by_std=False)
    mean = r.mean().item()  # 0.5
    # V(root)=0.5; V([1])=2/3; V([1,2])=0.5; leaves = r_i; V([7])=0
    exp_delta = [[2/3 - .5, .5 - 2/3, 1 - .5], [2/3 - .5, .5 - 2/3, 0 - .5],
                 [2/3 - .5, 1 - 2/3, 0.0], [0 - .5, 0.0, 0.0]]
    for i in range(4):
        exp = torch.tensor([r[i].item() - mean + d for d in exp_delta[i]], dtype=torch.float32)
        assert torch.allclose(adv[i], exp, atol=1e-6), (i, adv[i], exp)
        # telescoping: deltas along the path sum to r_i - mean (each ends as a singleton)
        assert math.isclose(sum(exp_delta[i]), r[i].item() - mean, abs_tol=1e-9)
    assert stats["tempo/branch_nodes_per_group"] == 3.0  # root, under [1], under [1,2]
    assert math.isclose(stats["tempo/frac_branches_at_root"], 1 / 3)


def test_no_merge_and_full_merge_equal_grpo_pattern():
    r = torch.tensor([1.0, -1.0, 1.0, -1.0])
    mask = torch.ones(4, 6, dtype=torch.long)
    spans = [[(0, 3), (3, 6)]] * 4
    base = ((r - r.mean()) / (r.std() + 1e-6)).reshape(-1, 1).expand(4, 6)
    # all identical: one chain, delta = 0 everywhere
    same = [["p", "q"]] * 4
    adv, st = tempo.compute_tempo_advantage_seg(r, mask, [0] * 4, same, spans, tempo.exact_normalized_cluster_fn)
    assert torch.allclose(adv, base.float(), atol=1e-6) and st["tempo/frac_tokens_delta_nonzero"] == 0.0
    # all distinct at depth 0: every response singleton at the root; delta = r_i - mean on segment 0
    diff = [[f"p{i}", "q"] for i in range(4)]
    adv, st = tempo.compute_tempo_advantage_seg(r, mask, [0] * 4, diff, spans, tempo.exact_normalized_cluster_fn)
    assert st["tempo/nodes_per_response"] == 1.0
    assert torch.allclose(adv[:, :3], 2 * base[:, :3].float(), atol=1e-5)
    assert torch.allclose(adv[:, 3:], base[:, 3:].float(), atol=1e-6)


def test_first_token_mode_and_noncompliant_rows():
    r = torch.tensor([1.0, -1.0, 1.0])
    mask = torch.ones(3, 4, dtype=torch.long)
    items = [["x", "u"], ["x", "v"], []]  # row 2 non-compliant: GRPO term only
    spans = [[(0, 2), (2, 4)], [(0, 2), (2, 4)], []]
    adv_b, _ = tempo.compute_tempo_advantage_seg(r, mask, [0] * 3, items, spans, tempo.exact_normalized_cluster_fn,
                                                 norm_by_std=False, delta_mode="broadcast")
    adv_f, _ = tempo.compute_tempo_advantage_seg(r, mask, [0] * 3, items, spans, tempo.exact_normalized_cluster_fn,
                                                 norm_by_std=False, delta_mode="first_token")
    mean = r.mean().item()
    # root: rows 0,1 share "x" (V=0), row 2 has no segment -> delta(x) = 0 - mean
    # under "x": branch u (V=1) / v (V=-1) -> deltas +1 / -1
    assert torch.allclose(adv_b[0], torch.tensor([1 - mean - mean] * 2 + [1 - mean + 1] * 2))
    assert torch.allclose(adv_b[2], torch.full((4,), 1 - mean))
    assert torch.allclose(adv_f[0], torch.tensor([1 - 2 * mean, 1 - mean, 2 - mean, 1 - mean]))


def test_embedding_and_key_fn_hook():
    fn = tempo.make_cluster_fn("embedding", sim_threshold=0.6)
    labels = fn(["car brakes ahead", "the car brakes ahead", "clear road no traffic"], 1)
    assert labels[0] == labels[1] != labels[2]
    try:
        tempo.make_cluster_fn("key_fn")
        raise AssertionError("stub key_fn should raise at setup")
    except NotImplementedError:
        pass
    kf = tempo.make_cluster_fn("key_fn", key_fn=lambda item, depth: item.split()[0].upper())
    assert kf(["stop now", "Stop later", "go"], 2) == ["STOP", "STOP", "GO"]


def test_stack_rollout_responses_layout():
    class Out:
        def __init__(self, ids, start):
            self.full_ids = ids
            self.attention_mask = torch.ones_like(ids)
            self.response_start_idx = start
    B = 2
    o0 = Out(torch.tensor([[9, 9, 1, 2], [9, 9, 3, 4]]), 2)
    o1 = Out(torch.tensor([[9, 9, 5, 6, 7], [9, 9, 8, 0, 0]]), 2)
    resp, mask, lengths = tempo.stack_rollout_responses([o0, o1])
    assert lengths == [2, 3]
    assert resp.tolist() == [[1, 2, 0], [5, 6, 7], [3, 4, 0], [8, 0, 0]]  # row = b * G + g
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]]


def test_structured_prompt_mentions_every_tag_in_order():
    text = tempo.build_structured_cot_instructions(tempo.DEFAULT_SEGMENT_TAGS)
    pos = [text.index(t) for t in tempo.DEFAULT_SEGMENT_TAGS]
    assert pos == sorted(pos) and "</decision>" in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
