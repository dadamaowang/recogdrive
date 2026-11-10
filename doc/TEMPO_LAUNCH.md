# Launching TEMPO Training

TEMPO is GRPO with an extra TD term computed over a prefix tree of the CoT sections that the G rollouts of a scene share.
- Algorithm spec: `TEMPO.md`
- Implementation: `navsim/agents/negdrive/utils/tempo.py`
- Launch script: `scripts/training/train_tempo.sh`

---

## 1. Quick start

```bash
# 1. smoke test on 1 GPU (1 train + 1 val batch, then exit)
FAST_DEV_RUN=true GPUS=1 BATCH_SIZE=1 G=2 bash scripts/training/train_tempo.sh

# 2. full TEMPO run (8 GPUs, defaults below)
bash scripts/training/train_tempo.sh

# 3. GRPO baseline with the same structured prompt (the fair comparison)
TEMPO_ENABLED=false bash scripts/training/train_tempo.sh
```

The script runs the CPU unit tests (`tests/test_tempo.py`) before training. Skip them with `RUN_TESTS=false`.

---

## 2. Before the first run

1. **Paths.** Defaults assume the server layout used by `train_negdrive.sh`. Override any of them from the environment:

   | Variable | Default |
   |---|---|
   | `NAVSIM_DEVKIT_ROOT` | `/root/recogdrive` (this repo on the server) |
   | `NAVSIM_EXP_ROOT` | `/root/navsim_workspace/exps` (outputs) |
   | `OPENSCENE_DATA_ROOT` | `/root/navsim_workspace/dataset` |
   | `MODEL_ROOT` | `/root/navsim_workspace/models` |
   | `VLM_PATH` | `$MODEL_ROOT/ReCogDrive-VLM-2B` |
   | `DIFF_PATH` | `$MODEL_ROOT/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt` |
   | `TRAIN_METRIC_CACHE_PATH` | `$NAVSIM_EXP_ROOT/metric_cache_train` |
   | `VAL_METRIC_CACHE_PATH` | `$NAVSIM_EXP_ROOT/metric_cache` |
   | `CONDA_ENV` | `ndfl03` |

2. **SwanLab.** The script does not store an API key. Run `export SWANLAB_API_KEY=...` in your shell, or use `SWANLAB_MODE=offline`.

3. **Git state.** Run on the branch that contains `utils/tempo.py` (currently `v0_on_jz`, uncommitted).

---

## 3. What the script sets

The script always passes `agent.rl_algorithm=grpo`, since TEMPO replaces the GRPO advantage. Everything else comes from environment variables:

### Rollouts and optimization
| Variable | Default | Hydra key | Note |
|---|---|---|---|
| `G` | 4 | `agent.per_sample_rollout` | rollouts per scene; the paper uses 6 |
| `MAX_TEXT_TOKENS` | 512 | `agent.max_text_tokens` | **keep ≥ 512**: 5 sections + 8 waypoints don't fit in the YAML default of 128 |
| `ROLLOUT_TEMPERATURE` | 0.7 | `agent.rollout_temperature` | |
| `BATCH_SIZE` | 8 | `dataloader.params.batch_size` | scenes per GPU per step |
| `VLM_LR` | 4e-5 | `agent.vlm_lr` | same as the NSR runs |
| `MAX_STEPS` | 450 | `trainer.params.max_steps` | |
| `VAL_EVERY` | 9 | `trainer.params.val_check_interval` | steps between validations (LoRA saved at each) |
| `VAL_BATCHES` | 1.0 | `trainer.params.limit_val_batches` | fraction or number of val batches |
| `SEED` | 42 | `seed` | |
| `GPUS` | 8 | `--nproc_per_node`, `trainer.params.devices` | |
| `FAST_DEV_RUN` | false | `trainer.params.fast_dev_run` | |

### Structured CoT and TEMPO
| Variable | Default | Hydra key | Note |
|---|---|---|---|
| `STRUCTURED_COT` | true | `agent.structured_cot` | tagged-section prompt; required by TEMPO |
| `SEGMENT_TAGS` | (YAML) | `agent.segment_tags` | default `<perception> <prediction> <decision> <check> <trajectory>`; Hydra list syntax |
| `TEMPO_ENABLED` | true | `agent.tempo_cfg.enabled` | false gives plain GRPO |
| `TD_COEF` | 1.0 | `agent.tempo_cfg.td_coef` | 0 gives exactly the GRPO advantage |
| `DELTA_MODE` | broadcast | `agent.tempo_cfg.delta_mode` | `broadcast` \| `first_token` |
| `NODE_MATCH` | exact_normalized | `agent.tempo_cfg.node_match` | `exact_normalized` \| `key_fn` \| `embedding` |
| `SIM_THRESHOLD` | 0.9 | `agent.tempo_cfg.sim_threshold` | `embedding` only |
| `KEY_FN` | (hook) | `agent.tempo_cfg.key_fn` | `"module.path:function"` |
| `EMBEDDER` | (bag of words) | `agent.tempo_cfg.embedder` | `"module.path:function"` |
| `STRICT_ORDER` | true | `agent.tempo_cfg.strict_order` | |
| `EXP_NAME` | auto | `experiment_name` | e.g. `tempo_exact_normalized_td1.0_broadcast_G4_s42` |

---

## 4. Choosing `NODE_MATCH`

This decides whether TEMPO does anything. Two sections are the same tree node only if `cluster_fn` gives them the same label.

| Mode | When to use | Command |
|---|---|---|
| `exact_normalized` | Sanity check only. Free-form sections almost never match word for word, so TEMPO ≈ GRPO. | default |
| `key_fn` | **Recommended first real run.** Key `<decision>` on its meta-action (the prompt makes it start with one of KEEP LANE / DECELERATE / STOP / …). | `NODE_MATCH=key_fn KEY_FN=my_keys:segment_key` |
| `embedding` | No hand-written key. Sections whose average cosine similarity ≥ threshold merge. | `NODE_MATCH=embedding SIM_THRESHOLD=0.8` |

**Writing a key function.** Put it in any importable module (on `PYTHONPATH`, e.g. under `navsim/`):

```python
# navsim/agents/negdrive/utils/my_keys.py
import re
ACTIONS = ["KEEP LANE", "ACCELERATE", "DECELERATE", "STOP", "TURN LEFT", "TURN RIGHT",
           "CHANGE LANE LEFT", "CHANGE LANE RIGHT"]

def segment_key(item: str, depth: int):
    text = item.upper()
    if depth == 2:                                   # <decision>
        for a in ACTIONS:
            if text.startswith(a) or re.search(rf"\b{a}\b", text[:40]):
                return a
        return "OTHER"
    return depth                                     # all other sections: one shared node
```

```bash
NODE_MATCH=key_fn KEY_FN=navsim.agents.negdrive.utils.my_keys:segment_key bash scripts/training/train_tempo.sh
```

With this key, every response shares the nodes at depths 0–1. The tree branches at `<decision>` by meta-action, so the decision section gets credited when one action wins more often than another.

To make the default hook work without `KEY_FN`, edit `segment_key_fn` in `utils/tempo.py`. If `NODE_MATCH=key_fn` is set and the hook is still a stub, training fails at setup (on purpose).

**Embedding.** The default embedder is bag-of-words. A sentence-embedding model usually separates sections better. Pass any `fn(list[str]) -> np.ndarray [n, d]` via `EMBEDDER=module:function`.

---

## 5. Recommended run plan

| # | Run | Command | Purpose |
|---|---|---|---|
| 0 | Smoke test | `FAST_DEV_RUN=true GPUS=1 BATCH_SIZE=1 G=2 bash scripts/training/train_tempo.sh` | code path runs on GPU |
| 1 | Prompt check | same as #0, read the printed `生成文字检查` text | the model actually writes the 5 tagged sections |
| 2 | GRPO + structured prompt | `TEMPO_ENABLED=false bash scripts/training/train_tempo.sh` | baseline |
| 3 | TEMPO, key_fn | `NODE_MATCH=key_fn KEY_FN=... bash scripts/training/train_tempo.sh` | main run |
| 4 | TEMPO, embedding | `NODE_MATCH=embedding SIM_THRESHOLD=0.8 bash scripts/training/train_tempo.sh` | key-free variant |
| 5 | Ablations | `TD_COEF=0` (must match run 2), `TD_COEF=0.5`, `DELTA_MODE=first_token`, `G=6` | |

Runs 2–5 should differ only in the TEMPO settings. Keep `SEED`, `G`, `BATCH_SIZE`, `VLM_LR` and `MAX_TEXT_TOKENS` fixed across them.

---

## 6. What to watch (SwanLab project `NegDriveComplete`)

### TEMPO health: check first
| Metric | Healthy | If not |
|---|---|---|
| `train/vlm/format_compliance_rate` | > 0.7 | Non-compliant responses get only the GRPO term. Check the generated text; consider SFT on the tagged format (out of scope here) or simplify the tags. Below 0.5 a `[TEMPO WARNING]` is printed. |
| `train/tempo/nodes_per_response` | clearly < 1.0 | ≈ 1.0 means no section ever matches a sibling, so TEMPO == GRPO. Switch to `key_fn` / `embedding`, or lower `SIM_THRESHOLD`. A warning is printed. |
| `train/tempo/frac_tokens_delta_nonzero` | > 0 | 0 means nothing branched this step. |
| `train/tempo/branch_nodes_per_group` | ≥ 1 | branch points per scene |
| `train/tempo/mean_branch_depth`, `frac_branches_at_root` | depends on key | With the meta-action key you expect branching at depth 2 (`<decision>`) |
| `train/tempo/mean_abs_delta` | > 0 | size of the TD correction |
| `train/vlm/segments_per_response`, `vlm/seg_tokens/<tag>`, `vlm/preamble_token_frac` | ~5 sections, no section near 0 tokens, small preamble | Truncation or a skipped section shows up here first |
| `train/vlm/split_errors` | 0 | Non-zero means a splitter exception (the response is treated as non-compliant) |

### Usual GRPO and outcome metrics
`train/grpo_mean_reward`, `train/grpo_valid_groups`, `train/grpo_zero_std_groups`, `train/grpo_num_updates`, `train/grpo_pg_loss`, `train/grad_norm`, `train/nc_fail_rate`, `train/dac_fail_rate`, `val/pdm_score`, `val/no_at_fault_collisions`, `val/drivable_area_compliance`, `val/ego_progress`.

Groups with all-equal rewards (`grpo_zero_std_groups`) get no update in GRPO or TEMPO. Higher `G` reduces how often that happens.

---

## 7. Outputs

- Logs and Hydra config: `$NAVSIM_EXP_ROOT/$EXP_NAME/<timestamp>/`
- LoRA checkpoints, one per validation: `.../checkpoints/iter_<step>_lora/`
- Evaluate a checkpoint with your usual CoT evaluation scripts. Pass `agent.structured_cot=true` so evaluation uses the same prompt the policy was trained with.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `tempo_cfg.enabled requires rl_algorithm=grpo` | Don't pass `agent.rl_algorithm` yourself; the script sets `grpo` |
| `requires agent.structured_cot=true` | Use `STRUCTURED_COT=true` (the default) |
| `NotImplementedError: tempo node_match=key_fn` | Set `KEY_FN=module:function` or implement `segment_key_fn` |
| `AssertionError: TEMPO needs all G rollouts…` | Should not happen (each rank generates its own groups). Report it with the batch/G settings. |
| Compliance near 0 | The model isn't following the tag format. Read the logged generations; responses cut at `MAX_TEXT_TOKENS` keep their completed sections, but a missing first tag makes the response non-compliant. |
| OOM | Lower `BATCH_SIZE` first, then `G`. TEMPO itself is CPU-side and adds no GPU memory. |
| Results identical to GRPO | Check `tempo/nodes_per_response` and `frac_tokens_delta_nonzero` (Section 6) |
