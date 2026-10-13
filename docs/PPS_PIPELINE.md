# PPS Pipeline: Planner-Potential Shaping for NegDrive

How the NegDrive variants on branch `dev-neg_v2` work, from data preparation to evaluation.
Each step lists **input → processing → output**, with tensor shapes and code locations.

Contents
1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [Symbols and shapes](#2-symbols-and-shapes)
3. [Stage 0: Prerequisites](#3-stage-0-prerequisites)
4. [Stage 1: Diagnostic (run before training)](#4-stage-1-diagnostic-run-before-training)
5. [Stage 2: One PPS training step, in detail](#5-stage-2-one-pps-training-step-in-detail)
6. [Stage 3: Validation](#6-stage-3-validation)
7. [Stage 4: Evaluation](#7-stage-4-evaluation)
8. [Worked example](#8-worked-example)
9. [How NSR / MARS / PPS differ](#9-how-nsr--mars--pps-differ)
10. [Configuration reference](#10-configuration-reference)
11. [Logged metrics](#11-logged-metrics)
12. [Commands](#12-commands)
13. [Known limitations](#13-known-limitations)

---

## 1. The idea in one minute

NegDrive trains the **VLM** (LoRA) to write better chain-of-thought (CoT). The **diffusion planner** is frozen and turns the CoT's hidden states into a trajectory, which PDM scores for safety.

**Problem 1: planner noise.** The planner is stochastic, so the same CoT can be safe with one noise draw and unsafe with another. NSR gives the whole CoT −1 based on a single draw.
→ **MARS** decodes each CoT under **K noise draws** (the same draws for every CoT of a scene) and uses the *average* failure as the CoT's cost.

**Problem 2: credit.** Even with a correct cost, NSR/MARS blames **every token** of a bad CoT equally, although usually one sentence caused the failure.
→ **PPS** exploits the fact that the VLM is causal: the hidden states of the first `t` CoT tokens are already inside the full forward pass. Feeding a **truncated** hidden-state sequence to the frozen planner shows what the plan would have been if the CoT had stopped there. Scoring these "prefix plans" gives a risk curve along the CoT:

```
prefix:    [prompt only] → [+Perception] → [+Dynamics] → [+Planning] → [full CoT]
risk Φ:        0.00            0.00           0.00         −0.67          −0.67
                                                       ▲
                                     the "Planning" segment made the plan unsafe → blame it
```

Each segment gets extra penalty equal to how much it **lowered** Φ. This uses no extra VLM forward passes and no learned critic; the frozen planner and PDM act as a free critic.

---

## 2. Symbols and shapes

| Symbol | Meaning | Typical value |
|---|---|---|
| `B` | scenes per batch (per GPU) | 8 |
| `G` | CoT rollouts per scene (`agent.per_sample_rollout`) | 4 |
| `K` | planner noise draws per CoT (`mars_cfg.num_planner_samples`) | 4 |
| `P` | max CoT segments per rollout (`pps_cfg.max_segments`) | 4 |
| `S` | number of denoising steps (DDIM inference steps) | 5 |
| `H` | trajectory horizon (waypoints) | 8 |
| `L_p` | prompt length in tokens (left-padded, image tokens included) | ~1–2.8k |
| `L_r` | generated response length (`max_text_tokens`) | ≤ 512 |
| `D_vlm` | VLM hidden size (2B / 8B) | 1536 / 3584 |
| `s` | a scene | |
| `c` | one sampled CoT | |
| `ε` | one planner noise draw (x_T + per-step noise) | |
| `cost` | per-trajectory cost ∈ [0, 1+β] | |
| `Φ` | potential = −E_ε[cost] ∈ [−1−β, 0], where higher means safer | |

---

## 3. Stage 0: Prerequisites

These are produced once, by existing scripts.

| Step | Input | Process | Output |
|---|---|---|---|
| 0.1 Data | NAVSIM / OpenScene download (`download/*.sh`) | — | `navsim_logs/{trainval,test}`, `sensor_blobs/…`, `maps/` |
| 0.2 Metric cache | logs + maps | `scripts/cache_dataset/run_metric_caching*.sh` | `metric_cache_train/`, `metric_cache/` (navtest): one `.xz` pickle per scene token with the map, agents and PDM-Closed reference trajectory |
| 0.3 Base VLM | ReCogDrive SFT | — | `ReCogDrive-VLM-2B/` (InternVL, causal LM) |
| 0.4 Frozen planner | ReCogDrive IL (+RL) | — | `ReCogDrive_Diffusion_Planner_2B_RL.ckpt` (loaded into `agent.action_head`, all params frozen) |
| 0.5 High-risk split | PDM CSV of the base model on navtrain / navtest | `scripts/filter_high_risk_scenes/filter_high_risk_scenes.py` keeps scenes with NC < 1 or DAC < 1 | `high_risk_train.txt`, `high_risk_test.txt` (token lists) |

---

## 4. Stage 1: Diagnostic (run before training)

The diagnostic answers two questions: **does planner noise dominate the reward variance, and does the CoT actually steer the planner?**

Command: `bash scripts/mars/run_mars_diagnostic.sh` (code: `navsim/planning/script/run_mars_diagnostic.py`)

| | |
|---|---|
| **Input** | high-risk scenes; base VLM (or a trained LoRA via `VLM_LORA_PATH`); frozen planner; metric cache |
| **Process** | For every scene, run Step 2.2–2.4 below (G CoTs × K noises), with no training |
| **Output** | `mars_diagnostic/metrics_rank*.pt`: tensors `[N_scenes, G, K]` of NC, DAC, EP, TTC, comfort, score |

Then run `scripts/mars/summarize_mars_diagnostic.py`, which prints:

| Metric | Meaning | How to read it |
|---|---|---|
| `var_noise` | E_c[Var_ε(fail)]: variance caused by planner noise | high → NSR's reward is mostly noise |
| `var_reasoning` | Var_c(E_ε[fail]): variance caused by the CoT | **≈ 0 → CoT does not steer the planner; fix that first (e.g. `pass_cot_token_only`)** |
| `noise_fraction` | var_noise / total | the motivating number for the paper |
| `flip_rate` | share of CoTs whose safety depends on ε | "one token flips the trajectory", quantified |

Then run `bash scripts/pps/run_pps_eval.sh` on the base model, which prints the **reasoning gain** (Section 7). If it is negative, the CoT currently makes plans *less* safe than no CoT, which is a strong motivation for PPS.

---

## 5. Stage 2: One PPS training step, in detail

Entry point: `AgentLightningVLMRL.training_step` → `_step` → `_training_step_pps`
(`navsim/planning/training/agent_lightning_module.py`). Manual optimization: one optimizer step per batch.

```
batch ─► 2.1 unpack ─► 2.2 draw shared noise
           │
           └─► for g in 1..G:
                 2.3a generate CoT  ─► 2.3b re-encode (hidden states)
                 2.4  decode K plans + PDM  ─► metrics[:, g, :]
                 2.5  probe prefixes of failed rollouts ─► Φ curve      (hook)
        ─► 2.6 outcome advantage (MARS)
        ─► 2.7 per-token advantage (PPS)
        ─► 2.8 policy-gradient update (LoRA)
```

### 2.1 Unpack the batch
`agent.unpack_features_cot_prompt(features)`, `get_diff_input(...)`

| | |
|---|---|
| **Input** | `features` from `NegDriveFeatureBuilder`: `history_trajectory [B,4,3]` (x, y, heading for the past 4 frames), `high_command_one_hot [B,4]`, `status_feature [B,8]` (command + velocity + acceleration), `image_path_tensor` (front camera path as char codes); `tokens_list [B]` (scene ids) |
| **Process** | Load the front image and tile it into 448×448 patches (`load_image`). Build the **CoT prompt**: history, command, and the 4-stage reasoning instructions (Perception → Dynamic Understanding → Planning & Reasoning → Advanced Reasoning → `[PT, …]`) |
| **Output** | `pixel_values_cat [ΣN_patches,3,448,448]`, `questions [B]` (str), `num_patches_list [B]`, `diff_input` = {`his_traj [B,12]`, `status_feature [B,8]`} for the planner |

### 2.2 Draw shared planner noise (MARS, common random numbers)
`sample_planner_noise` in `utils/mars.py`

| | |
|---|---|
| **Input** | `B, K, S, H`, action dim 3 |
| **Process** | Sample once per scene and **reuse for all G CoTs** of that scene (if `common_random_numbers=true`) |
| **Output** | `noise["init"] [B,K,H,3]` (x_T), `noise["steps"] [B,K,S,H,3]` (per-step DDIM noise) |

Why per-step noise as well: the planner samples with η = 1, so it injects new noise at **every** denoising step. Fixing only x_T would not make the plan deterministic.

### 2.3a Generate a CoT (rollout g)
`vlm.generate_text_actions(...)`, sampling at temperature `rollout_temperature`

| | |
|---|---|
| **Input** | pixels, `questions` |
| **Output** | `NegDriveGenOutput`: `full_ids [B, L_p+L_r]` (prompt + response), `attention_mask [B, L_p+L_r]`, `response_start_idx = L_p` |

### 2.3b Re-encode prompt + CoT
`vlm.forward_with_ids(pixels, full_ids, attention_mask)`

| | |
|---|---|
| **Input** | `full_ids`, `attention_mask` |
| **Output** | `last_hidden_states [B, L_p+L_r, D_vlm]` (the final-layer states) |
| **Key property** | The LM is **causal**: position `L_p+t` only saw tokens `≤ L_p+t`. So `last_hidden_states[:, :L_p+t]` is exactly what the VLM would have produced for "prompt + first t CoT tokens". This makes PPS free. |

### 2.4 Decode K trajectories and score them
`action_head.get_action(..., init_actions=noise.init[:,k], step_noises=noise.steps[:,k])`, then `action_head.reward_pdm(...)`

| | |
|---|---|
| **Input** | `last_hidden_states`, `diff_input`, `attention_mask`, noise k |
| **Process (planner)** | `feature_encoder` (D_vlm → 384); masked mean-pool of the VLM tokens; 16-block DiT with cross-attention to all VLM tokens; 5 DDIM steps from x_T using the given per-step noise; `denorm_odo` |
| **Output (planner)** | `pred_traj [B,H,3]` (x, y, heading in the ego frame, 4 s @ 2 Hz) |
| **Process (PDM)** | Non-reactive LQR simulation against the metric cache; computes NC, DAC, EP, TTC, comfort |
| **Output (stored)** | `metrics[key][:, g, k]` for key ∈ {NC, DAC, EP, TTC, comfort, score}; the final tensors are `[B,G,K]` |

### 2.5 Probe CoT prefixes (the PPS part)
Hook `_make_pps_probe_hook` → `_probe_prefix_potentials`, called **right after 2.4 for rollout g**, while `last_hidden_states` is still in memory.

**(a) Pick which rollouts to probe**

| | |
|---|---|
| **Input** | `metrics[:, g, :]` |
| **Process** | `cost = fail + β·(1−fail)·relu(ep_min − EP)/ep_min` (`compute_mars_cost`); probe scene b if mean_k cost > 0 (`probe=failures`) or always (`probe=all`) |
| **Output** | set of (b, g) to probe |

**(b) Segment the CoT** (`find_stop_index`, `segment_response` in `utils/pps.py`)

| | |
|---|---|
| **Input** | response token ids `full_ids[b, L_p:]`, decoded token pieces |
| **Process** | `num_real` = tokens up to and including the first EOS / `<|im_end|>`. Boundaries are placed first at the stage headings `"Dynamic Understanding"`, `"Planning & Reasoning"`, `"Advanced Reasoning"`, `"[PT"`; remaining slots go to sentence/line ends that split the largest gaps; at most `P` segments |
| **Output** | `boundaries = [0, b₁, …, b_{P−1}, L_r]` in response-token indices; prefix j = response tokens `[0, b_j)` |

**(c) Decode each prefix under the same K noises**

| | |
|---|---|
| **Input** | for each boundary b_j (except the last): `last_hidden_states[b, :L_p+b_j]` and `attention_mask[b, :L_p+b_j]`, repeated K times; `noise.init[b]`, `noise.steps[b]` (the **same** draws the full rollout used) |
| **Process** | planner → `pred_traj [K,H,3]` → PDM → cost per draw → `Φ_j = −mean_k cost` |
| **Output** | `phi [P+1]`: Φ₀ = prompt only, …, Φ_P = full CoT (copied from 2.4, not recomputed) |

Why **truncate** rather than mask: the planner's cross-attention ignores the attention mask, so masked-out later tokens would still leak in. Truncating the sequence removes them.

**(d) Store**: `probes[(b,g)] = {boundaries, phi, num_real, num_total, segments (text)}`

### 2.6 Outcome advantage (MARS)
`compute_mars_advantages`

| | |
|---|---|
| **Input** | `cost [B,G,K]` |
| **Process** | `A(c) = −mean_k cost` (`baseline=none`, always ≤ 0, NSR-style) or minus the scene mean over G (`baseline=group_mean`) |
| **Output** | `advantages [B,G]` |

### 2.7 Per-token advantage (PPS)
`build_token_advantages`

| | |
|---|---|
| **Input** | `boundaries`, `phi [P+1]`, `A(c)`, λ (`shaping_coef`), `negative_only` |
| **Process** | segment deltas `d_j = Φ_{j+1} − Φ_j`; if `negative_only`, clamp to `d_j ≤ 0`; for tokens of segment j: `A_t = A(c) + λ·d_j`; tokens after the stop token get 0 |
| **Output** | probed rollout: `token_adv [L_r]` (tensor). Unprobed rollout: scalar `A(c)` (skipped if \|A\| ≈ 0) |
| | → `update_samples = [(b, g, adv), …]` where `adv` is a float or a `[L_r]` tensor |

### 2.8 Policy-gradient update
`_run_distributed_manual_updates` → `_compute_mean_response_log_prob` → `compute_response_logprobs_tokens`

| | |
|---|---|
| **Input** | `update_samples`, the stored `gen_output`s and pixels |
| **Process** | For each sample, re-run the VLM **with gradients** on (pixels, full_ids) and get the token log-probs `log π(c_t \| c_<t, s)` over the response. Loss: scalar adv → `−A · mean_t log π_t / N`; token adv → `−mean_t (A_t · log π_t) / N`. N = global count of update samples across GPUs. One `backward` per sample (to save memory); ranks with fewer samples run zero-loss dummy backwards so DDP stays in sync |
| **Output** | gradients on the LoRA parameters only (vision encoder, projector and planner are frozen) → `clip_grad_norm_(1.0)` → `opt.step()`, `sch.step()` (in `training_step`) |

---

## 6. Stage 3: Validation

`validation_step` is unchanged and shared by all algorithms.

| | |
|---|---|
| **Input** | `high_risk_test` scenes (navtest), val metric cache |
| **Process** | greedy CoT (`val_do_sample=false`) → planner (default stochastic sampling) → PDM |
| **Output** | `val/pdm_score`, `val/no_at_fault_collisions`, `val/drivable_area_compliance`, `val/ego_progress`, … |

---

## 7. Stage 4: Evaluation

Command: `bash scripts/pps/run_pps_eval.sh` (code: `run_pps_eval.py`, summary: `scripts/pps/summarize_pps_eval.py`)

| | |
|---|---|
| **Input** | any policy (`VLM_LORA_PATH` empty = base model, or the NSR / MARS / PPS LoRA) |
| **Process** | Stage 2.1–2.5 with `probe=all`, no update |
| **Output** | `pps_eval/probes_rank*.pt` (per rollout: `phi`, `boundaries`, segment texts, metrics) and `summary.json` |

Reported numbers:

| Metric | Formula | Meaning |
|---|---|---|
| `phi_prompt` | mean Φ₀ | plan quality with the prompt only (no reasoning yet) |
| `phi_full` | mean Φ_P | plan quality with the full CoT |
| **`reasoning_gain`** | mean (Φ_P − Φ₀) | > 0: reasoning makes plans safer. PPS should push this up |
| `harmful_cot_rate` | P(Φ_P < Φ₀) | share of CoTs that make the plan worse |
| `blame_seg{j}` | mean min(d_j, 0) | which reasoning stage causes failures |
| `most_blamed` | top-k segment texts | qualitative examples for the paper |

For the main results table, also run the usual PDMS evaluation (`scripts/evaluation/test_negdrive_*.sh`) on full navtest and `high_risk_test`.

---

## 8. Worked example

One scene, one failed rollout, K = 3, β = 0 (safety only), λ = 1, `negative_only = true`.

**CoT (12 response tokens) and boundaries found in 2.5(b):**

| Segment j | Tokens | Text |
|---|---|---|
| 0 | 0–2 | "Foundational: car ahead." |
| 1 | 3–5 | " Dynamic Understanding: brakes." |
| 2 | 6–7 | " Planning & Reasoning: keep speed." |
| 3 | 8–11 | " Advanced Reasoning: ok. [PT …]" + EOS |

`boundaries = [0, 3, 6, 8, 12]`

**Prefix plans under the same 3 noises (2.5c):**

| Prefix | fail under ε₁ ε₂ ε₃ | Φ = −mean |
|---|---|---|
| prompt only | 0 0 0 | 0.00 |
| + seg 0 | 0 0 0 | 0.00 |
| + seg 1 | 0 0 0 | 0.00 |
| + seg 2 | 1 1 0 | −0.67 |
| full CoT | 1 1 0 | −0.67 |

**Deltas:** d = [0, 0, −0.67, 0]
**Outcome advantage (2.6):** A(c) = −0.67
**Per-token advantage (2.7):**

```
token:  0     1     2     3     4     5     6     7     8     9     10    11
A_t:  -0.67 -0.67 -0.67 -0.67 -0.67 -0.67 -1.33 -1.33 -0.67 -0.67 -0.67 -0.67
                                            └─ "keep speed" gets double blame
```

Under NSR, all 12 tokens would get −1 if this one noise draw failed, or 0 if it happened to draw ε₃.

---

## 9. How NSR / MARS / PPS differ

| | NSR | MARS | PPS |
|---|---|---|---|
| Planner draws per CoT | 1 (random) | K (shared per scene) | K (shared) + K per prefix |
| Reward signal | NC/DAC binary | E_ε[fail + β·progress hinge] | same as MARS |
| Advantage granularity | sequence | sequence | **token (per segment)** |
| Advantage value | −1 on failure | −E_ε[cost] | −E_ε[cost] + λ·min(ΔΦ, 0) |
| Handles planner noise | no | yes | yes |
| Handles "stopping is safe" | no | yes (β > 0) | yes |
| Locates the bad reasoning step | no | no | **yes** |
| Extra PDM sims per step (vs NSR) | 1× | K× | K× + (#failures · P · K) |
| `rl_algorithm` | `nsr` | `mars` | `pps` |

`pps` with `shaping_coef=0` is exactly `mars`; `mars` with `K=1, common_random_numbers=false, progress_coef=0` behaves like `nsr`.

---

## 10. Configuration reference

`navsim/planning/script/config/common/agent/negdrive_agent.yaml`

```yaml
rl_algorithm: pps

mars_cfg:                       # used by both mars and pps
  num_planner_samples: 4        # K
  common_random_numbers: true   # same K noises for all G CoTs of a scene
  progress_coef: 0.5            # β; 0 = pure safety cost
  progress_min: 0.6             # safe samples with EP < this are penalized
  baseline: none                # none (pure negative) | group_mean
  advantage_eps: 1.0e-6

pps_cfg:
  max_segments: 4               # P
  shaping_coef: 1.0             # λ
  negative_only: true           # only blame risk-increasing segments
  probe: failures               # failures | all
  use_stage_markers: true       # split at the prompt's stage headings
  advantage_eps: 1.0e-6
```

---

## 11. Logged metrics

All are logged under `train/` in SwanLab.

| Key | Meaning |
|---|---|
| `nc_fail_rate`, `dac_fail_rate` | over all G·K samples |
| `mars_cost`, `mars_ego_progress`, `mars_pdm_score` | batch means |
| `pps_num_probed` | failed rollouts that got prefix-probed |
| `pps_num_token_level` / `pps_num_updates` | token-level updates / all updates this step |
| `pps_phi_prompt`, `pps_phi_full`, `pps_reasoning_gain` | over probed rollouts |
| `pps_harmful_cot_rate`, `pps_neg_segment_rate` | share of CoTs / segments that increased risk |
| `pps_blame_seg{j}` | average blame at segment position j |
| `pps_pg_loss`, `grad_norm` | optimization health |

**What to watch:**
- `pps_reasoning_gain` should rise.
- `mars_ego_progress` should not collapse.
- If `pps_neg_segment_rate` stays ≈ 0, the prefixes do not change the plan; see Section 13.

---

## 12. Commands

```bash
# 0. smoke test (1 GPU, tiny)
FAST_DEV_RUN=true GPUS=1 G=2 MARS_K=2 bash scripts/pps/train_pps.sh

# 1. diagnostics on the base model
bash scripts/mars/run_mars_diagnostic.sh          # noise vs reasoning variance
bash scripts/pps/run_pps_eval.sh                  # reasoning gain / blame

# 2. training
bash scripts/pps/train_pps.sh                     # PPS (main method)
bash scripts/mars/train_mars.sh                   # MARS baseline
RL_ALGORITHM=nsr bash scripts/training/train_negdrive.sh   # NSR baseline (edit RL_ALGORITHM in the file)

# 3. ablations
bash scripts/pps/run_ablations.sh                 # λ, signed, probe-all, markers, P
bash scripts/mars/run_ablations.sh                # K, CRN, β, baseline

# 4. evaluate a trained policy
VLM_LORA_PATH=/path/to/checkpoints/<lora> bash scripts/pps/run_pps_eval.sh
```

Requirements: `SWANLAB_API_KEY` in the environment (or `SWANLAB_MODE=offline`), plus the paths at the top of each script (`NAVSIM_*`, `MODEL_ROOT`, metric caches).

---

## 13. Known limitations

1. **Policy invariance only for signed shaping.** With `negative_only=false`, the segment deltas telescope to Φ_P − Φ₀, so the shaping does not change the optimal policy (potential-based shaping). The default `negative_only=true` drops positive deltas; it is a deliberate bias toward blame and should be reported as such and compared in the ablation.
2. **Prefixes are out of distribution for the planner.** It was trained on hidden states of the prompt only (and was never shown partial CoTs). Check that Φ correlates with real outcomes using `run_pps_eval.py` before trusting the blame scores.
3. **Φ₀ is not the original ReCogDrive plan.** It uses the CoT prompt with no CoT generated yet, not ReCogDrive's planning prompt. Evaluate `compute_trajectory_recogdrive` separately for a true no-reasoning baseline.
4. **CoT signal dilution.** The planner mean-pools ~L_p + L_r tokens, most of them image tokens. If `var_reasoning` (Stage 1) or `pps_neg_segment_rate` is near zero, the CoT barely moves the plan; try `pass_cot_token_only`.
5. **Cost is CPU-bound.** PDM simulation dominates: B·G·K + (#failures)·P·K simulations per step. Lower `K`/`P`, keep `probe=failures`, or parallelize `reward_pdm` with a process pool.
6. **Not yet run on GPU.** The logic was tested on CPU with mock VLM/planner/PDM. Run the smoke test first.
