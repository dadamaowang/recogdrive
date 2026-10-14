# PPS-MC: Planner-Verified Monte Carlo Segment Credit for Reasoning VLAs

The algorithm implemented on branch `dev-neg_v2`.

It trains the reasoning (CoT) of a driving VLM against a **frozen diffusion planner**. Credit goes to the **reasoning segment** that made the plan unsafe, estimated by Monte Carlo continuation and robust to the planner's own sampling noise.

It combines three pieces, each switchable for ablation:

| Piece | Fixes | Switch |
|---|---|---|
| **CoT-only conditioning** | The planner barely uses the CoT when it reads image + prompt + CoT | `agent.pass_cot_token_only=true` |
| **MARS**: noise-marginalized outcome | One planner noise draw decides the reward for a whole CoT | `agent.mars_cfg.*` |
| **PPS-MC**: segment credit by Monte Carlo continuation | Every token of a failed CoT is blamed equally | `agent.rl_algorithm=pps`, `agent.pps_cfg.estimator=mc` |

Related docs:
- `docs/PPS_PIPELINE.md`: full step-by-step pipeline, including the cheaper `truncate` estimator.
- `docs/PPS_PILOT.md`: go / no-go pilot design.

---

## 1. Motivation

### 1.1 Setting
- The **policy** is the ReCogDrive InternVL VLM with LoRA. Given the front image, past trajectory and navigation command, it writes a CoT `c` with four stages: Perception → Dynamic Understanding → Planning & Reasoning → Advanced Reasoning → `[PT …]`.
- The **frozen planner** is ReCogDrive's DiT diffusion planner. It maps VLM hidden states to an 8-waypoint trajectory with stochastic DDIM sampling (η = 1, 5 steps).
- The **reward** comes from PDM (NAVSIM's simulator-based scorer): a safety failure is NC < 1 or DAC < 1, plus ego progress.
- **Training data** is the high-risk navtrain scenes (1,163), where base ReCogDrive fails.

### 1.2 Three problems with NSR (the previous method)
1. **The planner barely reads the CoT.** Measured on full navtest (`prompt_v1_pre/`):

   | | PDMS | NC | DAC | EP | Failure rate |
   |---|---|---|---|---|---|
   | 2B, full sequence | 48.4 | 80.0 | 70.3 | 46.6 | 45.5% |
   | 2B, **CoT-only** | **62.1** | 92.6 | 80.7 | 52.3 | **26.1%** |
   | 8B, full sequence | 52.6 | 80.4 | 76.7 | 49.4 | 40.0% |
   | 8B, **CoT-only** | **64.0** | 90.9 | 86.3 | 51.1 | **22.1%** |

   - CoT-only beats the full sequence by +13.7 (2B) and +11.4 (8B) PDMS, paired 95% CI [+12.9, +14.5] and [+10.6, +12.2].
   - NSR trained the CoT through the weaker path.
   - Caveat: the 8B full-sequence run is labelled `promptv2`; the other three are `promptv1`.
2. **Planner noise corrupts the outcome.** The same CoT can be safe under one noise draw and unsafe under another. NSR assigns −1 or 0 from a single draw.
3. **No credit inside the CoT.** One sentence (e.g. "keep speed") can make the plan unsafe, but NSR spreads the same penalty over all ~100–500 tokens.

### 1.3 Idea
- Fix (1) by conditioning the planner on CoT tokens only.
- Fix (2) by scoring each CoT under **K shared planner noises** (MARS).
- Fix (3) borrowing from text-LLM RL: estimate the **value of each CoT prefix** by letting the policy **finish** the CoT M times and scoring the completions with the planner + PDM (VinePPO / SPO-chain). A segment's credit is the change in value it causes.
- Keep NSR's identity: **negative-only** step credit. Segments are blamed, never praised.

---

## 2. Notation

| Symbol | Meaning | Default |
|---|---|---|
| s | scene (image, history, command, metric cache) | |
| π_θ | VLM policy (LoRA parameters θ) | |
| c = (c₁…c_L) | generated CoT tokens | L ≤ 512 |
| D(h, ε) | frozen planner: hidden states h, noise ε → trajectory τ | |
| h(c) | planner conditioning: hidden states of the CoT tokens only | |
| ε₁…ε_K | planner noises (x_T + per-step DDIM noise), shared by all CoTs of a scene | K = 4 |
| G | CoTs sampled per scene | 4 |
| P | CoT segments per rollout | 4 |
| 0 = b₀ < b₁ < … < b_P = L | segment boundaries | stage headings |
| M | Monte Carlo continuations per boundary | 2 |
| β, e_min | progress-hinge weight / threshold | 0.5 / 0.6 |
| λ | segment-credit weight | 1.0 |

**Per-trajectory cost** (`utils/mars.py: compute_mars_cost`):

```
fail(τ) = 1[NC(τ) < 1  or  DAC(τ) < 1]
cost(τ) = fail(τ) + β · (1 − fail(τ)) · relu(e_min − EP(τ)) / e_min        ∈ [0, 1+β]
```

The hinge penalizes "safe by stopping". PDM sets EP = 1 when the reference plan barely moves, so scenes that really require stopping are not penalized.

**Value of a CoT** (MARS, averaged over planner noise):

```
R(c) = −(1/K) Σ_k cost(D(h(c), ε_k))
```

**Value of a CoT prefix** (Monte Carlo continuation):

```
V(c<b) = E_{c' ~ π_θ(·| s, c<b)} [ R(c<b ⊕ c') ]
       ≈ (1/M) Σ_m R(c<b ⊕ c'_m)            with c'_m ~ π_θ(·| s, c<b)
```

---

## 3. Algorithm

### 3.1 Advantages

For a rollout c with boundaries b₀…b_P:

```
Φ_0 = V(empty prefix)  ≈ mean over the scene's G rollouts of R(c^(g))
Φ_j = V(c<b_j)         ≈ MC estimate,  j = 1 … P−1
Φ_P = R(c)             (the rollout's own value)

d_j = Φ_{j+1} − Φ_j                                  segment j's effect on the value
A_MARS(c) = R(c)                                     (baseline=none; ≤ 0)
A_t = A_MARS(c) + λ · min(d_j, 0)     for token t in segment j, t < stop token
A_t = 0                               for padding after the stop token
```

- **Unprobed rollouts** (`probe=failures` and cost = 0) keep the sequence-level `A_MARS`. If that is 0 they are skipped.
- **Telescoping:** with signed credit (`negative_only=false`), Σ_j d_j = R(c) − Φ₀ = R(c) − mean_g R. That is exactly a group-relative outcome advantage (GRPO without std normalization), so signed PPS-MC redistributes a GRPO-like advantage over segments. The default negative-only clamp breaks this identity on purpose (Section 6).

### 3.2 Objective

REINFORCE with token-level advantages, one sample at a time:

```
L(θ) = − (1/N) Σ_{(s,c) ∈ updates}  (1/|c|) Σ_t  A_t · log π_θ(c_t | s, c<t)
```

N is the global number of update samples across GPUs. The log-probs are **image-conditioned**: the policy sees the image, only the planner is CoT-only. There is no KL or ratio clipping, as in NSR.

### 3.3 Pseudocode

```
for each batch of scenes s_1..s_B:
    draw shared noises ε[b, 1..K]                       # once per scene (CRN)
    for g in 1..G:
        c[b,g]  ~ π_θ(· | s_b)                           # image-conditioned generation, T = 0.7
        h       = LM(c[b,g] tokens only)                 # CoT-only planner view
        cost[b,g,k] = cost(D(h, ε[b,k]))  for k = 1..K   # PDM
        if Σ_k cost[b,g,k] > 0:                          # probe failed rollouts
            boundaries = segment(c[b,g])                 # stage headings, else sentence ends
            for each inner boundary b_j, for m in 1..M:  # one generation batch
                c' ~ π_θ(· | s_b, c[b,g]<b_j)
                cost_j,m,k = cost(D(LM(c[b,g]<b_j ⊕ c'), ε[b,k]))
            Φ_j = −mean_{m,k} cost_j,m,k
            Φ_P = −mean_k cost[b,g,k]
    Φ_0(b) = −mean_{g,k} cost[b,g,k]                     # after all G rollouts
    build A_t per probed rollout, A_MARS per unprobed rollout
    policy-gradient step on LoRA (one backward per sample, DDP-padded)
```

---

## 4. Step-by-step: inputs and outputs

All code is in `navsim/planning/training/agent_lightning_module.py` unless noted. Shapes are per GPU.

| # | Step | Code | Input | Output |
|---|---|---|---|---|
| 1 | Unpack batch | `agent.unpack_features_cot_prompt`, `get_diff_input` | `history_trajectory [B,4,3]`, `status_feature [B,8]`, `high_command_one_hot [B,4]`, front-camera path, `tokens_list [B]` | `pixel_values_cat [ΣN,3,448,448]`, CoT prompt `questions [B]`, `num_patches_list [B]`, `diff_input {his_traj [B,12], status_feature [B,8]}` |
| 2 | Shared noise | `utils/mars.py: sample_planner_noise` | B, K, 5 steps, H=8 | `init [B,K,8,3]`, `steps [B,K,5,8,3]` |
| 3 | Generate CoT g | `vlm.generate_text_actions` | pixels, prompts, T = 0.7, ≤ 512 tokens | `full_ids [B, L_p+L_r]`, `attention_mask`, `response_start_idx = L_p` |
| 4 | Planner view | `_planner_view` → `_response_view` → `vlm.forward_response_only` (`negdrive_backbone.py`) | response ids, cut after the first stop token | `hidden [B, L_r, D_vlm]`, `mask [B, L_r]` (right-padded), `offset = 0` |
| 5 | Decode × K + score | `action_head.get_action(…, init_actions, step_noises)` + `reward_pdm` | hidden, mask, noise k | `pred_traj [B,8,3]` → `metrics[key][B,G,K]` (NC, DAC, EP, TTC, comfort, score) |
| 6 | Select probes | `_make_pps_probe_hook` | `cost[:, g, :]` | rollouts with mean_k cost > 0 (or all, with `probe=all`) |
| 7 | Segment | `_segment_rollout` → `utils/pps.py: segment_response` | response tokens | `boundaries [P+1]`, `num_real`, segment texts |
| 8 | MC continuations | `_probe_prefix_values_mc` → `vlm.generate_from_ids` (`negdrive_backbone.py`) | (P−1)·M rows of prompt + c<b_j, left-padded; image repeated | new tokens `[(P−1)M, ≤ 512 − b₁]` |
| 9 | Score completions | `_planner_view` + `score_hidden_states` | each completed CoT (prefix ⊕ continuation) and the scene's K noises | `costs[j] [M·K]` → `Φ_j` |
| 10 | Empty-prefix value | `collect_rollouts_with_probes` | `cost [B,G,K]` after all G rollouts | `Φ₀(b) = −mean_{g,k}` → `phi [P+1]` per probe |
| 11 | Advantages | `utils/mars.py: compute_mars_advantages`, `utils/pps.py: build_token_advantages` | `cost`, `phi`, boundaries, λ | float `A_MARS` or tensor `A_t [L_r]` per (b, g) |
| 12 | Update | `_run_distributed_manual_updates` → `_compute_mean_response_log_prob(token_weights=A_t)` | (b, g, A) samples, stored `gen_output`s, pixels | gradients on LoRA → grad clip 1.0 → AdamW step |

**Planner masking.** CoT-only batches are right-padded. With CoT-only on, the DiT's cross-attention ignores padded keys:
- `negdrive_dit.py`: `encoder_key_padding_mask`
- turned on by `action_head.mask_cross_attention = pass_cot_token_only`
- the boolean mask is handled in `blocks/attention.py`

With the full sequence, the original (unmasked) behaviour is unchanged.

---

## 5. Cost

Per training step, per GPU, with p = probed (failed) rollouts:

| Item | NSR | MARS | PPS-truncate | **PPS-MC** |
|---|---|---|---|---|
| VLM generations | B·G | B·G | B·G | B·G + p·(P−1)·M (batched per rollout) |
| VLM forwards for the planner | B·G (full seq) | B·G | B·G | B·G + p (text-only, cheap) |
| PDM simulations (CPU) | B·G | B·G·K | B·G·K + p·(P−1)·K | B·G·K + **p·(P−1)·M·K** |
| Backward passes | #failures | #updates | #updates | #updates |

With B=8, G=4, K=4, P=4, M=2 and ~40% failures (p ≈ 13), PDM goes from 32 (NSR) to 128 (MARS) to **~440** (PPS-MC) simulations per step. **CPU PDM is the bottleneck**; parallelize `reward_pdm` or lower M / P if steps are slow.

---

## 6. Design choices and their justification

| Choice | Why | Ablation |
|---|---|---|
| CoT-only conditioning | +11–14 PDMS on navtest. The planner's output becomes a function of the text, so segment credit is meaningful | `COT_ONLY=false` |
| Shared noise across CoTs (common random numbers) | Differences between CoTs of a scene are not confounded by planner noise; continuations reuse the same noises | `MARS_CRN=false` |
| K noise draws per CoT | Unbiased, lower-variance outcome (Rao–Blackwellized over ε) | `MARS_K ∈ {1,2,4,8}` |
| MC continuation instead of truncation | The planner only reads **complete** CoTs (no out-of-distribution prefixes); V is an unbiased value of the *current* policy (VinePPO) | `PPS_ESTIMATOR=truncate` |
| Stage-heading segmentation | Segments align with the prompt's reasoning stages and are interpretable | `PPS_MARKERS=false`, `PPS_SEGMENTS ∈ {2,6}` |
| Negative-only credit | Keeps NSR's "learn from failures" identity and avoids rewarding segments of a CoT that failed anyway. **Biased:** it gives up the telescoping property | `PPS_NEG_ONLY=false` (signed, telescopes to a GRPO-like advantage) |
| Φ₀ = scene mean over G×K | Value of the empty prefix under π_θ at no extra cost | — |
| Probe failures only | Cost is proportional to the failure count; safe CoTs need no blame | `PPS_PROBE=all` |
| λ = 1 | Segment credit has the same scale as the outcome advantage | `PPS_LAMBDA ∈ {0, 0.5, 1, 2}` (λ = 0 equals MARS) |

---

## 7. Configuration

`navsim/planning/script/config/common/agent/negdrive_agent.yaml`

```yaml
rl_algorithm: pps
pass_cot_token_only: true

mars_cfg:
  num_planner_samples: 4        # K
  common_random_numbers: true
  progress_coef: 0.5            # β
  progress_min: 0.6             # e_min
  baseline: none                # none | group_mean
  advantage_eps: 1.0e-6

pps_cfg:
  estimator: mc                 # mc | truncate
  mc_continuations: 2           # M
  max_segments: 4               # P
  shaping_coef: 1.0             # λ
  negative_only: true
  probe: failures               # failures | all
  use_stage_markers: true
  advantage_eps: 1.0e-6
```

The YAML default for `estimator` is `truncate`. Pass `PPS_ESTIMATOR=mc` in the scripts, or `agent.pps_cfg.estimator=mc`.

---

## 8. Running

```bash
# 0. smoke test (1 GPU, 4 scenes)
MAX_SCENES=4 GPUS=1 PPS_ESTIMATOR=mc bash scripts/pps/pilot/phaseA_signal_check.sh

# 1. pilot, phase A: does the signal exist? (no training; see docs/PPS_PILOT.md)
PPS_ESTIMATOR=mc bash scripts/pps/pilot/phaseA_signal_check.sh
PPS_ESTIMATOR=truncate bash scripts/pps/pilot/phaseA_signal_check.sh

# 2. pilot, phase B: short NSR vs MARS vs PPS comparison
PPS_ESTIMATOR=mc bash scripts/pps/pilot/phaseB_short_training.sh

# 3. full training
PPS_ESTIMATOR=mc bash scripts/pps/train_pps.sh
bash scripts/mars/train_mars.sh                       # MARS baseline (CoT-only)

# 4. ablations
bash scripts/pps/run_ablations.sh                     # λ, signed, probe-all, markers, P, mc, full-seq
bash scripts/mars/run_ablations.sh                    # K, CRN, β, baseline

# 5. analysis of a trained policy (reasoning gain, blame per stage, blamed texts)
VLM_LORA_PATH=<.../iter_N_lora> bash scripts/pps/run_pps_eval.sh
```

**Fair NSR baseline.** `scripts/training/train_negdrive.sh` still uses full-sequence conditioning. Add `agent.pass_cot_token_only=true` so the +11–14 PDMS from the conditioning path is not credited to the algorithm.

---

## 9. What to monitor

SwanLab project `NegDriveComplete`.

| Group | Keys | Expected |
|---|---|---|
| Outcome | `val/pdm_score`, `val/no_at_fault_collisions`, `val/drivable_area_compliance` | PPS-MC > MARS > NSR |
| Stop-collapse guard | `val/ego_progress`, `train/mars_ego_progress` | no drop > 0.05 from step 0 |
| Mechanism | `train/pps_reasoning_gain` (Φ_P − Φ₀), `train/pps_harmful_cot_rate`, `train/pps_blame_seg{j}` | gain ↑, harmful rate ↓, the most-blamed stage shrinks |
| Signal volume | `train/pps_num_probed`, `train/pps_num_token_level`, `train/pps_num_updates` | non-zero every step |
| Health | `train/grad_norm`, `train/pps_pg_loss` | no spikes or NaN |
| Degeneration | sampled CoT text in logs | CoTs keep their stage headings and don't shrink |

With `estimator=mc`, `pps_reasoning_gain` is R(c) − mean_g R. It is a within-scene comparison, not "reasoning vs no reasoning". For the latter, use `estimator=truncate` evaluation together with full-sequence conditioning (Φ₀ = prompt only).

---

## 10. Limitations and open questions

1. **No no-CoT reference yet.** CoT-conditioned planning (22–45% failure on navtest) is far worse than ReCogDrive without CoT; base ReCogDrive fails ~4.5% of navtest, if `high_risk_test` came from that run. The paper must report the no-CoT ReCogDrive score and show the trained CoT policy closes or beats that gap, at least on high-risk scenes.
2. **Negative-only credit is biased.** Report signed credit as an ablation; it is the theoretically clean variant.
3. **MC variance.** M = 2 continuations × K = 4 noises gives 8 samples per value; with binary failures, values have a resolution of 1/8. If the pilot's reliability check (S3) is weak, raise M before raising K.
4. **Continuation length.** Continuations get a token budget of `max_text_tokens − b₁` (b₁ = shortest prefix). Truncated continuations are scored as-is.
5. **Planner trained on prompt-only states.** Even CoT-only hidden states are out of distribution for the planner. CoT-only is just less so (Section 1.2). Fine-tuning the planner is out of scope by design (frozen decoder).
6. **Compute.** CPU-bound PDM (Section 5).
7. **Only CPU mock-tested so far.** Verify that batched, masked CoT-only decoding matches the batch-size-1 evaluation used in `prompt_v1_pre` before trusting training numbers.

---

## 11. Related work this builds on

| Work | What we take | Link |
|---|---|---|
| VinePPO (Kazemnejad et al., ICML 2025) | MC value of intermediate states by re-sampling continuations | [code](https://github.com/McGill-NLP/VinePPO) · [paper](https://arxiv.org/abs/2410.01679) |
| SPO (Guo et al., 2025) | Segment-level MC advantages; SPO-chain | [code](https://github.com/AIFrameResearch/SPO) · [paper](https://arxiv.org/abs/2505.23564) |
| Rewarding Progress / PAV (Setlur et al., ICLR 2025) | A step's reward is its *progress* (change in success probability) | [paper](https://arxiv.org/abs/2410.08146) |
| TreeRPO (Yang et al., 2025) | Sibling-group step advantages; future variant: groups over continuations | [paper](https://arxiv.org/abs/2506.05183) |
| High-entropy forking tokens (Wang et al., NeurIPS 2025) | Orthogonal add-on: restrict updates to high-entropy tokens | [paper](https://arxiv.org/abs/2506.01939) |
| Potential-based shaping (Ng et al., 1999) | Signed segment credit telescopes, so the optimal policy is unchanged | — |
| Credit assignment survey (2026) | Positioning and taxonomy | [paper](https://arxiv.org/abs/2604.09459) · [list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL) |
