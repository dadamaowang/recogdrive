# PPS Pilot: Is This Direction Worth a Full Study?

The goal is to decide **GO / NO-GO for PPS** with about one GPU-day, before spending weeks on full training runs and ablations.

The pilot has two phases. **Phase A costs no training** and answers most of the question. Phase B runs only if Phase A passes.

| Phase | What it tests | Training? | Cost (8×GPU) | Script |
|---|---|---|---|---|
| A: Signal check | Does the step-level signal PPS needs exist, is it reliable, and is it causal? | No | ~2–4 h | `scripts/pps/pilot/phaseA_signal_check.sh` |
| B: Short head-to-head | Does the signal translate into better learning than NSR / MARS? | 3 × 120 steps | ~1 day | `scripts/pps/pilot/phaseB_short_training.sh` |

---

## Phase A: Signal check (no training)

### What it does

On 200 random high-risk **train** scenes, with the **base** VLM and the frozen planner:

1. Generate **G = 4** CoTs per scene and decode each under **K = 4** shared planner noises.
2. For **every** rollout, decode the CoT prefixes at **P = 4** segment boundaries to get the curve `Φ₀ (prompt) … Φ₄ (full)`.
3. **Reliability control:** re-score the full CoT under 4 *fresh* noises to get `Φ_ref`.
4. **Causal control:** for rollouts with a blamed segment, remove that segment's hidden states and re-score under the same noises (`Φ_drop_blame`). Do the same with a *random other* segment (`Φ_drop_random`).

Code: `navsim/planning/script/run_pps_pilot.py`. Summary: `scripts/pps/pilot/summarize_pilot.py`.

### Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| scenes | 200 (`MAX_SCENES`), random from `high_risk_train` (1163) | ~800 rollouts and ~400 failures, enough for a bootstrap CI |
| policy | base `ReCogDrive-VLM-2B`, no LoRA (`agent.mode=eval`) | PPS must have signal before any training |
| `G` (`per_sample_rollout`) | 4 | same as training |
| `K` (`mars_cfg.num_planner_samples`) | 4 | same as training; Φ resolution is 0.25 |
| `common_random_numbers` | true | prefixes and controls share noise with the full rollout |
| `progress_coef` (β) | **0.0** | pilot measures *safety* blame only, which is cleaner to interpret |
| `P` (`pps_cfg.max_segments`) | 4 | matches the prompt's 4 reasoning stages |
| `use_stage_markers` | true | |
| `rollout_temperature` | 0.7 | same as training |
| `max_text_tokens` | 512 | |
| batch size | 2 per GPU | probing every rollout is memory-light but PDM-heavy |
| seed | 42 (+rank) | |

### Signals to supervise

`summarize_pilot.py` prints these and a verdict.

| # | Signal | Definition | Threshold | If it fails |
|---|---|---|---|---|
| **S1** | **Steer rate** | P(Φ_full ≠ Φ_prompt): does the CoT change the plan at all? | ≥ 0.20 | **Hard stop.** The planner ignores the CoT (signal diluted by ~2.8k image/prompt tokens in the mean-pool). Try `pass_cot_token_only` and re-run before anything else. |
| **S2a** | **Localizable rate** | P(some segment has ΔΦ < 0 \| rollout failed) | ≥ 0.30 | **Hard stop.** Failures are already present in the prompt-only plan; the CoT is not what causes them, so PPS has nothing to blame. Check `failed_but_prompt_safe_rate`. |
| S2b | Concentration | share of negative ΔΦ mass on the single worst segment | ≥ 0.60 | Blame is spread out, so step credit ≈ sequence credit. Weak but OK; try P = 6. |
| S3a | Φ reliability | Spearman(Φ_full, Φ_ref) under independent noises | ≥ 0.50 | Φ is too noisy at K = 4. Re-run with `MARS_K=8`. |
| S3b | Signal-to-noise | mean \|blame\| / mean \|Φ_full − Φ_ref\| | ≥ 1.5 | Blame is within noise. Increase K. |
| **S4** | **Causal gap** | E[Φ_drop_blame − Φ_full] − E[Φ_drop_random − Φ_full], with 95% bootstrap CI | CI lower bound > 0 | **Hard stop.** PPS blame does not identify the harmful text better than chance. |

Diagnostic extras, which have no threshold but go into the paper's motivation:
- `reasoning_gain`, `harmful_cot_rate`: does the CoT help or hurt on average?
- `prompt_fail_rate`, `failed_but_prompt_safe_rate`: how often a safe prompt-only plan is *broken* by reasoning.
- `blame_by_segment`, `argmax_blame_hist`: which stage (perception / dynamics / planning / counterfactual) is blamed.
- `S4_fix_rate`: share of failures fixed outright by deleting the blamed segment.

### Verdict rule

| Verdict | Condition | Action |
|---|---|---|
| **GO** | all 6 checks pass | Run Phase B |
| **GO (weak)** | S1, S2a, S4 pass; S2b/S3 fail | Re-run Phase A with `MARS_K=8` and/or `PPS_SEGMENTS=6`, then Phase B |
| **NO-GO** | S1, S2a or S4 fails | Stop PPS. If S1 failed, fix CoT conditioning first; that fix is a finding in itself. If S2a/S4 failed, fall back to MARS or the grounded-negatives idea |

### Command

```bash
bash scripts/pps/pilot/phaseA_signal_check.sh
# smaller / faster:
MAX_SCENES=100 GPUS=4 bash scripts/pps/pilot/phaseA_signal_check.sh
# if S3 is weak:
MARS_K=8 bash scripts/pps/pilot/phaseA_signal_check.sh
```

Output: `$NAVSIM_EXP_ROOT/pps_pilotA_*/<uid>/pps_pilot/{pilot_rank*.pt, summary.json}` plus the printed table.

Cost estimate: 800 rollouts × (4 full + 3 prefix + 4 ref + 8 ablation) ≈ **15k PDM simulations** + 800 VLM generations. PDM on CPU dominates the run time.

---

## Phase B: Short head-to-head training (only after GO)

### What it does

It trains three arms for 120 steps with **identical** rollouts, compute and seed. Only the advantage computation differs:

| Arm | Setting | Isolates |
|---|---|---|
| `nsr` | `rl_algorithm=nsr`, G = 4, 1 planner draw | your current method (baseline) |
| `mars` | `rl_algorithm=pps`, `shaping_coef=0` (exactly MARS), K = 4 | effect of noise marginalization + progress hinge |
| `pps` | `rl_algorithm=pps`, `shaping_coef=1`, K = 4, P = 4 | added effect of step-level shaping |

The `pps − mars` difference is the contribution of the new idea. `mars − nsr` is the contribution of the MARS fixes.

### Hyperparameters

| Parameter | Value |
|---|---|
| train scenes | `high_risk_train` (1163) |
| val scenes | full `high_risk_test` (546 navtest scenes); `VAL_BATCHES=1.0` |
| steps | 120 (`STEPS`), about 1 epoch at batch 8 × 8 GPUs |
| val interval | every 20 steps (`VAL_EVERY`), plus baseline validation at step 0 (on the `nsr` arm only) |
| batch size | 8 per GPU |
| G | 4 |
| K | 4 (mars, pps) |
| β / ep_min | 0.5 / 0.6 |
| baseline | none |
| P / λ / negative_only / probe | 4 / 1.0 / true / failures |
| LR | 4e-5, AdamW, wd 0.01 |
| LoRA | r = 16, α = 16, dropout 0.05 |
| precision | bf16-mixed |
| seed | 42 (repeat with `SEED=43` if the result is borderline) |

### Signals to supervise (SwanLab, project `NegDriveComplete`)

**Primary (decides GO):**

| Signal | Key | Success criterion for PPS vs MARS and NSR |
|---|---|---|
| Safety on held-out high-risk scenes | `val/no_at_fault_collisions`, `val/drivable_area_compliance` | PPS ≥ MARS and > NSR at the best val step |
| Overall quality | `val/pdm_score` | PPS > MARS by ≥ 1 point (×100) at step 120, and better area under the val curve |
| No stop-collapse | `val/ego_progress` | PPS and MARS do not fall below the step-0 value by more than 0.05 (NSR is expected to drop) |

**Mechanism (shows *why*, needed for the paper):**

| Signal | Key | Expected for PPS |
|---|---|---|
| Reasoning gain on train | `train/pps_reasoning_gain` | rises over steps (mars arm logs it too, as a control) |
| Harmful CoT rate | `train/pps_harmful_cot_rate` | falls |
| Blame by stage | `train/pps_blame_seg{0..3}` | the stage blamed most at step 0 shrinks fastest |
| Train failure rate | `train/nc_fail_rate`, `train/dac_fail_rate` | falls faster than in the mars arm |

**Health (abort if violated):**

| Signal | Key | Abort if |
|---|---|---|
| Gradient norm | `train/grad_norm` | spikes > 10× or NaN |
| Update count | `train/pps_num_updates` | ≈ 0 for many steps (no learning signal) |
| Loss | `train/pps_pg_loss` / `train/pg_loss` | NaN |
| CoT format | printed `生成文字检查` text | CoTs collapse or stop containing the stage headings (degenerate solution: shorter CoT → less blame) |

### Verdict rule

- **GO for the full study:** PPS beats MARS on `val/pdm_score` at the final val step **and** on the val-curve average, without an EP regression. Confirm with a second seed if the gap is < 1 point.
- **Partial:** MARS ≫ NSR but PPS ≈ MARS. MARS alone is a solid workshop paper; keep PPS as an analysis tool (blame statistics) rather than the training signal.
- **NO-GO:** PPS < MARS, or PPS shortens / degenerates the CoT.

### Command

```bash
bash scripts/pps/pilot/phaseB_short_training.sh                 # nsr, mars, pps sequentially
ARMS="mars pps" bash scripts/pps/pilot/phaseB_short_training.sh  # skip nsr if you already have it
SEED=43 ARMS="mars pps" bash scripts/pps/pilot/phaseB_short_training.sh   # second seed
```

Experiments are named `pps_pilotB_{nsr,mars,pps}_s{SEED}`. LoRA checkpoints go to `<output_dir>/checkpoints/iter_<step>_lora` at each validation. Feed the best one to `scripts/pps/run_pps_eval.sh` for the blame analysis.

---

## Before you start

1. **Smoke test** (about 10 min), to confirm the code runs on GPU:
   ```bash
   MAX_SCENES=4 GPUS=1 bash scripts/pps/pilot/phaseA_signal_check.sh
   ```
2. Set `SWANLAB_API_KEY` in your shell for Phase B (or `SWANLAB_MODE=offline`).
3. Check the paths at the top of each script (`NAVSIM_*`, `MODEL_ROOT`, metric caches).
