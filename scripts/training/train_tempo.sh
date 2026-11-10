#!/bin/bash
# =============================================================================
# TEMPO training for NegDrive (GRPO + segment-level prefix-tree TD advantage).
# Spec: TEMPO.md   Code: navsim/agents/negdrive/utils/tempo.py   Doc: doc/TEMPO_LAUNCH.md
#
# Every setting below can be overridden from the environment, e.g.
#   bash scripts/training/train_tempo.sh                                  # TEMPO, defaults
#   TEMPO_ENABLED=false bash scripts/training/train_tempo.sh              # GRPO baseline, same structured prompt
#   NODE_MATCH=embedding SIM_THRESHOLD=0.8 bash scripts/training/train_tempo.sh
#   TD_COEF=0.5 DELTA_MODE=first_token bash scripts/training/train_tempo.sh
#   FAST_DEV_RUN=true GPUS=1 BATCH_SIZE=1 G=2 bash scripts/training/train_tempo.sh   # smoke test
#
# SwanLab: export SWANLAB_API_KEY in your shell (not stored here), or SWANLAB_MODE=offline.
# =============================================================================

set -e

source ~/.bashrc
conda activate ${CONDA_ENV:-ndfl03}
which python

# ---------------- paths ----------------
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/root/navsim_workspace/exps"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/root/recogdrive"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"/root/navsim_workspace/dataset"}
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=${SWANLAB_MODE:-online}

MODEL_ROOT=${MODEL_ROOT:-"/root/navsim_workspace/models"}
VLM_PATH=${VLM_PATH:-"$MODEL_ROOT/ReCogDrive-VLM-2B"}
VLM_SIZE=${VLM_SIZE:-small}                       # small (2B) | large (8B)
DIFF_PATH=${DIFF_PATH:-"$MODEL_ROOT/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"}
DIT_TYPE=${DIT_TYPE:-small}
TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache_train"}
VAL_METRIC_CACHE_PATH=${VAL_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache"}

# ---------------- data ----------------
TRAIN_TEST_SPLIT=navtrain
TRAIN_SCENE_SUBSET=${TRAIN_SCENE_SUBSET:-high_risk}          # full | high_risk
HIGH_RISK_TOKENS_PATH=${HIGH_RISK_TOKENS_PATH:-"${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt"}
VAL_SCENE_SUBSET=${VAL_SCENE_SUBSET:-high_risk_test}         # full | high_risk_test
HIGH_RISK_VAL_TOKENS_PATH=${HIGH_RISK_VAL_TOKENS_PATH:-"${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_test/high_risk_test.txt"}

# ---------------- rollouts (GRPO) ----------------
G=${G:-4}                                # rollouts per scene (paper: 6; larger helps GRPO and TEMPO alike)
MAX_TEXT_TOKENS=${MAX_TEXT_TOKENS:-512}  # 5 tagged sections + 8 waypoints do not fit in 128
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}

# ---------------- structured CoT + TEMPO ----------------
STRUCTURED_COT=${STRUCTURED_COT:-true}
TEMPO_ENABLED=${TEMPO_ENABLED:-true}
TD_COEF=${TD_COEF:-1.0}                  # 0 reproduces the GRPO advantage
DELTA_MODE=${DELTA_MODE:-broadcast}      # broadcast | first_token
NODE_MATCH=${NODE_MATCH:-exact_normalized}   # exact_normalized | key_fn | embedding
SIM_THRESHOLD=${SIM_THRESHOLD:-0.9}      # embedding only
KEY_FN=${KEY_FN:-}                       # key_fn: "module.path:function" (empty = tempo.segment_key_fn hook)
EMBEDDER=${EMBEDDER:-}                   # embedding: "module.path:function" (empty = bag of words)
STRICT_ORDER=${STRICT_ORDER:-true}
SEGMENT_TAGS=${SEGMENT_TAGS:-}           # Hydra list, e.g. '["<perception>","<prediction>","<decision>","<check>","<trajectory>"]'

# ---------------- optimization ----------------
BATCH_SIZE=${BATCH_SIZE:-8}              # scenes per GPU per step
VLM_LR=${VLM_LR:-4e-5}
MAX_STEPS=${MAX_STEPS:-450}
VAL_EVERY=${VAL_EVERY:-9}
VAL_BATCHES=${VAL_BATCHES:-1.0}
SEED=${SEED:-42}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}

export GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT:-63669}

# ---------------- checks ----------------
if [ "$TEMPO_ENABLED" = "true" ] && [ "$STRUCTURED_COT" != "true" ]; then
    echo "ERROR: TEMPO_ENABLED=true requires STRUCTURED_COT=true"; exit 1
fi
if [ "$TEMPO_ENABLED" = "true" ] && [ "$NODE_MATCH" = "key_fn" ] && [ -z "$KEY_FN" ]; then
    echo "NOTE: NODE_MATCH=key_fn without KEY_FN uses the stub hook tempo.segment_key_fn;"
    echo "      training fails at setup unless you implemented it."
fi
if [ "${RUN_TESTS:-true}" = "true" ]; then
    echo "[$(date)] Running TEMPO unit tests (RUN_TESTS=false to skip)"
    python $NAVSIM_DEVKIT_ROOT/tests/test_tempo.py | tail -1
fi

if [ "$TEMPO_ENABLED" = "true" ]; then
    TAG="tempo_${NODE_MATCH}_td${TD_COEF}_${DELTA_MODE}"
else
    TAG="grpo_structured${STRUCTURED_COT}"
fi
EXP_NAME=${EXP_NAME:-"${TAG}_G${G}_s${SEED}"}

OPTIONAL_ARGS=()
[ -n "$KEY_FN" ] && OPTIONAL_ARGS+=("agent.tempo_cfg.key_fn=$KEY_FN")
[ -n "$EMBEDDER" ] && OPTIONAL_ARGS+=("agent.tempo_cfg.embedder=$EMBEDDER")
[ -n "$SEGMENT_TAGS" ] && OPTIONAL_ARGS+=("agent.segment_tags=$SEGMENT_TAGS")

echo "[$(date)] Start: $EXP_NAME  (GPUS=$GPUS, G=$G, TEMPO=$TEMPO_ENABLED, node_match=$NODE_MATCH, td_coef=$TD_COEF)"
torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_negdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    train_scene_subset=${TRAIN_SCENE_SUBSET} \
    high_risk.tokens_path=${HIGH_RISK_TOKENS_PATH} \
    val_scene_subset=${VAL_SCENE_SUBSET} \
    high_risk.val_tokens_path=${HIGH_RISK_VAL_TOKENS_PATH} \
    validation.metric_cache_path=${VAL_METRIC_CACHE_PATH} \
    experiment_name=$EXP_NAME \
    agent=negdrive_agent \
    agent.rl_algorithm=grpo \
    agent.structured_cot=${STRUCTURED_COT} \
    agent.tempo_cfg.enabled=${TEMPO_ENABLED} \
    agent.tempo_cfg.td_coef=${TD_COEF} \
    agent.tempo_cfg.delta_mode=${DELTA_MODE} \
    agent.tempo_cfg.node_match=${NODE_MATCH} \
    agent.tempo_cfg.sim_threshold=${SIM_THRESHOLD} \
    agent.tempo_cfg.strict_order=${STRICT_ORDER} \
    "${OPTIONAL_ARGS[@]}" \
    cache_path=null \
    force_cache_computation=false \
    seed=${SEED} \
    dataloader.params.batch_size=${BATCH_SIZE} \
    dataloader.params.num_workers=8 \
    trainer.params.fast_dev_run=${FAST_DEV_RUN} \
    trainer.params.limit_train_batches=1.0 \
    trainer.params.limit_val_batches=${VAL_BATCHES} \
    trainer.params.max_epochs=50 \
    trainer.params.max_steps=${MAX_STEPS} \
    trainer.params.devices=${GPUS} \
    trainer.params.strategy="ddp_find_unused_parameters_false" \
    trainer.params.val_check_interval=${VAL_EVERY} \
    trainer.params.log_every_n_steps=1 \
    trainer.params.precision=bf16-mixed \
    agent.metric_cache_path=${TRAIN_METRIC_CACHE_PATH} \
    agent.vlm_path=${VLM_PATH} \
    agent.vlm_size=${VLM_SIZE} \
    agent.diff_path=${DIFF_PATH} \
    agent.dit_type=${DIT_TYPE} \
    agent.per_sample_rollout=${G} \
    agent.bag_g=${G} \
    agent.max_text_tokens=${MAX_TEXT_TOKENS} \
    agent.rollout_temperature=${ROLLOUT_TEMPERATURE} \
    agent.max_padding_len=2800 \
    agent.vlm_lr=${VLM_LR} \
    agent.opt_weight_decay=0.01 \
    agent.opt_eps=1e-8 \
    agent.lora_r=16 \
    agent.lora_alpha=16 \
    agent.lora_dropout=0.05

echo "[$(date)] Finish: $EXP_NAME"
