#!/bin/bash
# Variance-decomposition diagnostic: how much of the safety-reward variance comes from
# the diffusion planner's noise vs. from the VLM's CoT? Run this BEFORE training.
#
# Usage:
#   bash scripts/mars/run_mars_diagnostic.sh
#   G=8 MARS_K=8 MAX_BATCHES=20 bash scripts/mars/run_mars_diagnostic.sh
#   VLM_LORA_PATH=/path/to/nsr_lora bash scripts/mars/run_mars_diagnostic.sh   # diagnose a trained policy

source ~/.bashrc
conda activate ndfl03

export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/root/navsim_workspace/exps"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/root/recogdrive"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"/root/navsim_workspace/dataset"}
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export HYDRA_FULL_ERROR=1

MODEL_ROOT=${MODEL_ROOT:-"/root/navsim_workspace/models"}
VLM_PATH=${VLM_PATH:-"$MODEL_ROOT/ReCogDrive-VLM-2B"}
VLM_LORA_PATH=${VLM_LORA_PATH:-""}
DIFF_PATH=${DIFF_PATH:-"$MODEL_ROOT/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"}
TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache_train"}
VAL_METRIC_CACHE_PATH=${VAL_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache"}

G=${G:-8}
MARS_K=${MARS_K:-8}
MARS_CRN=${MARS_CRN:-true}
SPLIT=${SPLIT:-val}                  # val = high_risk_test (navtest), train = high_risk_train
MAX_BATCHES=${MAX_BATCHES:-null}     # per rank; null = whole split
BATCH_SIZE=${BATCH_SIZE:-4}
EXP_NAME=${EXP_NAME:-"mars_diag_${SPLIT}_G${G}_K${MARS_K}_crn${MARS_CRN}"}

export GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT:-63679}

torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_mars_diagnostic.py \
    train_test_split=navtrain \
    train_scene_subset=high_risk \
    high_risk.tokens_path="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt" \
    val_scene_subset=high_risk_test \
    high_risk.val_tokens_path="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_test/high_risk_test.txt" \
    validation.metric_cache_path=${VAL_METRIC_CACHE_PATH} \
    experiment_name=$EXP_NAME \
    agent=negdrive_agent \
    agent.mode=eval \
    agent.pass_cot_token_only=${COT_ONLY:-true} \
    agent.rl_algorithm=mars \
    agent.mars_cfg.num_planner_samples=${MARS_K} \
    agent.mars_cfg.common_random_numbers=${MARS_CRN} \
    +diagnostic.split=${SPLIT} \
    +diagnostic.max_batches=${MAX_BATCHES} \
    cache_path=null \
    seed=42 \
    dataloader.params.batch_size=${BATCH_SIZE} \
    dataloader.params.num_workers=4 \
    agent.metric_cache_path=${TRAIN_METRIC_CACHE_PATH} \
    agent.vlm_path=${VLM_PATH} \
    agent.vlm_lora_path=${VLM_LORA_PATH} \
    agent.vlm_size="small" \
    agent.diff_path=${DIFF_PATH} \
    agent.dit_type="small" \
    agent.per_sample_rollout=${G} \
    agent.max_text_tokens=512 \
    agent.max_padding_len=2800

OUTPUT_DIR="$NAVSIM_EXP_ROOT/$EXP_NAME"
DIAG_DIR=$(ls -d "$OUTPUT_DIR"/*/mars_diagnostic 2>/dev/null | tail -1)
[ -z "$DIAG_DIR" ] && DIAG_DIR="$OUTPUT_DIR/mars_diagnostic"
python $NAVSIM_DEVKIT_ROOT/scripts/mars/summarize_mars_diagnostic.py "$DIAG_DIR" --json "$DIAG_DIR/summary.json"
