#!/bin/bash
# PPS pilot, phase A (no training, ~2-3 h on 8 GPUs): does the PPS signal exist?
# 200 high-risk TRAIN scenes x G=4 CoTs x K=4 noises, prefix probes (P=4) on every rollout,
# plus a fresh-noise re-score and a drop-blamed-vs-drop-random segment ablation.
# Prints a GO / NO-GO table (scripts/pps/pilot/summarize_pilot.py).
#
#   bash scripts/pps/pilot/phaseA_signal_check.sh
#   MAX_SCENES=100 GPUS=4 bash scripts/pps/pilot/phaseA_signal_check.sh

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
DIFF_PATH=${DIFF_PATH:-"$MODEL_ROOT/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"}
TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache_train"}
VAL_METRIC_CACHE_PATH=${VAL_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache"}

MAX_SCENES=${MAX_SCENES:-200}
G=${G:-4}
MARS_K=${MARS_K:-4}
PPS_SEGMENTS=${PPS_SEGMENTS:-4}
EXP_NAME=${EXP_NAME:-"pps_pilotA_${PPS_ESTIMATOR:-truncate}_cot${COT_ONLY:-true}_n${MAX_SCENES}_G${G}_K${MARS_K}_P${PPS_SEGMENTS}"}

export GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT:-63699}

torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pps_pilot.py \
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
    agent.rl_algorithm=pps \
    agent.mars_cfg.num_planner_samples=${MARS_K} \
    agent.mars_cfg.common_random_numbers=true \
    agent.mars_cfg.progress_coef=0.0 \
    agent.pps_cfg.max_segments=${PPS_SEGMENTS} \
    agent.pps_cfg.use_stage_markers=true \
    agent.pps_cfg.estimator=${PPS_ESTIMATOR:-truncate} \
    agent.pps_cfg.mc_continuations=${PPS_MC_M:-2} \
    +diagnostic.split=train \
    +diagnostic.max_scenes=${MAX_SCENES} \
    cache_path=null \
    seed=42 \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=4 \
    agent.metric_cache_path=${TRAIN_METRIC_CACHE_PATH} \
    agent.vlm_path=${VLM_PATH} \
    agent.vlm_size="small" \
    agent.diff_path=${DIFF_PATH} \
    agent.dit_type="small" \
    agent.per_sample_rollout=${G} \
    agent.max_text_tokens=512 \
    agent.max_padding_len=2800

PILOT_DIR=$(ls -d "$NAVSIM_EXP_ROOT/$EXP_NAME"/*/pps_pilot 2>/dev/null | tail -1)
[ -z "$PILOT_DIR" ] && PILOT_DIR="$NAVSIM_EXP_ROOT/$EXP_NAME/pps_pilot"
python $NAVSIM_DEVKIT_ROOT/scripts/pps/pilot/summarize_pilot.py "$PILOT_DIR" --json "$PILOT_DIR/summary.json"
