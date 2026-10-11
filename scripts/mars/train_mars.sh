#!/bin/bash
# MARS training: noise-marginalized negative reinforcement for the NegDrive VLM.
# Same setup as scripts/training/train_negdrive.sh, with rl_algorithm=mars.
#
# Usage:
#   bash scripts/mars/train_mars.sh                        # defaults below
#   MARS_K=8 MARS_CRN=false EXP_NAME=mars_k8_nocrn bash scripts/mars/train_mars.sh
#
# SwanLab: export SWANLAB_API_KEY in your shell (not stored here), or SWANLAB_MODE=offline.

source ~/.bashrc
conda activate ndfl03

export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/root/navsim_workspace/exps"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/root/recogdrive"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"/root/navsim_workspace/dataset"}
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=${SWANLAB_MODE:-online}

MODEL_ROOT=${MODEL_ROOT:-"/root/navsim_workspace/models"}
VLM_PATH=${VLM_PATH:-"$MODEL_ROOT/ReCogDrive-VLM-2B"}
DIFF_PATH=${DIFF_PATH:-"$MODEL_ROOT/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"}
TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache_train"}
VAL_METRIC_CACHE_PATH=${VAL_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache"}

TRAIN_TEST_SPLIT=navtrain
TRAIN_SCENE_SUBSET=high_risk
HIGH_RISK_TOKENS_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt"
VAL_SCENE_SUBSET=high_risk_test
HIGH_RISK_VAL_TOKENS_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_test/high_risk_test.txt"

# ---------------- MARS hyper-parameters ----------------
G=${G:-4}                                # CoT rollouts per scene
MARS_K=${MARS_K:-4}                      # planner noises per CoT
MARS_CRN=${MARS_CRN:-true}               # common random numbers across CoTs
MARS_PROGRESS_COEF=${MARS_PROGRESS_COEF:-0.5}
MARS_PROGRESS_MIN=${MARS_PROGRESS_MIN:-0.6}
MARS_BASELINE=${MARS_BASELINE:-none}     # none | group_mean

EXP_NAME=${EXP_NAME:-"mars_G${G}_K${MARS_K}_crn${MARS_CRN}_beta${MARS_PROGRESS_COEF}_${MARS_BASELINE}"}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
BATCH_SIZE=${BATCH_SIZE:-8}

export GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT:-63669}

echo "[$(date)] Start MARS training: $EXP_NAME"
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
    agent.rl_algorithm=mars \
    agent.mars_cfg.num_planner_samples=${MARS_K} \
    agent.mars_cfg.common_random_numbers=${MARS_CRN} \
    agent.mars_cfg.progress_coef=${MARS_PROGRESS_COEF} \
    agent.mars_cfg.progress_min=${MARS_PROGRESS_MIN} \
    agent.mars_cfg.baseline=${MARS_BASELINE} \
    cache_path=null \
    force_cache_computation=false \
    seed=42 \
    dataloader.params.batch_size=${BATCH_SIZE} \
    dataloader.params.num_workers=8 \
    trainer.params.fast_dev_run=${FAST_DEV_RUN} \
    trainer.params.limit_train_batches=1.0 \
    trainer.params.limit_val_batches=1.0 \
    trainer.params.max_epochs=50 \
    trainer.params.max_steps=450 \
    trainer.params.devices=${GPUS} \
    trainer.params.strategy="ddp_find_unused_parameters_false" \
    trainer.params.val_check_interval=9 \
    trainer.params.log_every_n_steps=1 \
    trainer.params.precision=bf16-mixed \
    agent.metric_cache_path=${TRAIN_METRIC_CACHE_PATH} \
    agent.vlm_path=${VLM_PATH} \
    agent.vlm_size="small" \
    agent.diff_path=${DIFF_PATH} \
    agent.dit_type="small" \
    agent.per_sample_rollout=${G} \
    agent.bag_g=${G} \
    agent.max_text_tokens=512 \
    agent.max_padding_len=2800 \
    agent.vlm_lr=4e-5 \
    agent.opt_weight_decay=0.01 \
    agent.opt_eps=1e-8 \
    agent.lora_r=16 \
    agent.lora_alpha=16 \
    agent.lora_dropout=0.05

echo "[$(date)] Finish MARS training: $EXP_NAME"
