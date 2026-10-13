#!/bin/bash
# PPS pilot, phase B (only if phase A says GO): short head-to-head training.
# Three arms with IDENTICAL rollouts/compute settings; only the advantage differs:
#   nsr        : rl_algorithm=nsr (G=4, 1 planner draw)            -- the paper-1 method
#   mars       : rl_algorithm=pps, shaping_coef=0 (== MARS, K=4)    -- isolates noise marginalization
#   pps        : rl_algorithm=pps, shaping_coef=1 (K=4, P=4)        -- full method
# 120 steps each, validation every 20 steps on the full high_risk_test split (546 navtest scenes).
#
#   bash scripts/pps/pilot/phaseB_short_training.sh            # all three arms, sequentially
#   ARMS="mars pps" bash scripts/pps/pilot/phaseB_short_training.sh

set -e
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

ARMS=${ARMS:-"nsr mars pps"}
STEPS=${STEPS:-120}
VAL_EVERY=${VAL_EVERY:-20}
VAL_BATCHES=${VAL_BATCHES:-1.0}       # fraction (float) or number of batches per rank (int)
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}
export GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT:-63709}

for ARM in $ARMS; do
    case $ARM in
        nsr)  ALGO=nsr; LAMBDA=0.0 ;;
        mars) ALGO=pps; LAMBDA=0.0 ;;
        pps)  ALGO=pps; LAMBDA=1.0 ;;
        *) echo "unknown arm $ARM"; exit 1 ;;
    esac
    EXP_NAME="pps_pilotB_${ARM}_s${SEED}"
    echo "[$(date)] ===== phase B arm=$ARM ($EXP_NAME) ====="
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
        $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_negdrive.py \
        train_test_split=navtrain \
        train_scene_subset=high_risk \
        high_risk.tokens_path="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt" \
        val_scene_subset=high_risk_test \
        high_risk.val_tokens_path="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_test/high_risk_test.txt" \
        validation.metric_cache_path=${VAL_METRIC_CACHE_PATH} \
        experiment_name=$EXP_NAME \
        agent=negdrive_agent \
        agent.rl_algorithm=${ALGO} \
        agent.mars_cfg.num_planner_samples=4 \
        agent.mars_cfg.common_random_numbers=true \
        agent.mars_cfg.progress_coef=0.5 \
        agent.mars_cfg.progress_min=0.6 \
        agent.mars_cfg.baseline=none \
        agent.pps_cfg.max_segments=4 \
        agent.pps_cfg.shaping_coef=${LAMBDA} \
        agent.pps_cfg.negative_only=true \
        agent.pps_cfg.probe=failures \
        run_baseline_validation=$([ "$ARM" = "nsr" ] && echo true || echo false) \
        cache_path=null \
        force_cache_computation=false \
        seed=${SEED} \
        dataloader.params.batch_size=${BATCH_SIZE} \
        dataloader.params.num_workers=8 \
        trainer.params.max_epochs=50 \
        trainer.params.max_steps=${STEPS} \
        trainer.params.val_check_interval=${VAL_EVERY} \
        trainer.params.limit_val_batches=${VAL_BATCHES} \
        trainer.params.log_every_n_steps=1 \
        trainer.params.devices=${GPUS} \
        trainer.params.strategy="ddp_find_unused_parameters_false" \
        trainer.params.precision=bf16-mixed \
        agent.metric_cache_path=${TRAIN_METRIC_CACHE_PATH} \
        agent.vlm_path=${VLM_PATH} \
        agent.vlm_size="small" \
        agent.diff_path=${DIFF_PATH} \
        agent.dit_type="small" \
        agent.per_sample_rollout=4 \
        agent.bag_g=4 \
        agent.max_text_tokens=512 \
        agent.max_padding_len=2800 \
        agent.vlm_lr=4e-5 \
        agent.opt_weight_decay=0.01 \
        agent.opt_eps=1e-8 \
        agent.lora_r=16 \
        agent.lora_alpha=16 \
        agent.lora_dropout=0.05
done
echo "[$(date)] phase B done. Compare val/* curves in SwanLab (project NegDriveComplete), experiments pps_pilotB_*."
