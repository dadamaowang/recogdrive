

source ~/.bashrc
conda activate ndfl03 
which python
nvidia-smi


MASTER_PORT=63669
PORT=63665
GPUS=2

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export GPUS=${GPUS:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}


export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/root/navsim_workspace/exps"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/root/recogdrive"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"/root/navsim_workspace/dataset"}
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

MODEL_ROOT=${MODEL_ROOT:-"/root/navsim_workspace/models"}
VLM_PATH=${VLM_PATH:-"$MODEL_ROOT/ReCogDrive-VLM-2B"}
DIFF_PATH=${DIFF_PATH:-"$MODEL_ROOT/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"}
TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache_train"}
VAL_METRIC_CACHE_PATH=${VAL_METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache"}


TRAIN_TEST_SPLIT=navtrain
# 【训练集选择】full: 全量 train_logs | high_risk: trainval 高风险子集
TRAIN_SCENE_SUBSET=high_risk
HIGH_RISK_TOKENS_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt"
# 【验证集选择】full: 全量 val_logs | high_risk_test: navtest 高风险子集
VAL_SCENE_SUBSET=high_risk_test
HIGH_RISK_VAL_TOKENS_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_test/high_risk_test.txt"
VAL_METRIC_CACHE_PATH="/root/navsim_workspace/exps/metric_cache"


# ---------------- MARS hyper-parameters ----------------
G=${G:-4}                                # CoT rollouts per scene
MARS_K=${MARS_K:-4}                      # planner noises per CoT
MARS_CRN=${MARS_CRN:-true}               # common random numbers across CoTs
MARS_PROGRESS_COEF=${MARS_PROGRESS_COEF:-0.5}
MARS_PROGRESS_MIN=${MARS_PROGRESS_MIN:-0.6}
MARS_BASELINE=${MARS_BASELINE:-none}     # none | group_mean
# ---------------- PPS hyper-parameters ----------------
PPS_SEGMENTS=${PPS_SEGMENTS:-4}          # CoT segments per probed rollout
PPS_LAMBDA=${PPS_LAMBDA:-1.0}            # shaping coefficient
PPS_NEG_ONLY=${PPS_NEG_ONLY:-true}       # only blame risk-increasing segments
PPS_PROBE=${PPS_PROBE:-failures}         # failures | all
PPS_MARKERS=${PPS_MARKERS:-true}         # split at stage headings
PPS_ESTIMATOR=${PPS_ESTIMATOR:-truncate} # truncate | mc
PPS_MC_M=${PPS_MC_M:-2}                  # continuations per prefix (mc)
COT_ONLY=${COT_ONLY:-true}               # planner reads CoT tokens only (recommended, see prompt_v1_pre)

EXP_NAME=${EXP_NAME:-"pps_${PPS_ESTIMATOR}_cot${COT_ONLY}_G${G}_K${MARS_K}_P${PPS_SEGMENTS}_l${PPS_LAMBDA}_neg${PPS_NEG_ONLY}_${PPS_PROBE}"}
FAST_DEV_RUN=${FAST_DEV_RUN:-true}
BATCH_SIZE=${BATCH_SIZE:-8}


export HYDRA_FULL_ERROR=1
export SWANLAB_MODE=${SWANLAB_MODE:-online}
# SwanLab experiment tracking (国内云端，无需代理)
# 首次使用：在 https://swanlab.cn 获取 API Key，或执行 swanlab login
export SWANLAB_API_KEY="x"
# 离线模式（不上传云端，本地 swanboard 查看）：export SWANLAB_MODE=offline
export SWANLAB_MODE=${SWANLAB_MODE:-online}


echo "[$(date)] Start PPS training: $EXP_NAME"
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
    agent.rl_algorithm=pps \
    agent.mars_cfg.num_planner_samples=${MARS_K} \
    agent.mars_cfg.common_random_numbers=${MARS_CRN} \
    agent.mars_cfg.progress_coef=${MARS_PROGRESS_COEF} \
    agent.mars_cfg.progress_min=${MARS_PROGRESS_MIN} \
    agent.mars_cfg.baseline=${MARS_BASELINE} \
    agent.pps_cfg.max_segments=${PPS_SEGMENTS} \
    agent.pps_cfg.shaping_coef=${PPS_LAMBDA} \
    agent.pps_cfg.negative_only=${PPS_NEG_ONLY} \
    agent.pps_cfg.probe=${PPS_PROBE} \
    agent.pps_cfg.use_stage_markers=${PPS_MARKERS} \
    agent.pps_cfg.estimator=${PPS_ESTIMATOR} \
    agent.pps_cfg.mc_continuations=${PPS_MC_M} \
    agent.pass_cot_token_only=${COT_ONLY} \
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

echo "[$(date)] Finish PPS training: $EXP_NAME"