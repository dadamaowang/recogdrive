
# NOTEICE
# 1. metric_cache 路径检查
# 2. default_training 里超参都检查一下


source ~/.bashrc
conda activate ndfl02
which python


# # open clash (optional, not required for SwanLab)
# ~/clash -d ~/.config/clash/ > clash_job.log 2>&1 &
# CLASH_PID=$!

# echo "Waiting for Clash to start..."
# for i in {1..15}; do
#     if ss -tuln | grep -q ":7890"; then
#         echo "Clash is ready!"
#         break
#     fi
#     sleep 1
# done

# ps aux | grep clash

# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890


nvidia-smi

export NAVSIM_EXP_ROOT="/root/navsim_workspace/exps"    

export NAVSIM_DEVKIT_ROOT="/root/recogdrive"
export OPENSCENE_DATA_ROOT="/root/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# export CACHE_PATH=null


TRAIN_TEST_SPLIT=navtrain
# 【训练集选择】full: 全量 train_logs | high_risk: trainval 高风险子集
TRAIN_SCENE_SUBSET=high_risk
HIGH_RISK_TOKENS_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_train/high_risk_train.txt"
# 【验证集选择】full: 全量 val_logs | high_risk_test: navtest 高风险子集
VAL_SCENE_SUBSET=high_risk_test
HIGH_RISK_VAL_TOKENS_PATH="${NAVSIM_DEVKIT_ROOT}/scripts/filter_high_risk_scenes/high_risk_test/high_risk_test.txt"
VAL_METRIC_CACHE_PATH="/root/navsim_workspace/exps/metric_cache"

# RL algorithm: nsr | grpo
RL_ALGORITHM=nsr
if [ "$RL_ALGORITHM" = "grpo" ]; then
    EXP_NAME=0617_yrs_GRPO-train
else
    EXP_NAME=0613_yrs_NSR-train
fi

EXP_NAME=test_env_ndfl02
FAST_DEV_RUN=true

MASTER_PORT=63669
PORT=63665
GPUS=2

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
export GPUS=${GPUS:-8}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

# SwanLab experiment tracking (国内云端，无需代理)
# 首次使用：在 https://swanlab.cn 获取 API Key，或执行 swanlab login
export SWANLAB_API_KEY="oKEbC9aKSzvCLOISQ3DLL"
# 离线模式（不上传云端，本地 swanboard 查看）：export SWANLAB_MODE=offline
export SWANLAB_MODE=${SWANLAB_MODE:-offline}


HYDRA_FULL_ERROR=1
torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_negdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    train_scene_subset=${TRAIN_SCENE_SUBSET} \
    high_risk.tokens_path=${HIGH_RISK_TOKENS_PATH} \
    val_scene_subset=${VAL_SCENE_SUBSET} \
    high_risk.val_tokens_path=${HIGH_RISK_VAL_TOKENS_PATH} \
    validation.metric_cache_path=${VAL_METRIC_CACHE_PATH} \
    experiment_name=$EXP_NAME \
    agent=negdrive_agent \
    agent.rl_algorithm=${RL_ALGORITHM} \
    cache_path=null \
    force_cache_computation=false \
    seed=42 \
    dataloader.params.batch_size=16 \
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
    agent.metric_cache_path="/root/navsim_workspace/exps/metric_cache_train" \
    agent.vlm_path="/root/navsim_workspace/models/ReCogDrive-VLM-2B" \
    agent.vlm_size="small" \
    agent.diff_path="/root/navsim_workspace/models/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"  \
    agent.dit_type="small" \
    agent.per_sample_rollout=6 \
    agent.bag_g=2 \
    agent.max_text_tokens=512 \
    agent.max_padding_len=2800 \
    agent.vlm_lr=4e-5 \
    agent.opt_weight_decay=0.01 \
    agent.opt_eps=1e-8 \
    agent.lora_r=16 \
    agent.lora_alpha=16 \
    agent.lora_dropout=0.05 


# ==========================================
# Finish
# ==========================================
echo "[$(date)] Finish Training "
# kill $CLASH_PID 2>/dev/null
# wait $CLASH_PID 2>/dev/null
echo "[$(date)] ✅ GPU Released"