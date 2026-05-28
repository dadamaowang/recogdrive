
source ~/.bashrc
conda activate nd
which python

nvidia-smi


export NAVSIM_EXP_ROOT="/UserData/xnq/navsim_workspace/exp"    

export NAVSIM_DEVKIT_ROOT="/root/recogdrive"
export OPENSCENE_DATA_ROOT="/UserData/xnq/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"


TRAIN_TEST_SPLIT=navtest
EXP_NAME=test_recogdrive_2b_ori


export MASTER_PORT=63669
export PORT=63665
export GPUS=2
export GPUS_PER_NODE=$GPUS

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
# export CUDA_LAUNCH_BLOCKING=1 # open only when debug

# export HYDRA_FULL_ERROR=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/test_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXP_NAME \
    metric_cache_path="/UserData/rcd/navsim_workspace/exp/metric_cache/" \
    agent=negdrive_agent \
    agent.mode="eval" \
    agent.metric_cache_path="/UserData/rcd/navsim_workspace/exp/metric_cache/" \
    cache_path=null \
    force_cache_computation=false \
    agent.vlm_path="/UserData/xnq/recog_ori_models/ReCogDrive_VLM_2B" \
    agent.mode="eval" \
    agent.vlm_size="small" \
    agent.diff_path="/UserData/xnq/recog_ori_models/Diffusion_Planner_2B/Diffusion_Planner_For_2B.ckpt"  \
    agent.dit_type="small" \
    agent.per_sample_rollout=4 \
    agent.bag_g=4 \
    agent.max_text_tokens=128 \
    agent.max_padding_len=2800 \
    agent.vlm_lr=1e-5 \
    agent.opt_weight_decay=0.01 \
    agent.opt_eps=1e-8 \


