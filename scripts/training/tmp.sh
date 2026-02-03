
export NAVSIM_EXP_ROOT="/root/autodl-tmp/exps/nv"     # exp log 存放; cache_dataset 位置存放

export NAVSIM_DEVKIT_ROOT="/root/recogdrive/"
export OPENSCENE_DATA_ROOT="/root/autodl-tmp/Dataset/navtrain_tiny"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

CACHE_PATH=$NAVSIM_EXP_ROOT/negdrive_mc_train   # 这里是通过 run_metric_caching_train/test 脚本搞的 metric_caching

TRAIN_TEST_SPLIT=navtrain   # data
EXP_NAME=tmp_test


export MASTER_PORT=63669
export PORT=63665
export GPUS=1
export GPUS_PER_NODE=1
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=localhost
export MLP_WORKER_0_PORT=63669

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1

export HYDRA_FULL_ERROR=1


# torchrun \
#     --standalone \
#     --nproc_per_node=1 \
#     --nnodes=1 \
#     --node_rank=$MLP_ROLE_INDEX \
#     --master_addr=$MLP_WORKER_0_HOST \
#     --nproc_per_node=${GPUS} \
#     --master_port=$MLP_WORKER_0_PORT \
#     navsim/planning/script/run_training_negdrive.py 


# --------
# ONE GPU 
# -------

export CUDA_VISIBLE_DEVICES=0  

torchrun \
    --standalone \
    --nproc_per_node=1 \
    navsim/planning/script/run_training_negdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXP_NAME \
    cache_path=$CACHE_PATH \
    agent=negdrive_agent \
    force_cache_computation=False




# -----------------------------
# ONE GPU 参数说明
# ----------------------------

# export CUDA_VISIBLE_DEVICES=0  

# torchrun \
#     --standalone \
#     --nproc_per_node=1 \
#     navsim/planning/script/run_training_negdrive.py \
#     train_test_split=$TRAIN_TEST_SPLIT \
#     experiment_name=$EXP_NAME \
#     cache_path=$CACHE_PATH \
#     agent=negdrive_agent \
#     force_cache_computation=False 这里记住必须 False 不然 cache 的时候 bug


