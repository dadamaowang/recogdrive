
nvidia-smi

export NAVSIM_EXP_ROOT="/UserData/xnq/navsim_workspace/exp"     # exp log 存放; cache_dataset 位置存放

export NAVSIM_DEVKIT_ROOT="/root/recogdrive"
export OPENSCENE_DATA_ROOT="/UserData/xnq/navsim_workspace/dataset"
# export OPENSCENE_DATA_ROOT="/UserData/xnq/nav_mini"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# export CACHE_PATH=null

TRAIN_TEST_SPLIT=navtrain   # data
EXP_NAME=pre_exp_2b_01


export MASTER_PORT=63669
export PORT=63665
export GPUS=4
export GPUS_PER_NODE=$GPUS

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"


torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_negdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXP_NAME \
    cache_path=null \
    agent=negdrive_agent \
    force_cache_computation=False \
    dataloader.params.batch_size=8 \
    trainer.params.max_epochs=2 \
    trainer.params.devices=${GPUS} \
    trainer.params.strategy="ddp_find_unused_parameters_false" \
    agent.metric_cache_path="/UserData/rcd/navsim_workspace/exp/metric_cache_train" \
    agent.vlm_path="/UserData/xnq/my_models/Policy/ReCogDrive-VLM-2B" \
    agent.vlm_size="small" \
    agent.diff_path="/UserData/xnq/my_models/Planner/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"

