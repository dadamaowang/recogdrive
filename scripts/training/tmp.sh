
export NAVSIM_EXP_ROOT="/root/autodl-tmp/exps/nv"     # exp log 存放
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
    agent=negdrive_agent 


# -----------------
# agent config
# -----------------
    # agent=negdrive_agent   navsim/planning/script/config/common/agent/[negdrive_agent].yaml
    # agent.lr=1e-4 \
    # agent.vlm_path='/path/to/pretrain_model' \
    # agent.cam_type='single' \
    # agent.grpo=True \
    # agent.cache_hidden_state=True \
    # agent.vlm_type="internvl" \
    # agent.checkpoint_path="'$CHECKPOINT'" \
    # agent.dit_type="small" \
    # agent.vlm_size="small" \
    # agent.sampling_method="ddim" \
    # agent.metric_cache_path="/path/to/metric_cache_dir" \
    # agent.reference_policy_checkpoint="'$CHECKPOINT'" \
    # trainer.params.max_epochs=10 \
    # dataloader.params.batch_size=8 \
    # experiment_name=training_internvl_agent_dit \
    # train_test_split=$TRAIN_TEST_SPLIT \
    # cache_path="/path/to/recogdrive_agent_cache_dir_train" \
    # use_cache_without_dataset=True \
    # force_cache_computation=False > train_negdriv_test01.txt 2>&1
