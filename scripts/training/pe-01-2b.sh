
nvidia-smi

export NAVSIM_EXP_ROOT="/UserData/xnq/navsim_workspace/exp"    

export NAVSIM_DEVKIT_ROOT="/root/recogdrive"
export OPENSCENE_DATA_ROOT="/UserData/xnq/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# export CACHE_PATH=null

TRAIN_TEST_SPLIT=navtrain   
EXP_NAME=debug_ckpt_check
FAST_DEV_RUN=false

export MASTER_PORT=63669
export PORT=63665
export GPUS=2
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
    agent=negdrive_agent \
    cache_path=null \
    force_cache_computation=false \
    seed=42 \
    dataloader.params.batch_size=4 \
    dataloader.params.num_workers=8 \
    trainer.params.fast_dev_run=${FAST_DEV_RUN} \
    trainer.params.max_epochs=10 \
    trainer.params.devices=${GPUS} \
    trainer.params.strategy="ddp_find_unused_parameters_false" \
    trainer.params.limit_train_batches=0.1 \
    trainer.params.limit_val_batches=0.1 \
    trainer.params.precision=bf16-mixed \
    agent.metric_cache_path="/UserData/rcd/navsim_workspace/exp/metric_cache_train" \
    agent.vlm_path="/UserData/xnq/my_models/Policy/ReCogDrive-VLM-2B" \
    agent.vlm_lora_path="/UserData/xnq/navsim_workspace/exp/debug_ckpt_check/2026.05.21.16.55.00/tb/debug_ckpt_check/checkpoints/last_lora" \
    agent.vlm_size="small" \
    agent.diff_path="/UserData/xnq/my_models/Planner/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"  \
    agent.per_sample_rollout=4 \
    agent.bag_g=4 \
    agent.max_text_tokens=128 \
    agent.max_padding_len=2800 \
    agent.vlm_lr=1e-5 \
    agent.opt_weight_decay=0.01 \
    agent.opt_eps=1e-8 \

