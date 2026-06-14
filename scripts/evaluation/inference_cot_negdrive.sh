# NOTICE:
# 1. 提交选择 2 GPU
# VLM_PATH 和 Diff_PATH 尺寸参数一致


nvidia-smi

source ~/.bashrc


VLLM_ENV_NAME="serve_vllm"    
CLIENT_ENV_NAME="nd"  

VLLM_GPU=0
CLIENT_GPU=1

EXP_NAME="debug_0614_inf_cot"


# ------------------------
# Prepare vLLM Inference 
# ------------------------

NODE_NAME="localhost"
PORT=8001
API_URL="http://${NODE_NAME}:${PORT}"
echo "[$(date)] vLLM Inference Service: $API_URL"

echo "[$(date)] Activate $VLLM_ENV_NAME and start..."
conda activate $VLLM_ENV_NAME

export CUDA_VISIBLE_DEVICES=$VLLM_GPU
VLLM_CMD=$(which vllm)
echo "[$(date)] which vLLM : $VLLM_CMD"


VLM_PATH="/root/navsim_workspace/models/ReCogDrive-VLM-2B/"    # 记得检查 agent.diff_path 大小兼容


nohup $VLLM_CMD serve $VLM_PATH \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 12}' \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --tokenizer $VLM_PATH \
    --tokenizer-mode auto \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 \
    --port $PORT \
    --host 0.0.0.0 \
    > vllm_server_${EXP_NAME}.log 2>&1 &


# LORA_PATH="/root/navsim_workspace/exps/lora_tmp/"
# MAX_LORA_RANK=16

# nohup $VLLM_CMD serve $VLM_PATH \
#     --trust-remote-code \
#     --limit-mm-per-prompt '{"image": 12}' \
#     --max-model-len 4096 \
#     --dtype bfloat16 \
#     --tokenizer $VLM_PATH \
#     --tokenizer-mode auto \
#     --max-num-seqs 128 \
#     --enable-lora \
#     --max-lora-rank $MAX_LORA_RANK \
#     --lora-modules neg_lora=$LORA_PATH \
#     --gpu-memory-utilization 0.90 \
#     --port $PORT \
#     --host 0.0.0.0 \
#     > vllm_server_${EXP_NAME}.log 2>&1 &



VLLM_PID=$!
echo "[$(date)] vLLM service is launched: $VLLM_PID on GPU: $VLLM_GPU"

echo "[$(date)] wait vLLM initialize..."
MAX_RETRIES=120
RETRY_COUNT=0

while true; do
    if curl -s -f "${API_URL}/health" > /dev/null; then
        echo "[$(date)] ✅ vLLM Service is ready for inference！"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[$(date)] ❌ vLLM Time Out:"
        cat vllm_server_${EXP_NAME}.log
        kill $VLLM_PID
        exit 1
    fi
    sleep 5
done



# ------------------------
#  Execute test script
# ------------------------
echo "[$(date)] switch to $CLIENT_ENV_NAME environment..."
conda activate $CLIENT_ENV_NAME
export CUDA_VISIBLE_DEVICES=$CLIENT_GPU


export NAVSIM_EXP_ROOT="/root/navsim_workspace/exps"    

export NAVSIM_DEVKIT_ROOT="/root/recogdrive"
export OPENSCENE_DATA_ROOT="/root/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"


TRAIN_TEST_SPLIT=navtest
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/cot_inference.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    experiment_name=$EXP_NAME \
    metric_cache_path="/root/navsim_workspace/exps/metric_cache" \
    force_cache_computation=false \
    cache_path=null \
    +max_workers=24 \
    +api_base=$API_URL \
    agent=negdrive_agent \
    agent.mode="eval" \
    agent.metric_cache_path="/root/navsim_workspace/exps/metric_cache" \
    agent.vlm_path=$VLM_PATH \
    agent.vlm_size="small" \
    agent.diff_path="/root/navsim_workspace/models/ReCogDrive-Dif-2B/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"  \
    agent.dit_type="small" \
    agent.pass_cot_token_only=true \
    agent.per_sample_rollout=4 \
    agent.bag_g=4 \
    agent.max_text_tokens=512 \
    agent.max_padding_len=2800 \
    agent.vlm_lr=1e-5 \
    agent.opt_weight_decay=0.01 \
    agent.opt_eps=1e-8 \
    # agent.vlm_lora_path="/root/navsim_workspace/exps/0605_2b_negdrive_train_v1/last_lora/" \


# ==========================================
# release GPUs
# ==========================================
echo "[$(date)] Finished: $VLLM_PID)..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "[$(date)] ✅ GPU Released."