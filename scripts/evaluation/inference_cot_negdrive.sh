
nvidia-smi

source ~/.bashrc


VLLM_ENV_NAME="serve_vllm"    
CLIENT_ENV_NAME="nd"  

EXP_NAME="tmp_test"


# ------------------------
# Prepare vLLM Inference 
# ------------------------

NODE_NAME="localhost"
PORT=8001
API_URL="http://${NODE_NAME}:${PORT}"
echo "[$(date)] vLLM Inference Service: $API_URL"

echo "[$(date)] Activate $VLLM_ENV_NAME and start..."
conda activate $VLLM_ENV_NAME


VLLM_CMD=$(which vllm)
echo "[$(date)] which vLLM : $VLLM_CMD"


VLM_PATH="/root/navsim_workspace/models/ReCogDrive-VLM-2B/"


# nohup $VLLM_CMD serve $VLM_PATH \
#     --trust-remote-code \
#     --limit-mm-per-prompt '{"image": 12}' \
#     --max-model-len 4096 \
#     --dtype bfloat16 \
#     --tokenizer $VLM_PATH \
#     --tokenizer-mode auto \
#     --gpu-memory-utilization 0.90 \
#     --port $PORT \
#     --host 0.0.0.0 \
#     > vllm_server_${EXP_NAME}.log 2>&1 &


LORA_PATH="/root/navsim_workspace/exps/lora_tmp/"
MAX_LORA_RANK=16

nohup $VLLM_CMD serve $VLM_PATH \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 12}' \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --tokenizer $VLM_PATH \
    --tokenizer-mode auto \
    --enable-lora \
    --max-lora-rank $MAX_LORA_RANK \
    --lora-modules neg_lora=$LORA_PATH \
    --gpu-memory-utilization 0.90 \
    --port $PORT \
    --host 0.0.0.0 \
    > vllm_server_${EXP_NAME}.log 2>&1 &




VLLM_PID=$!
echo "[$(date)] vLLM service is launched: $VLLM_PID"

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



# python infer_client.py



# ==========================================
# release GPUs
# ==========================================
echo "[$(date)] Finished: $VLLM_PID)..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "[$(date)] ✅ GPU Released."