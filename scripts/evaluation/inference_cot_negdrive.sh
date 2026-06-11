
nvidia-smi

source ~/.bashrc


VLLM_ENV_NAME="serve_vllm"    
CLIENT_ENV_NAME="nd"  

EXP_NAME="tmp_test"

# ------------------------
# 准备 vLLM Inference 服务
# ------------------------

NODE_NAME="localhost"
PORT=8001
API_URL="http://${NODE_NAME}:${PORT}"
echo "[$(date)] vLLM Inference Service: $API_URL"

echo "[$(date)] Activate $VLLM_ENV_NAME and start..."
conda activate $VLLM_ENV_NAME


VLLM_CMD=$(which vllm)
echo "[$(date)] 使用的 vLLM 路径: $VLLM_CMD"

VLM_PATH="/root/navsim_workspace/models/ReCogDrive-VLM-2B/"


nohup $VLLM_CMD serve $VLM_PATH \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 12}' \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --tokenizer $VLM_PATH \
    --tokenizer-mode auto \
    --gpu-memory-utilization 0.90 \
    --port $PORT \
    --host 0.0.0.0 \
    > vllm_server_${EXP_NAME}.log 2>&1 &


# LORA_PATH=""
# nohup vllm serve OpenGVLab/InternVL2-8B \
#     --trust-remote-code \
#     --dtype bfloat16 \
#     --max-model-len 4096 \
#     --limit-mm-per-prompt image=1 \
#     --enable-lora \
#     --max-lora-rank 16 \
#     --lora-modules negdrive_lora=/path/to/your/negdrive_lora \
#     --port 8000 \
#     --host 0.0.0.0 \
#     > vllm_lora_server.log 2>&1 &


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
#  执行测试脚本
# ------------------------

# ==========================================
# 5. 切换到 CLIENT 环境，执行推理脚本
# ==========================================
echo "[$(date)] 切换到 $CLIENT_ENV_NAME 环境准备执行推理..."
conda activate $CLIENT_ENV_NAME



# python infer_client.py



# # ==========================================
# # 6. 清理后台进程，释放 GPU
# # ==========================================
echo "[$(date)] 推理完成，正在清理 vLLM 后台进程 (PID: $VLLM_PID)..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "[$(date)] ✅ 任务全部完成，GPU 资源已安全释放。"