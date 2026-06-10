
nvidia-smi

source ~/.bashrc



VLLM_ENV_NAME="base"      # 包含 vllm, torch, transformers 的重型环境
CLIENT_ENV_NAME="nd"  # 仅包含 requests, openai, Pillow 的轻量环境

# ==========================================
# 2. 动态获取节点信息和端口
# ==========================================
NODE_NAME=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
PORT=$(( 15000 + ($SLURM_JOB_ID % 10000) ))
API_URL="http://${NODE_NAME}:${PORT}"
echo "[$(date)] 服务将运行在: $API_URL"

# ==========================================
# 3. 在 VLLM 环境中启动后台服务
# ==========================================
echo "[$(date)] 激活 $VLLM_ENV_NAME 并启动 vLLM 服务..."
conda activate $VLLM_ENV_NAME

# 【关键技巧】获取当前环境下 vllm 命令的绝对路径，防止后续切换环境后路径失效
VLLM_CMD=$(which vllm)
echo "[$(date)] 使用的 vLLM 路径: $VLLM_CMD"

MODEL_PATH="OpenGVLab/InternVL2-8B"

# 使用 nohup 后台运行，并明确指定使用刚才获取的绝对路径命令
nohup $VLLM_CMD serve $MODEL_PATH \
    --trust-remote-code \
    --limit-mm-per-prompt image=1 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --port $PORT \
    --host 0.0.0.0 \
    > vllm_server_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM 服务已启动，PID: $VLLM_PID"

# ==========================================
# 4. 轮询等待服务就绪
# ==========================================
echo "[$(date)] 等待 vLLM 服务就绪..."
MAX_RETRIES=60
RETRY_COUNT=0

while true; do
    if curl -s -f "${API_URL}/health" > /dev/null; then
        echo "[$(date)] ✅ vLLM 服务已就绪！"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[$(date)] ❌ 错误: vLLM 服务启动超时！查看日志:"
        cat vllm_server_${SLURM_JOB_ID}.log
        kill $VLLM_PID
        exit 1
    fi
    sleep 5
done

# ==========================================
# 5. 切换到 CLIENT 环境，执行推理脚本
# ==========================================
echo "[$(date)] 切换到 $CLIENT_ENV_NAME 环境准备执行推理..."
conda activate $CLIENT_ENV_NAME

# 验证 client 环境确实很轻量 (可选)
echo "[$(date)] Client 环境 Python 路径: $(which python)"

# 创建并执行推理脚本
cat << EOF > infer_client.py
import requests
import json

api_url = "${API_URL}"

payload = {
    "model": "OpenGVLab/InternVL2-8B",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请仔细观察这张图片，并一步步推理（Chain of Thought），最后给出结论。"},
                # 实际使用时请替换为真实的 base64 图片数据
                # {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
    ],
    "max_tokens": 1024,
    "temperature": 0.7
}

print(f"[{api_url}/v1/chat/completions] 发送请求中...")
response = requests.post(f"{api_url}/v1/chat/completions", json=payload)

if response.status_code == 200:
    result = response.json()
    print("\n=== 模型推理结果 ===")
    print(result['choices'][0]['message']['content'])
    print(f"\n=== 耗时统计 ===")
    print(f"Prompt Tokens: {result['usage']['prompt_tokens']}")
    print(f"Completion Tokens: {result['usage']['completion_tokens']}")
else:
    print(f"❌ 请求失败: {response.status_code}")
    print(response.text)
EOF

python infer_client.py

# ==========================================
# 6. 清理后台进程，释放 GPU
# ==========================================
echo "[$(date)] 推理完成，正在清理 vLLM 后台进程 (PID: $VLLM_PID)..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "[$(date)] ✅ 任务全部完成，GPU 资源已安全释放。"