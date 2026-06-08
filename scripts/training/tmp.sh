


# 1. 启动 Clash 并记录它的进程 PID
~/clash -d ~/.config/clash/ > clash_job.log 2>&1 &
CLASH_PID=$!

# 2. 等待 Clash 启动完成 (最多等 15 秒)
echo "Waiting for Clash to start..."
for i in {1..15}; do
    # 检查 7890 端口是否已经开启监听
    if ss -tuln | grep -q ":7890"; then
        echo "Clash is ready!"
        break
    fi
    sleep 1
done

ps aux | grep clash

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

curl www.google.com