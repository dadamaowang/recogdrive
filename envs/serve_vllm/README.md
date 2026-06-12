
部署 vLLM 在线推理需要的环境


conda create -n serve_vllm python=3.10


（开代理）
pip install uv
uv pip install vllm --torch-backend=auto

测试 
python -c "
import torch
import vllm

print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
print('gpu=', torch.cuda.get_device_name(0))
print('vllm=', vllm.__version__)
"
