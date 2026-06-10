import base64
import requests
from io import BytesIO
from PIL import Image
import torch
from typing import List

# 假设这是您的类方法
def generate_text_actions_api(
        self,
        images: List[Image.Image],      # <--- 修改：传入原始 PIL 图像列表，而不是 pixel_values
        questions: List[str],
        max_new_tokens: int = 256,
        api_url: str = "http://localhost:8000" # 传入 vLLM 服务的地址
) -> NegDriveGenOutput:
    """
    通过 vLLM API 服务进行推理，并重构出 RL 训练所需的 Tensor 格式。
    """
    device = torch.device("cuda")
    batch_size = len(images)
    
    # ----------------------
    # 1. 准备 API 请求 Payload
    # ----------------------
    # 将图片转为 Base64
    base64_images = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="JPEG") # 或 PNG，取决于您的图像格式
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(f"data:image/jpeg;base64,{img_str}")

    # 构造 OpenAI 兼容的 messages 格式 (vLLM 会自动处理 InternVL 的 chat template)
    # 注意：这里不需要手动拼 <image> 标签，vLLM 会根据 image_url 自动插入
    messages_batch = []
    for b64_img, question in zip(base64_images, questions):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_img}},
                    {"type": "text", "text": question}
                ]
            }
        ]
        messages_batch.append(messages)

    # ----------------------
    # 2. 发送 HTTP 请求
    # ----------------------
    # 为了简单，这里循环请求。如果是高并发，建议使用 asyncio 或 vLLM 的 batch API
    text_actions = []
    for messages in messages_batch:
        payload = {
            "model": "OpenGVLab/InternVL2-8B", # 需与启动时的模型名一致
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": 1.0, # 对应您原来的 do_sample=True
            "top_p": 1.0,
        }
        
        response = requests.post(f"{api_url}/v1/chat/completions", json=payload)
        if response.status_code == 200:
            result = response.json()
            text_actions.append(result['choices'][0]['message']['content'])
        else:
            raise RuntimeError(f"vLLM API 请求失败: {response.text}")

    print(f"生成文字检查: {text_actions}")

    # ----------------------
    # 3. 重构 Tensor (关键步骤)
    # ----------------------
    # 因为 API 只返回文本，我们需要用 tokenizer 把 prompt 和 response 重新编码
    # 注意：这里假设 self._build_queries 能够把 question 转成纯文本 prompt。
    # 如果 vLLM 的 chat template 和 HF 的 _build_queries 不一致，这里的 token 数量可能对不上！
    prompts_text = self._build_queries_text_only(questions) # 您需要实现一个只返回纯文本的方法
    
    full_ids_list = []
    attention_mask_list = []
    prompt_lens = []

    for prompt_txt, response_txt in zip(prompts_text, text_actions):
        # 编码 prompt
        prompt_inputs = self.tokenizer(prompt_txt, return_tensors='pt', add_special_tokens=True)
        prompt_ids = prompt_inputs['input_ids'][0]
        prompt_len = len(prompt_ids)
        prompt_lens.append(prompt_len)
        
        # 编码 prompt + response (为了获取完整的 full_ids)
        # 注意：有些 tokenizer 在 encode 全文时，可能会在中间插入额外的 special tokens。
        # 最稳妥的做法是分别 encode，然后 cat。
        response_inputs = self.tokenizer(response_txt, return_tensors='pt', add_special_tokens=False)
        response_ids = response_inputs['input_ids'][0]
        
        # 拼接
        full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        full_ids_list.append(full_ids)
        
        # 构造 attention mask (全 1)
        mask = torch.ones_like(full_ids)
        attention_mask_list.append(mask)

    # Padding 到 batch 内最大长度
    max_len = max([len(ids) for ids in full_ids_list])
    padded_full_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    padded_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    
    # 假设是 right padding (如果是 left padding，请修改切片逻辑)
    for i in range(batch_size):
        seq_len = len(full_ids_list[i])
        padded_full_ids[i, :seq_len] = full_ids_list[i]
        padded_mask[i, :seq_len] = attention_mask_list[i]

    # 注意：由于每个 batch 的 prompt_len 可能不同（如果 question 长度不一），
    # 这里的 response_start_idx 应该是一个 List，或者取最大值。
    # 为了兼容您原来的 NegDriveGenOutput，这里取第一个的 prompt_len（假设 prompt 长度一致）
    response_start_idx = prompt_lens[0] 

    return NegDriveGenOutput(
        full_ids=padded_full_ids,
        attention_mask=padded_mask,
        response_start_idx=response_start_idx,
        text_actions=text_actions
    )

# 辅助方法：只构建纯文本 prompt，不包含 <image> 占位符（因为 vLLM API 会自动处理）
def _build_queries_text_only(self, questions: List[str]) -> List[str]:
    # 根据您原来的 _build_queries 逻辑修改，去掉图像相关的占位符
    # 例如直接返回 questions，或者加上特定的 system prompt
    return questions 