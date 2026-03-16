# -*- encoding: utf-8 -*-
'''
@File    :   negdrive_backbone.py
@Time    :   2026/01/17 19:11:55
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   正在改成生成 text tokens, 还是用 diffusion planner 优化
@Desc    :   重要
'''


from typing import List, Optional, Tuple, Union, Literal
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import LoraConfig, get_peft_model, TaskType

from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


system_message = """    
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""
# TODO 这个 system prompt 有空也得改下


@dataclass  # NOTE 记得 dataclass dec
class NegDriveBackboneOutput:  
    """
    Output For Backbone RLVR training
    """

    logits: torch.Tensor    # [B, SeqLen, VocabSize] 
    hidden_states: tuple    #  tuple of [B, SeqLen, HiddenDim]
    input_ids: torch.Tensor     # [B, SeqLen]
    attention_mask: torch.Tensor    # [B, SeqLen]



@dataclass
class NegDriveGenOutput:
    """
    Everything needed downstream after VLM text generation.

    """
    full_ids: torch.Tensor    # [B, PromptLen + NewTokens]
    attention_mask: torch.Tensor    # [B, PromptLen + NewTokens]
    response_start_idx: int     # scalar - where generated tokens begin
    text_actions: List[str]     # decoded text, len=B






class NegDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda",
                 # TODO 添加 lora config 
                 ):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == "internvl":
            # TODO 头 + LoRA
            # LoRA 策略问题

            self.model = AutoModel.from_pretrained(     
                checkpoint_path,
                torch_dtype="auto",     
                low_cpu_mem_usage=True,     # TODO 
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device  
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256
            
            self._set_internvl_finetune_mode()  

            
        elif self.model_type == 'qwen':
            raise NotImplementedError
            # TODO qwen 也得 config
            # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #     checkpoint_path,
            #     torch_dtype=torch.bfloat16,
            #     device_map=self.device,
            #     trust_remote_code=True
            # )
            # self.tokenizer = AutoProcessor.from_pretrained(
            #     checkpoint_path,
            #     trust_remote_code=True
            # )
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")

        self.model.gradient_checkpointing_enable()
        self._print_trainable_parameters()  
        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")


    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
        

    def _set_internvl_finetune_mode(self):
        """
        (for internvl) Freeze vision encoder, mlp projector, and apply lora to language model 

        """

        # freeze vision encoder 
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        print('VISION ENCODER FROZEN.')

        # freeze mlp that projects vision features into language space.
        # In InternVL this is self.model.mlpq
        for param in self.model.mlp1.parameters():
            param.requires_grad = False 
        print("MLP PROJECTOR FROZEN")
    
        # apply LoRA to language model
        self._apply_lora_to_language_model(r=16,
                                           lora_alpha=32,
                                           lora_dropout=0.05)


    # def set_finetune_mode(self, finetune: bool): 全量微调
    #     """
    #     Sets the training mode for the VLM and configures which parameters are trainable.
    #     """
    #     self.finetune = finetune
    #     # self.finetune_llm_mode = mode

    #     # 1. 首先冻结所有参数
    #     for param in self.model.parameters():
    #         param.requires_grad = False

    #     if not self.finetune:
    #         print("Setting VLM to evaluation mode with all parameters frozen.")
    #         self.model.eval()
    #         return

    #     # 2. 如果需要微调，则解冻特定参数
    #     print(f"Setting VLM to training mode. Finetuning attention and MLP layers.")
    #     self.model.train()

    #     # 解冻attention和MLP层的参数用于微调
    #     trainable_keywords = ['attn', 'mlp']
    #     trainable_count = 0
    #     for name, param in self.model.named_parameters():
    #         if any(keyword in name for keyword in trainable_keywords):
    #             param.requires_grad = True
    #             trainable_count += 1
        
    #     print(f"Unfroze {trainable_count} parameter groups for fine-tuning.")

    def _apply_lora_to_language_model(self,
                    r: int,
                    lora_alpha: int,
                    lora_dropout: float,
                    ):
        """
        Apply LoRA to **Qwen2ForCausalLM** language model inside VLM backbone (attention and MLP layers).
        TODO 不同的 self.model.language_model 

        LoRA freezes the original weights and adds small trainable
        low-rank matrices A and B beside each target linear layer:
            W' = W + (B @ A) * (alpha / r) 
        
        Qwen2 layer names (confirmed by inspection):
            Attention: q_proj, k_proj, v_proj, o_proj
            MLP:       gate_proj, up_proj, down_proj  


        """
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                # Qwen2 attention projections
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # Qwen2 MLP projections
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
        )
        self.model.language_model = get_peft_model(
            self.model.language_model,
            lora_config
        )
        print(f"LORA APPLIED: R={r}, ALPHA={lora_alpha}, DROPOUT={lora_dropout}")
        

    def _print_trainable_parameters(self):
        """
        Docstring for _print_trainable_parameters
        
        :param self: Description
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100 * trainable / total if total > 0 else 0
        print(f"Trainable parameters: {trainable:,} / {total:,} "
              f"({pct:.2f}%)"
              )
        

    def forward(self, 
                pixel_values: torch.Tensor, 
                questions: List[str], 
                num_patches_list: List[int]
                ):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")

        queries = self._build_queries(pixel_values, questions, num_patches_list)

        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, 
                                      return_tensors='pt', 
                                      padding='max_length', 
                                      max_length=2800
                                      )  # TODO change max length

        device = torch.device('cuda')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

        # return self.model(
        #         # pixel_values=pixel_values.bfloat16(),  # 原始 code 是这样的 
        #         pixel_values=pixel_values,
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         image_flags=image_flags.squeeze(-1),
        #         output_hidden_states=True,
        #         return_dict=True,
        # )

        model_outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

        """
        NOTE output class definition
        @dataclass
        class CausalLMOutputWithPast(ModelOutput):
            loss: Optional[torch.FloatTensor] = None

            logits: torch.FloatTensor = None
                (batch_size, sequence_length, config.vocab_size)
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)

            past_key_values: Optional[List[torch.FloatTensor]] = None(?)

            hidden_states: Optional[Tuple[torch.FloatTensor]] = None (?)
                returned when ``output_hidden_states=True``
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.

            attentions: Optional[Tuple[torch.FloatTensor]] = None (?)
        """
        return NegDriveBackboneOutput(
            logits=model_outputs.logits,
            hidden_states=model_outputs.hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    

    def generate_text_actions(
                self,
                pixel_values: torch.Tensor,
                questions: List[str],
                num_patches_list: List[int], 
                max_new_tokens: int = 64,   # TODO 这里到底生成多少个比较好
        ) -> NegDriveGenOutput:
        """
        Run VLM in generation mode to produce one text response per batch item. 
        Called G times in _step() - each call produces different tokens (do_sample=True) 
        source of diversity for GRPO's G group samples

        Args:
            pixel_values:     [B * NumPatches, C, H, W]
            questions:        List[str], len=B
            num_patches_list: List[int], len=B
            max_new_tokens:   How many tokens to generate per response.
                            
        Returns:
            GenerationOutput                            
        """
        queries = self._build_queries(pixel_values, questions, num_patches_list)
        
        self.tokenizer.padding_size = 'left'
        model_inputs = self.tokenizer(
            queries,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2800,
        )
        device = torch.device("cuda")
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        prompt_len = input_ids.shape[1]

        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

        import inspect
        print(inspect.signature(self.model.generate))
        print("问题排查")

        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )   # [B, PromptLen + max_new_tokens]
        print(f"检查三：生成的 id 检查：{generated_ids}")

        # TODO 目的？
        # Build attention mask for full sequence (prompt + generated)
        full_len = generated_ids.shape[1]
        full_attention_mask = torch.ones(
            generated_ids.shape[0], full_len,
            dtype=torch.long, device=device
        )
        # Restore prompt padding (left-padded prompt may have 0s on the left)
        full_attention_mask[:, :prompt_len] = attention_mask

        # Decode generated tokens only (strip prompt)
        response_ids = generated_ids[:, prompt_len:]
        text_actions = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        print(f"检查四：生成的文字检查: {text_actions}")

        return NegDriveGenOutput(
            full_ids=generated_ids,
            attention_mask=full_attention_mask,
            response_start_idx=prompt_len,
            text_actions=text_actions
        )


    def _build_queries(self,
                       pixel_values,
                       questions,
                       num_patches_list
                      ):
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        return queries
    

    def compute_gaussian_logprob(
            self,
            policy_output, # NegDriveBackboneOutput,
            ref_output = None, # NegDriveBackboneOutupt
            log_std: float = 0.0,
            pool_strategy: str = "last_non_pad"
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For GRPO reward computation when VLM act as cognitive backbone

        """

        policy_h = self.pool_hidden_state(
            hidden_states = policy_output.hidden_states,
            atten_masks = policy_output.attention_mask
        )

        # TODO 处理 ref_h 
        # # Pool both hidden states → [B, HiddenDim]
        # policy_h = pool_hidden_state(policy_output.hidden_states, policy_output.attention_mask, pool)
        # ref_h    = pool_hidden_state(ref_output.hidden_states,    ref_output.attention_mask,    pool)

        # # ref_h must NOT contribute gradients — it's the fixed Gaussian mean
        # ref_h = ref_h.detach()        

        # scalar: σ²
        sigma_sq = torch.exp(
            2 * torch.tensor(log_std, dtype=policy_h.dtype, device=policy_h.device)
        )   # TODO
        """
        Concrete Numbers

        log_std = 0.0   →  σ=1.0,  σ²=1.0   (default, balanced)
        log_std = 1.0   →  σ=2.72, σ²=7.39  (wide, tolerant of drift)
        log_std = -1.0  →  σ=0.37, σ²=0.14  (tight, penalizes drift heavily)
        
        """
        log_probs = -0.5 * (policy_h.pow(2) / sigma_sq).sum(dim=-1)    # TODO

        # TODO ref_h 的
        # diff         = policy_h - ref_h                     # [B, HiddenDim]
        # log_probs    = -0.5 * (diff.pow(2) / sigma_sq).sum(dim=-1)  # [B]

        return log_probs, policy_h  # both returned — policy_h reused by diffusion planner


    def pool_hidden_state(self,
                          hidden_states: torch.Tensor,
                          atten_masks: torch.Tensor,    # [B, S], 1=real token, 0=padding.
                          pool_strategy: Literal["last_non_pad", "mean"] = "last_non_pad", 
                          ) -> torch.Tensor:
        """
        NOTE 
        for GRPO, we need **one vector per batch item** to compute reward.
        In this case, we got outputs.hidden_state [B, SeqLen, HiddenDim], 
        which has SeqLen vectors and we only need one (or, dim is one)
        e.g., we can pick the last one, or we compute mean, etc.

        TODO: 选择的策略？可调研

        pool_strategy:
            - last_non_pad : 
            - mean: 

        Args:
            pool_strategy:
                last_non_pad: last real token (best for causal LM) TODO
                mean: mean over all real tokens
                ...
        
        Returns:
            h: [B, HiddenDim]
            
        """
        last_layer = hidden_states[-1]
        B, S, D = last_layer.shape

        if pool_strategy == "last_non_pad":
            """
            NOTE 
            attention_mask tells you which tokens are real (1) vs padding (0)
            [1, 1, 1, 1, 1, 1, 0, 0, 0]
            ↑ real tokens ↑  ↑ padding ↑

            sum can tell the last real token's idx

            standard causal LMs's last token has attented to all previous tokens, thus it is the most info-rich position
            """
            last_idx = atten_masks.sum(dim=-1) - 1      # TODO 这个为什么每次不一样
            last_idx = last_idx.clamp(min=0).long()    
            # (1) safety guard (could produce -1)
            # (2) Tensor indexing in PyTorch requires integer (Long) dtype.

            h = last_layer[
                torch.arange(B, device=last_layer.device),
                last_idx,
            ]   # torch.Size([1, 1536]) [B, HiddenDim]

        elif pool_strategy == "mean":
            raise NotImplementedError

    #     elif pool == "mean":
    #         mask = attention_mask.unsqueeze(-1).float()     # [B, S, 1]
    #         h = (last_layer * mask).sum(dim=1)              # [B, HiddenDim]
    #         h = h / mask.sum(dim=1).clamp(min=1)            # [B, HiddenDim]

    #     else:
    #         raise ValueError(f"Unknown pool strategy: '{pool}'")
        return h



    # def extract_logprobs_from_outputs(
    #         self,
    #         outputs: NegDriveBackboneOutput,
    #         resp_start_idx: Optional[int] = None,
    #     ) -> torch.Tensor:
    #     """
    #     Extract per-token log prob of generated tokens. (to compute loss for RL training)

    #     TODO
    #     当使用的 VLM 添加了 head, 使用这种方式

    #     """
        
    #     logits = outputs.logits     # torch.Size([B, 2800(seq_len), 151682])

    #     return outputs

