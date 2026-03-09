# -*- encoding: utf-8 -*-
'''
@File    :   negdrive_backbone.py
@Time    :   2026/01/17 19:11:55
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   正在考察 forward 和后续模型训练联动
@Desc    :   重要
'''


from typing import List, Optional, Tuple, Union, Literal
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

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


class NegDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda"
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
            # TODO : 只是一个 backbone ? 好像不是很合理
            # （1） 改成加一个头，微调
            # （2） COVT
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

        
        self.model.gradient_checkpointing_enable()  # TODO 

        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")


    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
        

    def forward(self, 
                pixel_values: torch.Tensor, 
                questions: List[str], 
                num_patches_list: List[int]
                ):
        
        # TODO 看看 Qwen 能不能也用这个 forward 
        
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        
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
        NOTE output class here
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
    

    def compute_gaussian_logprob(
            self,
            policy_output, # NegDriveBackboneOutput,
            ref_output = None, # NegDriveBackboneOutupt
            log_std: float = 0.0,
            pool_strategy: str = "last_non_pad"
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is for GRPO reward computation when VLM act as cognitive backbone

        """

        policy_h = self.pool_hidden_state(
            hidden_states = policy_output.hidden_states,
            atten_masks = policy_output.attention_mask
        )


        # # Pool both hidden states → [B, HiddenDim]
        # policy_h = pool_hidden_state(policy_output.hidden_states, policy_output.attention_mask, pool)
        # ref_h    = pool_hidden_state(ref_output.hidden_states,    ref_output.attention_mask,    pool)

        # # ref_h must NOT contribute gradients — it's the fixed Gaussian mean
        # ref_h = ref_h.detach()                              # [B, HiddenDim]

        # # Gaussian log prob (per hidden dimension, then summed)
        # sigma_sq = torch.exp(
        #     2 * torch.tensor(log_std, dtype=policy_h.dtype, device=policy_h.device)
        # )  # scalar: σ²

        # diff         = policy_h - ref_h                     # [B, HiddenDim]
        # log_probs    = -0.5 * (diff.pow(2) / sigma_sq).sum(dim=-1)  # [B]

        # return log_probs, policy_h  # both returned — policy_h reused by diffusion planner



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

            # DOING




#     if pool == "last_non_pad":
#         # Sum of attention mask = index of last real token + 1
#         last_idx = attention_mask.sum(dim=-1) - 1       # [B]
#         last_idx = last_idx.clamp(min=0).long()

#         h = last_layer[
#             torch.arange(B, device=last_layer.device),
#             last_idx,
#         ]  # [B, HiddenDim]

#     elif pool == "mean":
#         mask = attention_mask.unsqueeze(-1).float()     # [B, S, 1]
#         h = (last_layer * mask).sum(dim=1)              # [B, HiddenDim]
#         h = h / mask.sum(dim=1).clamp(min=1)            # [B, HiddenDim]

#     else:
#         raise ValueError(f"Unknown pool strategy: '{pool}'")

#     return h  # [B, HiddenDim]





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
