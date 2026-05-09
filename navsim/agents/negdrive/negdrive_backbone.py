# -*- encoding: utf-8 -*-
'''
@File    :   negdrive_backbone.py
@Time    :   2026/01/17 19:11:55
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   
@Desc    :  
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
                 max_padding_len: int = 2800,
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
        
        self.max_padding_len = max_padding_len

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == "internvl":
            # TODO 头 + LoRA
            # LoRA 策略问题

            self.model = AutoModel.from_pretrained(     
                checkpoint_path,
                dtype=torch.bfloat16,     
                low_cpu_mem_usage=True,     
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=None  
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
        if not hasattr(self.model.language_model, "peft_config"):
            raise RuntimeError("LoRA 失败！！！！")
        print(f"LORA APPLIED: R={r}, ALPHA={lora_alpha}, DROPOUT={lora_dropout}")


        # Check if LoRA layers are applied to the model and calculate VRAM usage.

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        

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
                num_patches_list: List[int],
                max_new_tokens: int = 512
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
                max_new_tokens: int = 256,   # TODO 这里到底生成多少个比较好
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
        # ----------------------
        #   Prepare Input
        # ----------------------
        queries = self._build_queries(pixel_values, questions, num_patches_list)
        
        self.tokenizer.padding_size = 'left'
        model_inputs = self.tokenizer(
            queries,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_padding_len,
        )
        device = torch.device("cuda")

        input_ids = model_inputs['input_ids'].to(device)    # [B, SeqLen]
        attention_mask = model_inputs['attention_mask'].to(device)  # [B, SeqLen]
        # ensure [B, SeqLen] (in case B=1)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, SeqLen]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)    # [1, SeqLen]

        self._check_mask_ratio(attention_mask)

        prompt_len = input_ids.shape[1]

        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

        # -----------------------
        #   Generate New Tokens
        # -----------------------
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            # image_flags=image_flags,  # TODO 这个不用有啥问题
            max_new_tokens=max_new_tokens,  
            do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )   # [B, max_new_tokens] 
        # 把 generated_ids 拼起来

        # ---------------------------------------
        #   Build Full Sequence Attention Mask
        # ---------------------------------------
        #  (prompt + generated)   TODO log_prob 计算，目的等
        full_ids = torch.cat([input_ids, generated_ids], dim=1)   # [B, prompt_len + new_tokens]

        full_attention_mask = torch.cat([
            attention_mask, 
            torch.ones(input_ids.shape[0], generated_ids.shape[1], dtype=torch.long, device=device),
        ], dim=1)   # [B, prompt_len + new_tokens]

        # ---------------------------------------
        #   Decode Generated Text
        # ---------------------------------------
        text_actions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        print(f"生成文字检查: {text_actions}")

        return NegDriveGenOutput(
            full_ids=full_ids,
            attention_mask=full_attention_mask,
            response_start_idx=prompt_len,
            text_actions=text_actions
        )

    
    def forward_with_ids(self,
                         pixel_values: torch.Tensor,   
                         input_ids: torch.Tensor,     # [B, SeqLen] - pre-build, includes response 
                         attention_mask: torch.Tensor    # [B, SeqLen]
                         ) -> NegDriveBackboneOutput:
        """
        Forward pass using pre-built input_ids instead of rebuilding from questions.
        Used after generate_text_actions() to get hidden states conditioned
        on the generated text response.

        This is the key method that makes each G rollout produce a DIFFERENT 
        hidden state - because each rollout has different generated tokens in input_ids.
        """
        device = torch.device('cuda')
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

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
    

    def _check_mask_ratio(self, attention_mask: torch.Tensor):
        """check if input+image have padding"""
        
        mask_ratio = attention_mask.float().mean().item()
        print(f"Attention mask ratio (non-padding tokens): {mask_ratio:.4f} (100%=no padding, 0%=all padding)")

        seq_lengths = attention_mask.sum(dim=1).tolist()
        print(f"Sequence lengths (non-padding tokens) per batch item: {seq_lengths}")

    
