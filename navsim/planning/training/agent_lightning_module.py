# -*- encoding: utf-8 -*-
'''
@File    :   agent_lightning_module.py
@Time    :   2026/01/04 10:46:58
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   DOING
@Desc    :   None
'''

import pytorch_lightning as pl

import torch
from torch import Tensor
from typing import Dict, Tuple, List, Any
import torch.nn.functional as F 

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.abstract_agent import AbstractAgent

from navsim.agents.negdrive.utils.internvl_preprocess import load_image
from navsim.agents.negdrive.utils.utils import format_number
from navsim.agents.negdrive.negdrive_backbone import NegDriveGenOutput




def decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
    """
    Decodes a batch of path tensors back into a list of file path strings.
    
    Args:
        path_tensor (torch.Tensor): A 2D tensor of shape 
            (batch_size, max_path_length) from the collate_fn.
    
    Returns:
        List[str]: A list of decoded file path strings.
    """
    decoded_paths = []
    for single_path_tensor in path_tensor:
        chars = []
        for code in single_path_tensor:
            code_item = code.item()
            if code_item == 0: 
                break
            chars.append(chr(code_item))
        decoded_paths.append("".join(chars))
    return decoded_paths



class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            predictions = self.agent.compute_loss(features, targets, prediction)

            loss = predictions.loss
            reward = predictions.reward
            policy_loss = predictions.policy_loss
            bc_loss = predictions.bc_loss
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()



def compute_response_logprobs(
        model: torch.nn.Module,
        pixel_values: torch.Tensor, 
        generation_output: NegDriveGenOutput,
    ) -> torch.Tensor:
    """ TODO 参考 verl 有空研究下这个计算方式
    Compute mean log prob over RESPONSE TOKENS ONLY, via a forward pass
    using the already-generated full_ids as input.

    NOTE
    Why a second forward pass?
    - generate() produces tokens autoregressively — no single logit tensor
    - We need logits over the full sequence in one shot for efficiency
    - So we feed full_ids back through the model as a normal forward pass
      and read off logits[prompt_len:] which correspond to generated tokens

    Args:
        model:             The VLM model (self.model inside backbone).
                           Pass policy model for training, ref model for KL.
        pixel_values:      [B*NumPatches, C, H, W]
        generation_output: Output from generate_text_actions().
                           Contains full_ids, attention_mask, response_start_idx.

    Returns:
        log_probs: [B]  mean log prob over response tokens per batch item.
    """

    full_ids       = generation_output.full_ids        # [B, S]
    attention_mask = generation_output.attention_mask  # [B, S]
    response_start = generation_output.response_start_idx  # scalar int    

    B, S = full_ids.shape
    device = full_ids.device

    # Build image_flags
    num_patches = pixel_values.shape[0]
    image_flags = torch.ones(num_patches, dtype=torch.long, device=device)    

    # Forward pass with full sequence (prompt + generated tokens)
    # This gives us logits at every position in one efficient call
    outputs = model(
        pixel_values=pixel_values,
        input_ids=full_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        output_hidden_states=False,  # don't need hidden states here
        return_dict=True,
    )
    logits = outputs.logits   # [B, S, VocabSize]

    # ── Causal shift ────────────────────────────────────────────────
    # logits[t] predicts the token at position t+1
    # So to score token at position t, use logits at position t-1
    shift_logits = logits[:, :-1, :]       # [B, S-1, V]
    shift_ids    = full_ids[:, 1:]         # [B, S-1]
    shift_mask   = attention_mask[:, 1:]   # [B, S-1]

    # ── Log probs over vocabulary ────────────────────────────────────
    log_probs_all = F.log_softmax(shift_logits, dim=-1)   # [B, S-1, V]

    # Gather log prob of the actual token at each position
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=shift_ids.unsqueeze(-1)      # [B, S-1, 1]
    ).squeeze(-1)     # [B, S-1]

    # ── Response-only mask ───────────────────────────────────────────
    # Zero out prompt positions — only score generated tokens
    # response_start_idx is in the original (unshifted) sequence
    # After shift, response starts at response_start_idx - 1
    response_mask = shift_mask.clone()
    response_mask[:, :response_start - 1] = 0   # zero out prompt

    # Apply combined mask
    token_log_probs = token_log_probs * response_mask     # [B, S-1]
    denom = response_mask.sum(dim=-1).clamp(min=1)        # [B]

    # Mean log prob per sequence over response tokens only
    mean_log_probs = token_log_probs.sum(dim=-1) / denom  # [B]
    return mean_log_probs



class AgentLightningVLMRL(pl.LightningModule):
    """Pytorch lightning wrapper for learnable vlm recogdrive agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

        self.G = 4  # TODO setting



    def _step(self, 
              batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
              logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch

        # ---------------------------- UNPACK features ------------------------------
        #                             (images + prompts)
        # ---------------------------------------------------------------------------
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()
        
        history_trajectory = features["history_trajectory"].cuda()  
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)

        high_command_one_hot = features["high_command_one_hot"].cuda()
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)
        
        image_path_tensor = features["image_path_tensor"]
        if image_path_tensor.ndim == 1: image_path_tensor = image_path_tensor.unsqueeze(0)
        image_paths = decode_paths_from_tensor(image_path_tensor)

        pixel_values_list = [load_image(path) for path in image_paths] 
        num_patches_list = [p.shape[0] for p in pixel_values_list]
        pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()

        navigation_commands = ['turn left', 'go straight', 'turn right']
        command_indices = torch.argmax(high_command_one_hot, dim=-1)
        command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

        questions = []
        batch_size = high_command_one_hot.shape[0]
        for i in range(batch_size):
            history_trajectory_sample = history_trajectory[i]
            command_str_sample = command_str_list[i]

            history_str = ' '.join([
                f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                f'{format_number(history_trajectory_sample[j, 1].item())}, '
                f'{format_number(history_trajectory_sample[j, 2].item())})'
                for j in range(history_trajectory_sample.shape[0])
            ])
                                    
            prompt = (
                "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                "1. Visual perception from front camera view\n"
                f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                f"3. Active navigation command: [{command_str_sample.upper()}]"
            )     # TODO 【TOEXP】

            output_requirements = (
                "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                "- Each point format: (x:float, y:float, heading:float)\n"
                "- Use [PT, ...] to encapsulate the trajectory\n"
                "- Maintain numerical precision to 2 decimal places"
            )   # TODO 【TOEXP】

            questions.append(f"{prompt}{output_requirements}")

        # ---------------------------- ROLLOUT ------------------------------
        # Generate G text responses 
        # Run Diffusion Planner 
        # Score PDM
        # -------------------------------------------------------------------
        all_policy_log_probs_old = []   # [G, B] - log probs at generation time 
        all_rewards = []    # [G, B] - PDM scores 

        with torch.no_grad():
            # VLM forward once for hidden states -> diffusion planner 
            # This hidden state is shared across all G rollouts 
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Prepare Diffusion Planner Input
                fwd_output = self.agent.vlm.forward(
                    pixel_values_cat, 
                    questions, 
                    num_patches_list=num_patches_list,
                )
            last_hidden_states = fwd_output.hidden_states[-1]

            status_feature = features["status_feature"].cuda()
            if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
            if last_hidden_states.ndim == 2: last_hidden_states = last_hidden_states.unsqueeze(0)

            history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
            state_input = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

            diff_dtype = next(self.agent.action_head.parameters()).dtype
            diff_input = BatchFeature({
                "state": state_input.to(diff_dtype),
                "his_traj": history_trajectory_reshaped.to(diff_dtype),
                "status_feature": status_feature.to(diff_dtype)
            }
            )  

            for g in range(self.G):
                
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Generate text action
                    gen_output = self.agent.vlm.generate_text_actions(
                        pixel_values_cat, 
                        questions, 
                        num_patches_list=num_patches_list,
                        max_new_tokens=64,
                    )
                    print("检查五：成功")

                    #  Compute log prob of this generation (old policy)
                    old_log_probs = compute_response_logprobs(
                        model = self.agent.vlm.model,
                        pixel_values=pixel_values_cat,
                        generation_output=gen_output
                    )
                    print("检查六：old log probs: ")
                    print(old_log_probs)
                    
                    all_policy_log_probs_old.append(old_log_probs)

                    # Run diffusion planner with shared hidden state 
                    # NOTE: use SAME hidden states for all G rollouts 
                    # Diversity comes from text generation, not diffusion noise

                    actions = self.agent.action_head.get_action(
                        last_hidden_states.to(diff_dtype),
                        diff_input
                    )   
                    print("输出动作检查：")
                    print(actions)

                    # Score
                    # TODO 01 compatible 
                    # 02 change into binary reward 
                    
                    




  

        loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        return loss

        # if logging_prefix == 'train':
        #     predictions = self.agent.compute_loss(features, targets, prediction)

        #     loss = predictions.loss
        #     reward = predictions.reward
        #     policy_loss = predictions.policy_loss
        #     bc_loss = predictions.bc_loss
        #     self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #     self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #     self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #     self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # else:
        #     prediction = self.agent.forward(features,targets)
        #     loss = self.agent.compute_loss(features, targets, prediction)
        #     self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss

    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, 
                      batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
                      batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")
        

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        # """
        # Step called on validation samples
        # :param batch: tuple of dictionaries for feature and target tensors (batched)
        # :param batch_idx: index of batch (ignored)
        # :return: scalar loss
        # """

        loss = torch.tensor(0.0, device=self.device)
        
        self.log("val/loss", loss)
        # return self._step(batch, "val")


    def configure_optimizers(self):
        print('Configure Optimizers ...')
        return self.agent.get_optimizers()
