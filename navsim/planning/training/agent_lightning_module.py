# -*- encoding: utf-8 -*-
'''
@File    :   agent_lightning_module.py
@Time    :   2026/01/04 10:46:58
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   正在优化
@Desc    :   
'''

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch import Tensor
from typing import Dict, Tuple, List, Any
import torch.nn.functional as F 

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.abstract_agent import AbstractAgent

from navsim.agents.negdrive.utils.internvl_preprocess import load_image
from navsim.agents.negdrive.utils.utils import format_number
from navsim.agents.negdrive.negdrive_backbone import NegDriveGenOutput

from omegaconf import DictConfig

import os


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
    """ TODO 参考 verl 
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


def compute_response_logprobs_tokens(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    generation_output,              # GenerationOutput
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns per-token log probs AND eos_mask for response tokens only.

    Returns:
        token_log_probs: [B, ResponseLen]  per-token log probs
        eos_mask:        [B, ResponseLen]  1=real token, 0=padding
    """
    full_ids       = generation_output.full_ids
    attention_mask = generation_output.attention_mask
    response_start = generation_output.response_start_idx

    B, S   = full_ids.shape
    device = full_ids.device

    num_patches = pixel_values.shape[0]
    image_flags = torch.ones(num_patches, dtype=torch.long, device=device)

    
    memo = get_realtime_vram()
    print(f"【Logprob Memory Check】：before outputs: {memo} ")

    outputs = model(
        pixel_values=pixel_values,
        input_ids=full_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
        use_cache=False,    # 不需要自回归生成，关掉，节省显存
    )

    memo = get_realtime_vram()
    print(f"【Logprob Memory Check】：After outputs, before logits: {memo} ")


    logits = outputs.logits     # [B, S, V]

    memo = get_realtime_vram()
    print(f"【Logprob Memory Check】：After logits, before cross entropy: {memo} ")


    # Causal shift
    shift_logits = logits[:, :-1, :]        # [B, S-1, V]
    shift_ids    = full_ids[:, 1:]          # [B, S-1]
    shift_mask   = attention_mask[:, 1:]    # [B, S-1]

    # ── KEY FIX: slice to response positions BEFORE log_softmax ───────
    # response_start - 1 because of causal shift
    response_start_shifted = max(response_start - 1, 0)

    # Only keep response portion — discard prompt logits entirely
    response_logits = shift_logits[:, response_start_shifted:, :]  # [B, ResponseLen, V]
    response_ids    = shift_ids[:,   response_start_shifted:]       # [B, ResponseLen]
    response_mask   = shift_mask[:,  response_start_shifted:]       # [B, ResponseLen]

    # Free full logits immediately — no longer needed
    del logits, shift_logits, shift_ids, shift_mask

    token_log_probs = -F.cross_entropy(
        input=response_logits.reshape(-1, response_logits.size(-1)),
        target=response_ids.reshape(-1),
        reduction='none'
    ).reshape(response_logits.shape[:-1])  # [B, L_resp]

    memo = get_realtime_vram()
    print(f"【Logprob Memory Check】：After cross entropy: {memo} ")

    eos_mask = response_mask.float()
    token_log_probs = token_log_probs * eos_mask

    del response_logits, response_ids, response_mask

    return token_log_probs, eos_mask    


def compute_negdrive_advantages(
        policy_log_probs_tokens: torch.Tensor,
        rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        gamma: float = 1.0, 
        mode: str = "nsr",
        positive_advantage_weight: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """ TODO 核心
    Compute token-level advantages for NSR/PSR/weighted-REINFORCE.
    Adapted from veRL's compute_psr_nsr_outcome_advantage for your
    sequence-level binary reward setting.

    Args:
        policy_log_probs_tokens: [B, ResponseLen] per-token log probs
        rewards:                 [B] scalar reward per sequence {-1.0, 1.0}
        eos_mask:                [B, ResponseLen] 1=real response token, 0=pad
        gamma:                   discount factor for return computation
                                 1.0 = flat reward across all tokens (recommended)
                                 <1.0 = earlier tokens get less credit
        mode:                    "negative" → learn from failures only (your case)
                                 "positive" → learn from successes only
                                 "weighted" → learn from both
        positive_advantage_weight: how much to weight positive samples in
                                   "weighted" mode. Keep small (0.1) so
                                   negative samples dominate.

    Returns:
        advantages: [B, ResponseLen] token-level advantages
        returns:    [B, ResponseLen] discounted returns
    """    
    with torch.no_grad():
        B, T = eos_mask.shape

        # ── Step 1: Broadcast scalar reward to token level ────────────
        # Each token in the response gets the sequence's reward
        # Shape: [B] → [B, T]
        token_level_rewards = rewards.unsqueeze(-1).expand(B, T) * eos_mask
        # reward only applied at real response tokens, 0 at padding

        # ── Step 2: Compute discounted returns ────────────────────────
        # returns[t] = sum_{k=t}^{T} gamma^(k-t) * reward[k]
        # With gamma=1.0 and outcome reward: returns[t] = reward for all t
        returns = torch.zeros_like(token_level_rewards)    # [B, T]
        running_return = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset running return to 0 after EOS token
            running_return = running_return * eos_mask[:, t]

        # ── Step 3: Identify correct and incorrect sequences ──────────
        correct_idx   = (rewards == 1.0)    # [B] bool — safe trajectories
        incorrect_idx = (rewards == -1.0)   # [B] bool — unsafe trajectories


        # ── Step 4: Compute advantages based on mode ──────────────────
        if mode == "nsr":
            # Only penalize failures — push log_prob DOWN for unsafe text
            # correct:   advantage ≈ 0     → no update
            # incorrect: advantage = -1    → penalize
            advantages = torch.zeros_like(returns)              # [B, T]
            advantages[incorrect_idx] = returns[incorrect_idx] - 1

        elif mode == "psr":
            # Only reward successes — push log_prob UP for safe text
            # correct:   advantage = +1    → reward
            # incorrect: advantage ≈ 0     → no update
            advantages = torch.zeros_like(returns)
            advantages[correct_idx] = returns[correct_idx]

        elif mode == "weighted":
            # Learn from both, but weight positives less
            # correct:   advantage = returns * weight  (dampened positive)
            # incorrect: advantage = returns - 1       (full negative)
            advantages = returns.clone()
            advantages[correct_idx]   *= positive_advantage_weight
            advantages[incorrect_idx] -= 1

        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. "
                "Choose 'negative', 'positive', or 'weighted'."
            )

        # ── Step 5: Apply EOS mask ────────────────────────────────────
        # Zero out padding positions
        advantages = advantages * eos_mask                      # [B, T]  
    
    return advantages, returns


def check_mask_ratio(attention_mask: torch.Tensor):
    """check if input+image have padding"""

    print("检查文字 reasoning 之后的 mask ratio: ")
    
    mask_ratio = attention_mask.float().mean().item()
    print(f"Attention mask ratio (non-padding tokens): {mask_ratio:.4f} (100%=no padding, 0%=all padding)")

    seq_lengths = attention_mask.sum(dim=1).tolist()
    print(f"Sequence lengths (non-padding tokens) per batch item: {seq_lengths}")





class AgentLightningVLMRL(pl.LightningModule):
    """Pytorch lightning wrapper for negdrive agent."""

    def __init__(self, agent: AbstractAgent, cfg: DictConfig = None):
        """【正在优化】
        """
        super().__init__()

        # self.save_hyperparameters(cfg)    # TODO tensorborad 超参这里出问题；后边再解决不是特别重要

        self.agent = agent

        # ------------------------------
        # Set RLVR-related Hyparameters
        # ------------------------------ 
        self.max_gen_text_tokens = agent.max_text_tokens
        self.G = agent.per_sample_rollout
        self.bag_g = agent.bag_g


        self.automatic_optimization = False  # NOTE negdrive 算法的负样本动态优化和不等长梯度特性，要求必须手动优化


    def configure_optimizers(self):
        print('Configure Optimizers ...')

        total_steps = self.trainer.estimated_stepping_batches
        print(f"total training steps : {total_steps}")
        self.agent.set_total_training_steps(total_steps)

        return self.agent.get_optimizers()


    def training_step(self, 
                      batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
                      batch_idx: int) -> Tensor:
        """
        Step called on training samples

        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """

        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()
        skipped = self._step(batch, "train")
        if skipped:
            return
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.agent.vlm.parameters() if p.requires_grad],
            max_norm=1.0,
            )    # TODO 
        # clip_grad_norm_ returns the total norm BEFORE clipping
        self.log("train/grad_norm", grad_norm,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


        opt.step()
        sch.step()


    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):

        features, targets, tokens_list = batch
        pixel_values_cat, questions, num_patches_list, history_trajectory = self.unpack_features(features)
        diff_dtype, diff_input = self.get_diff_input(features, history_trajectory)

        all_rewards = []
        with torch.no_grad():
            for g in range(self.bag_g):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gen_output = self.agent.vlm.generate_text_actions(
                        pixel_values_cat, 
                        questions, 
                        num_patches_list=num_patches_list,
                        max_new_tokens=self.max_gen_text_tokens,  
                    )

                    fwd_output = self.agent.vlm.forward_with_ids(
                        pixel_values_cat,
                        gen_output.full_ids,
                        gen_output.attention_mask
                    )

                    last_hidden_states = fwd_output.hidden_states[-1].clone()
                    del fwd_output
                    if last_hidden_states.ndim == 2: 
                        last_hidden_states = last_hidden_states.unsqueeze(0)

                actions = self.agent.action_head.get_action(
                    last_hidden_states.to(diff_dtype),
                    diff_input
                )   # [B, T, 3]

                reward = self.agent.action_head.get_grpo_reward(
                    actions,
                    tokens_list=tokens_list,
                )   # [B]
                all_rewards.append(reward.cpu())
                del actions, reward, gen_output
        
        # Compute Metrics
        rewards_tensor = torch.stack(
            [r.to(self.device) for r in all_rewards], dim=1
        ).float()   # [B, bag_g]
        del all_rewards

        # Best-of-G reward: if ANY rollout succeeds, count as success
        # This measures the model's exploration ability
        # TODO val 的 reward 选择
        best_of_g_reward  = rewards_tensor.max(dim=1).values   # [B]
        mean_reward = rewards_tensor.mean(dim=1)    # [B]

        self.log("val/best_of_g_reward", best_of_g_reward.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mean_reward", mean_reward.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        del rewards_tensor
        torch.cuda.empty_cache()



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

        pixel_values_cat, questions, num_patches_list, history_trajectory = self.unpack_features(features)
        diff_dtype, diff_input = self.get_diff_input(features, history_trajectory)

        # =============================
        # Rollout
        # =============================
        all_gen_output = [] 
        all_rewards = []    # [G, B] - PDM scores 
        all_logprobs_old_tokens = []    

        with torch.no_grad():            
            for g in range(self.G):  
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Generate text reasoning
                    gen_output = self.agent.vlm.generate_text_actions(
                        pixel_values_cat, 
                        questions, 
                        num_patches_list=num_patches_list,
                        max_new_tokens=self.max_gen_text_tokens,  
                    )
                    all_gen_output.append(gen_output)

                    # Forward pass with full_ids, get last hidden states
                    fwd_output = self.agent.vlm.forward_with_ids(
                        pixel_values_cat,
                        gen_output.full_ids,
                        gen_output.attention_mask
                    )

                    # ── Behavior policy log probs (old policy) ────────────
                    # Computed NOW, inside no_grad, same tokens
                    # This is π_old used in ratio π_θ/π_old
                    old_token_log_probs, eos_mask = compute_response_logprobs_tokens(
                        model=self.agent.vlm.model,
                        pixel_values=pixel_values_cat,
                        generation_output=gen_output,
                    )
                    all_logprobs_old_tokens.append(old_token_log_probs.cpu())

                # get actions from planner 

                check_mask_ratio(gen_output.attention_mask)

                last_hidden_states = fwd_output.hidden_states[-1].clone()
                del fwd_output
                if last_hidden_states.ndim == 2: 
                    last_hidden_states = last_hidden_states.unsqueeze(0)

                # attn_mask_for_last_hidden_states = gen_output.attention_mask.clone()
                # if attn_mask_for_last_hidden_states.ndim == 2:
                #     attn_mask_for_last_hidden_states = attn_mask_for_last_hidden_states.unsqueeze(0)

                # Latent Stability Check 01
                vl_features_mean = last_hidden_states.float().mean().item()
                vl_features_std = last_hidden_states.float().std().item()

                # 转化成 torch.Tensor 不然没法被 log 
                vl_features_mean = BatchFeature(data={"vl_features_mean": vl_features_mean})
                vl_features_std = BatchFeature(data={"vl_features_std": vl_features_std})

                self.log(f"{logging_prefix}/vl_features_mean", vl_features_mean["vl_features_mean"], 
                                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f"{logging_prefix}/vl_features_std", vl_features_std["vl_features_std"], 
                                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)


                actions = self.agent.action_head.get_action(
                    last_hidden_states.to(diff_dtype),
                    diff_input,
                    attention_mask = gen_output.attention_mask,
                )   # [B, T, 3]
                del last_hidden_states

                """
                BatchFeature(data={"pred_traj": final_actions})
                {'pred_traj': tensor([[[ 9.6389e-01,  7.4900e-02,  5.8308e-03],
                [ 1.8365e+00,  2.7319e-02,  8.8125e-03],
                [ 2.5693e+00,  2.9800e-02,  6.6231e-03],
                [ 2.9645e+00,  3.3642e-02,  6.5169e-03],
                [ 3.2720e+00, -2.1763e-03,  7.9466e-03],
                [ 3.4912e+00,  2.3960e-02,  1.1656e-03],
                [ 4.1012e+00, -3.6682e-02,  6.9115e-03],
                [ 3.6042e+00, -7.6752e-03,  1.7909e-03]],

                [[ 1.2922e+00,  1.6138e-01,  5.1677e-02],
                [ 2.7716e+00,  1.2371e-01,  8.4622e-02],
                [ 2.6529e+00,  1.1328e-01,  1.0839e-01],
                [ 2.6180e+00,  1.7940e-01,  1.1723e-01],
                [ 2.7896e+00,  1.7507e-01,  1.3273e-01],
                [ 3.1623e+00,  1.4215e-01,  1.3982e-01],
                [ 3.7028e+00,  2.4421e-01,  1.5477e-01],
                [ 3.5862e+00,  2.0090e-01,  1.6826e-01]]], device='cuda:0')
                ============= Other custome Monitor log
                "vl_embeds", etc 
    
                }
                """
                self.log(f"{logging_prefix}/vl_embeds_mean", actions["vl_embeds_mean"],
                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f"{logging_prefix}/vl_embeds_std", actions["vl_embeds_std"],
                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f"{logging_prefix}/vl_embeds_norm", actions["vl_embeds_norm"],
                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

                # get rewards
                reward = self.agent.action_head.get_grpo_reward(
                    actions,
                    tokens_list=tokens_list,
                )   # [B]
                all_rewards.append(reward.cpu())
                del actions, reward
            
                # self._log_vram(f"【显存检查 02- Rollout】after_rollout_g{g}")

        torch.cuda.empty_cache()
        # After torch.cuda.empty_cache() at end of rollout:
        # self._log_vram("【显存检查 03- Rollout Complete】after_rollout_complete")

        # =============================
        # Filter out failues
        # =============================
        rewards_tensor = torch.stack(
            [r.to(self.device) for r in all_rewards], dim=1
        ).float()
        del all_rewards

        """
        Rewards Tensor: 
        tensor([[0., 0., 0.],
                [0., 0., 0.]], device='cuda:0')
        形状：[B, G]
        """

        failure_mask = (rewards_tensor == 0)   # [B, G] bool
        rewards_tensor[failure_mask] = -1     
        """【EXPTODO】
        设计 reward 
        """
        num_failures = failure_mask.sum().item()
        print(f"负样本数目：{num_failures}")
        self.log(f"{logging_prefix}/num_failures", int(num_failures),
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


        # =============================
        # Update on failure samples
        # =============================
        failure_samples = []
        B = rewards_tensor.shape[0]
        for g in range(self.G):
            for b in range(B):
                if failure_mask[b, g]:
                    failure_samples.append((b, g))  

        # Synchronize across ranks
        num_failures_local = len(failure_samples)

        # Check if ANY rank has failures
        any_failures_tensor = torch.tensor(float(num_failures_local > 0), device=self.device)
        torch.distributed.all_reduce(any_failures_tensor, op=torch.distributed.ReduceOp.SUM)    

        if any_failures_tensor.item() == 0:
            # No rank has failures, all skip together
            del all_gen_output, all_logprobs_old_tokens, rewards_tensor, failure_mask
            torch.cuda.empty_cache()

            print("没有负样本")

            if self.automatic_optimization:
                return self._zero_loss()
            else:
                return True
        
        # Find max failure count across ranks — all ranks loop this many times
        max_failures_tensor = torch.tensor(float(num_failures_local), device=self.device)
        torch.distributed.all_reduce(max_failures_tensor, op=torch.distributed.ReduceOp.MAX)
        num_iterations = int(max_failures_tensor.item())

        # Pad with None so all ranks do same number of backward() calls
        while len(failure_samples) < num_iterations:
            failure_samples.append(None)

        # Use global num_failures for loss normalization (not local)
        global_num_failures_tensor = torch.tensor(float(num_failures_local), device=self.device)
        torch.distributed.all_reduce(global_num_failures_tensor, op=torch.distributed.ReduceOp.SUM)
        global_num_failures = max(global_num_failures_tensor.item(), 1)


        total_loss    = torch.tensor(0.0, device=self.device)
        total_pg_loss = 0.0
        for sample in failure_samples:

            if sample is None:
                # TODO 之后添加了 flash attention 还是得手动回传梯度，所以后边都手动吧

                # Dummy backward to stay synchronized with other ranks
                dummy = sum(p.sum() * 0.0 for p in self.agent.vlm.parameters()
                    if p.requires_grad)
                self.manual_backward(dummy)
                torch.cuda.empty_cache()
                continue


            (b, g) = sample
            # Before failure sample forward pass:
            self._log_vram(f"【显存检查 04- Failure Sample】before_forward_b{b}_g{g}")

            gen_output_g = all_gen_output[g]
            old_lp_bg = all_logprobs_old_tokens[g][b:b+1].to(self.device) # [1, ResponseLen]

            gen_output_single = NegDriveGenOutput(
                full_ids=gen_output_g.full_ids[b:b+1],  # [1, SeqLen]
                attention_mask=gen_output_g.attention_mask[b:b+1],  # [1, SeqLen]
                response_start_idx=gen_output_g.response_start_idx,    # scalar, unchanged 
                text_actions=None
            )
            
            # ── Slice pixel_values for this sample only ───────────────────────
            # pixel_values_cat is [TotalPatches, C, H, W]
            # need to extract patches belonging to sample b
            start_patch = sum(num_patches_list[:b])
            end_patch   = start_patch + num_patches_list[b]
            pv_single   = pixel_values_cat[start_patch:end_patch]    # [NumPatches_b, C, H, W]

            # ── Forward pass: B=1, avoids OOM ────────────────────────────────
            with torch.autocast("cuda", dtype=torch.bfloat16):
                token_lp_b, eos_mask_b = compute_response_logprobs_tokens(
                    model=self.agent.vlm.model,
                    pixel_values=pv_single,               # [NumPatches_b, C, H, W]
                    generation_output=gen_output_single,  # B=1
                )  # [1, ResponseLen]  ← has grad_fn      

            # After compute_response_logprobs_tokens:
            self._log_vram(f"【显存检查 05- Failure Sample】after_forward_b{b}_g{g}")

            # NSR loss
            log_ratio_b     = token_lp_b - old_lp_bg.detach()
            ratio_b         = torch.exp(log_ratio_b)
            ratio_clipped_b = torch.clamp(ratio_b, 1 - 0.2, 1 + 0.2)

            per_token_loss_b = torch.max(ratio_b, ratio_clipped_b) * eos_mask_b
            num_tokens_b     = eos_mask_b.sum().clamp(min=1)


            loss_b = per_token_loss_b.sum() / num_tokens_b / global_num_failures   # scaled (1/global_num_failures) loss 
            self.manual_backward(loss_b)
            total_pg_loss += loss_b.item() / global_num_failures

            # After loss computation:
            self._log_vram(f"【显存检查 06- Failure Sample】after_loss_b{b}_g{g}")

            del token_lp_b, eos_mask_b, log_ratio_b, ratio_b, ratio_clipped_b, per_token_loss_b, old_lp_bg, loss_b
            torch.cuda.empty_cache()


        # Final cleanup
        del rewards_tensor, failure_mask, all_logprobs_old_tokens
        del pixel_values_cat, questions, num_patches_list, history_trajectory, diff_input
        try:
            del all_gen_outputs     # free any remaining gen_outputs
        except:
            pass
        torch.cuda.empty_cache()


        # ── Logging ───────────────────────────────────────────────────────
        self.log(f"{logging_prefix}/pg_loss", total_pg_loss,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        

        return False 


    def unpack_features(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str], List[int]]:
        """
        Unpack and prepare features for the VLM.

        Args:
            features: Dictionary containing raw feature tensors.

        """
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
            )   

            output_requirements = (
                "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                "- Each point format: (x:float, y:float, heading:float)\n"
                "- Use [PT, ...] to encapsulate the trajectory\n"
                "- Maintain numerical precision to 2 decimal places"
            )   

            questions.append(f"{prompt}{output_requirements}")
        
        return pixel_values_cat, questions, num_patches_list, history_trajectory



    # def unpack_features(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str], List[int]]:
    #     """
    #     Unpack and prepare features for the VLM.

    #     Args:
    #         features: Dictionary containing raw feature tensors.

    #     """
    #     for key, tensor in features.items():
    #         if isinstance(tensor, torch.Tensor):
    #             features[key] = tensor.cuda()
        
    #     history_trajectory = features["history_trajectory"].cuda()  
    #     if history_trajectory.ndim == 2:
    #         history_trajectory = history_trajectory.unsqueeze(0)

    #     high_command_one_hot = features["high_command_one_hot"].cuda()
    #     if high_command_one_hot.ndim == 1:
    #         high_command_one_hot = high_command_one_hot.unsqueeze(0)
        
    #     image_path_tensor = features["image_path_tensor"]
    #     if image_path_tensor.ndim == 1: image_path_tensor = image_path_tensor.unsqueeze(0)
    #     image_paths = decode_paths_from_tensor(image_path_tensor)

    #     pixel_values_list = [load_image(path) for path in image_paths] 
    #     num_patches_list = [p.shape[0] for p in pixel_values_list]
    #     pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()

    #     navigation_commands = ['turn left', 'go straight', 'turn right']
    #     command_indices = torch.argmax(high_command_one_hot, dim=-1)
    #     command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

    #     questions = []
    #     batch_size = high_command_one_hot.shape[0]
    #     for i in range(batch_size):
    #         history_trajectory_sample = history_trajectory[i]
    #         command_str_sample = command_str_list[i]

    #         history_str = ' '.join([
    #             f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
    #             f'{format_number(history_trajectory_sample[j, 1].item())}, '
    #             f'{format_number(history_trajectory_sample[j, 2].item())})'
    #             for j in range(history_trajectory_sample.shape[0])
    #         ])
                                    
    #         prompt = (
    #             "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
    #             "1. Visual perception from front camera view\n"
    #             f"2. Historical motion context (last 4 timesteps):{history_str}\n"
    #             f"3. Active navigation command: [{command_str_sample.upper()}]"
    #         )   

    #         output_requirements = (
    #             "\nOutput requirements:\n- Predict 8 future trajectory points\n"
    #             "- Each point format: (x:float, y:float, heading:float)\n"
    #             "- Use [PT, ...] to encapsulate the trajectory\n"
    #             "- Maintain numerical precision to 2 decimal places"
    #         )   

    #         questions.append(f"{prompt}{output_requirements}")
        
    #     return pixel_values_cat, questions, num_patches_list, history_trajectory



    def get_diff_input(self, features: Dict[str, torch.Tensor], history_trajectory: torch.Tensor) -> BatchFeature:
        """
        Prepare input for the Diffusion-based planner.

        """
        # extract and prepare planner input
        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        state_input = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        diff_dtype = next(self.agent.action_head.parameters()).dtype
        diff_input = BatchFeature({
                "state": state_input.to(diff_dtype),
                "his_traj": history_trajectory_reshaped.to(diff_dtype),
                "status_feature": status_feature.to(diff_dtype)
            }
        )
        return diff_dtype, diff_input
            

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     """
    #     Load LoRA weights back into the VLM.
    #     Called automatically by Lightning when resuming from checkpoint.
    #     """
    #     lora_state_dict = checkpoint['state_dict']

    #     # Load with strict=False — checkpoint only has LoRA keys,
    #     # not the full model state dict
    #     missing, unexpected = self.agent.vlm.load_state_dict(
    #         lora_state_dict, strict=False
    #     )

    #     # Only real problem is if LoRA keys themselves are missing
    #     lora_missing = [k for k in missing if 'lora_' in k]
    #     if lora_missing:
    #         print(f"WARNING: Missing LoRA keys: {lora_missing}")
    #     else:
    #         print(f"LoRA weights loaded successfully "
    #             f"({len(lora_state_dict)} tensors).")



    def _zero_loss(self) -> torch.Tensor:
        """
        Returns a zero loss that connected to VLM parameters.
        Used when there are no failure samples in a batch.

        This gives the AMP scaler valid inf checks to record,
        while producing zero gradient - effectively a no-op update.

        How it works:
            sum(param * 0) = 0  (scalar, connected to graph)
            gradient = 0        (no actual update happens)
        """
        # # Pick one small LoRA parameter to connect to TODO 
        # # We use next(iter()) to get just one parameter — cheap
        # for name, param in self.agent.vlm.named_parameters():
        #     if param.requires_grad and 'lora_A' in name:
        #         return param.sum() * 0.0   # zero but connected to graph ✅

        # Fallback: use first trainable parameter
        for param in self.agent.vlm.parameters():
            if param.requires_grad:
                return param.sum() * 0.0

        # Should never reach here
        return torch.tensor(0.0, device=self.device, requires_grad=True)


    def _log_vram(self, tag: str):
        """Print VRAM usage at a specific point. Remove after debugging."""
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        max_alloc = torch.cuda.max_memory_allocated() / 1e9
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[rank{rank}][{tag}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={max_alloc:.2f}GB")
        torch.cuda.reset_peak_memory_stats()   # reset peak after each checkpoint


    # def on_validation_epoch_end(self):
    # TODO
    #     # 记录轨迹视频
    #     if self.global_rank == 0:  # 仅主进程记录，避免多卡重复
    #         video_tensor = make_trajectory_video(self.val_predictions)  # [B, C, T, H, W]
    #         self.logger.experiment.add_video(
    #             "val/trajectories", 
    #             video_tensor, 
    #             fps=10, 
    #             global_step=self.global_step
    #         )
            
    #         # 记录文本推理示例
    #         sample_text = self.val_outputs[0]["reasoning"]
    #         self.logger.experiment.add_text(
    #             "val/reasoning_sample", 
    #             sample_text, 
    #             global_step=self.global_step
    #         )



def get_realtime_vram(device=0):
    """Return Current VRAM usage for the specified GPU device."""

    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved  = torch.cuda.memory_reserved(device)

    return {
        "total_gb":   total / 1024**3,
        "free_gb":    free / 1024**3,
        "allocated_gb": allocated / 1024**3,  # 实际被 tensor 占用的显存
        "reserved_gb":  reserved / 1024**3,   # PyTorch 缓存池预留（含碎片）
        "used_pct":   (1 - free / total) * 100
    }


class VRAMMonitor(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        status = get_realtime_vram(pl_module.device.index)

        print(f"Batch 批次：{batch_idx} \
              已用: {status['allocated_gb']:.2f}GB \
              | 剩余: {status['free_gb']:.2f}GB \
                | 利用率: {status['used_pct']:.1f}%")

        pl_module.log("vram/allocated_gb", status["allocated_gb"], prog_bar=True, sync_dist=True)
        pl_module.log("vram/free_gb", status["free_gb"], prog_bar=True, sync_dist=True)
        pl_module.log("vram/used_pct", status["used_pct"], prog_bar=True, sync_dist=True)



class OptimizerHealthMonitor(Callback):
    """

    """
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 频率控制 & DDP 去重
        if trainer.global_step % self.log_every != 0 or trainer.global_rank != 0:
            return

        # 1. 当前学习率
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        # # 2. 全局梯度范数 (只读计算，不修改梯度)
        # grads = [p.grad.detach() for p in pl_module.parameters() if p.grad is not None]
        # grad_norm = torch.stack(grads).norm().item() if grads else 0.0

        # # 3. 是否触发梯度裁剪 (假设 Trainer 设了 gradient_clip_val=1.0)
        # clip_triggered = grad_norm > 1.0

        # 4. 同步到 TensorBoard / W&B
        pl_module.log("optimizer/lr", lr, on_step=True, sync_dist=False)
        # pl_module.log("optimizer/grad_norm", grad_norm, on_step=True, sync_dist=False)
        # pl_module.log("optimizer/clip_triggered", float(clip_triggered), on_step=True, sync_dist=False)



# Custom checkpoint callback that saves LoRA weights alongside Lightning checkpoints.
class LoRAModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        pl_module = trainer.lightning_module
        if getattr(pl_module, "global_rank", 0) != 0:
            return

        ckpt_name = os.path.splitext(os.path.basename(filepath))[0]
        lora_dir = os.path.join(os.path.dirname(filepath), f"{ckpt_name}_lora")
        os.makedirs(lora_dir, exist_ok=True)

        pl_module.agent.vlm.model.language_model.save_pretrained(lora_dir)

        print(f"[LoRA SAVED] {lora_dir}")