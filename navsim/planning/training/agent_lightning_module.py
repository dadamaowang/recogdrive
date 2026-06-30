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

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import torch.nn.functional as F 

from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.abstract_agent import AbstractAgent

from navsim.agents.negdrive.utils.internvl_preprocess import load_image
from navsim.agents.negdrive.utils.utils import format_number
from navsim.agents.negdrive.negdrive_backbone import NegDriveGenOutput

from omegaconf import DictConfig, OmegaConf

import os
from pathlib import Path

from navsim.common.dataloader import MetricCacheLoader

# PDM sub-metrics logged during validation (excluding aggregate score).
VAL_PDM_SUBMETRICS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
]

# NSR: outcome advantage for safety failures (NC or DAC below perfect score).
NSR_FAILURE_ADVANTAGE = -1.0
PDM_PERFECT_SCORE = 1.0
BINARY_SAFE_REWARD = 1.0
BINARY_UNSAFE_REWARD = -1.0


@dataclass
class RolloutBundle:
    """Shared rollout artifacts for NSR and GRPO training steps."""
    all_gen_output: List[NegDriveGenOutput]
    pixel_values_cat: torch.Tensor
    questions: List[str]
    num_patches_list: List[int]
    history_trajectory: torch.Tensor
    diff_input: BatchFeature
    diff_dtype: torch.dtype
    nc_tensor: torch.Tensor
    dac_tensor: torch.Tensor


def compute_binary_safety_rewards(
    nc_tensor: torch.Tensor,
    dac_tensor: torch.Tensor,
) -> torch.Tensor:
    """Map NC/DAC scores to binary rewards: +1 (both perfect) or -1 (otherwise)."""
    safe_mask = (nc_tensor >= PDM_PERFECT_SCORE) & (dac_tensor >= PDM_PERFECT_SCORE)
    rewards = torch.full_like(nc_tensor, BINARY_UNSAFE_REWARD)
    rewards[safe_mask] = BINARY_SAFE_REWARD
    return rewards


def compute_grpo_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
    skip_zero_std_groups: bool = True,
    clip_lower_q: float = 0.0,
    clip_upper_q: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group-relative advantages within each scene's G rollouts.

    Returns:
        advantages: [B, G]
        valid_group_mask: [B] True when the group has non-zero std and contributes to loss
    """
    with torch.no_grad():
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean_r) / (std_r + eps)

        if skip_zero_std_groups:
            valid_group_mask = std_r.squeeze(1) > eps
            advantages = advantages * valid_group_mask.unsqueeze(1).to(advantages.dtype)
        else:
            valid_group_mask = torch.ones(rewards.shape[0], dtype=torch.bool, device=rewards.device)

        if clip_lower_q > 0.0 or clip_upper_q < 1.0:
            adv_min = torch.quantile(advantages.reshape(-1), clip_lower_q)
            adv_max = torch.quantile(advantages.reshape(-1), clip_upper_q)
            advantages = advantages.clamp(min=adv_min, max=adv_max)

    return advantages, valid_group_mask


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

    NOTE: Current NSR training uses constant NSR_FAILURE_ADVANTAGE in _step()
    instead of this helper. Wire here when enabling PSR/weighted modes.

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

    # print("检查文字 reasoning 之后的 mask ratio: ")
    #
    # mask_ratio = attention_mask.float().mean().item()
    # print(f"Attention mask ratio (non-padding tokens): {mask_ratio:.4f} (100%=no padding, 0%=all padding)")
    #
    # seq_lengths = attention_mask.sum(dim=1).tolist()
    # print(f"Sequence lengths (non-padding tokens) per batch item: {seq_lengths}")
    pass





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
        self.rl_algorithm = getattr(agent, "rl_algorithm", "nsr")
        self.rollout_temperature = float(getattr(agent, "rollout_temperature", 0.7))
        self.val_do_sample = bool(getattr(agent, "val_do_sample", False))

        grpo_cfg = getattr(agent, "grpo_cfg", None) or {}
        if isinstance(grpo_cfg, DictConfig):
            grpo_cfg = OmegaConf.to_container(grpo_cfg, resolve=True)
        self.grpo_advantage_eps = float(grpo_cfg.get("advantage_eps", 1e-8))
        self.grpo_skip_zero_std_groups = bool(grpo_cfg.get("skip_zero_std_groups", True))
        self.grpo_clip_lower_q = float(grpo_cfg.get("clip_advantage_lower_quantile", 0.0))
        self.grpo_clip_upper_q = float(grpo_cfg.get("clip_advantage_upper_quantile", 1.0))

        self.cfg = cfg
        self.val_metric_cache_loader = None
        if cfg is not None:
            val_subset = cfg.get("val_scene_subset", "full")
            if val_subset == "high_risk_test":
                val_cache_path = OmegaConf.select(
                    cfg, "validation.metric_cache_path", default=None
                )
                if val_cache_path:
                    self.val_metric_cache_loader = MetricCacheLoader(
                        Path(val_cache_path)
                    )

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


    def _get_val_metric_cache_loader(self) -> MetricCacheLoader:
        if self.val_metric_cache_loader is not None:
            return self.val_metric_cache_loader
        return self.agent.action_head.metric_cache_loader

    def _log_validation_pdm_metrics(
        self,
        score: Tensor,
        details: Dict[str, Tensor],
    ) -> None:
        """Log PDM total score and sub-metrics (single rollout per scene)."""
        self.log(
            "val/pdm_score",
            score.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for metric_name in VAL_PDM_SUBMETRICS:
            if metric_name not in details:
                continue
            self.log(
                f"val/{metric_name}",
                details[metric_name].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):

        features, targets, tokens_list = batch
        pixel_values_cat, questions, num_patches_list, history_trajectory = self.agent.unpack_features_cot_prompt(features)
        diff_dtype, diff_input = self.get_diff_input(features, history_trajectory)
        val_cache_loader = self._get_val_metric_cache_loader()

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen_output = self.agent.vlm.generate_text_actions(
                    pixel_values_cat,
                    questions,
                    num_patches_list=num_patches_list,
                    max_new_tokens=self.max_gen_text_tokens,
                    do_sample=self.val_do_sample,
                    temperature=self.rollout_temperature,
                )

                fwd_output = self.agent.vlm.forward_with_ids(
                    pixel_values_cat,
                    gen_output.full_ids,
                    gen_output.attention_mask,
                )

                last_hidden_states = fwd_output.hidden_states[-1].clone()
                del fwd_output
                if last_hidden_states.ndim == 2:
                    last_hidden_states = last_hidden_states.unsqueeze(0)

            actions = self.agent.action_head.get_action(
                last_hidden_states.to(diff_dtype),
                diff_input,
                attention_mask=gen_output.attention_mask,
            )

            score, details = self.agent.action_head.get_grpo_reward(
                actions,
                tokens_list=tokens_list,
                metric_cache_loader=val_cache_loader,
                return_details=True,
            )
            del actions, gen_output

        self._log_validation_pdm_metrics(score, details)

        del score, details
        torch.cuda.empty_cache()



    def _step(self,
              batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
              logging_prefix: str) -> bool:
        """Dispatch to NSR or GRPO training step."""
        if self.rl_algorithm == "nsr":
            return self._training_step_nsr(batch, logging_prefix)
        if self.rl_algorithm == "grpo":
            return self._training_step_grpo(batch, logging_prefix)
        raise ValueError(
            f"Unknown rl_algorithm: {self.rl_algorithm!r}. Expected 'nsr' or 'grpo'."
        )

    def _collect_rollouts(
        self,
        features: Dict[str, Tensor],
        tokens_list: List[str],
        logging_prefix: str,
    ) -> RolloutBundle:
        """Run G VLM rollouts and collect NC/DAC metrics for each sample."""
        pixel_values_cat, questions, num_patches_list, history_trajectory = (
            self.agent.unpack_features_cot_prompt(features)
        )
        diff_dtype, diff_input = self.get_diff_input(features, history_trajectory)

        all_gen_output: List[NegDriveGenOutput] = []
        all_nc: List[torch.Tensor] = []
        all_dac: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.G):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gen_output = self.agent.vlm.generate_text_actions(
                        pixel_values_cat,
                        questions,
                        num_patches_list=num_patches_list,
                        max_new_tokens=self.max_gen_text_tokens,
                        do_sample=True,
                        temperature=self.rollout_temperature,
                    )
                    all_gen_output.append(gen_output)

                    fwd_output = self.agent.vlm.forward_with_ids(
                        pixel_values_cat,
                        gen_output.full_ids,
                        gen_output.attention_mask,
                    )

                last_hidden_states = fwd_output.hidden_states[-1].clone()
                del fwd_output
                if last_hidden_states.ndim == 2:
                    last_hidden_states = last_hidden_states.unsqueeze(0)

                vl_features_mean = last_hidden_states.float().mean().item()
                vl_features_std = last_hidden_states.float().std().item()
                self.log(
                    f"{logging_prefix}/vl_features_mean",
                    vl_features_mean,
                    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/vl_features_std",
                    vl_features_std,
                    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
                )

                actions = self.agent.action_head.get_action(
                    last_hidden_states.to(diff_dtype),
                    diff_input,
                    attention_mask=gen_output.attention_mask,
                )
                del last_hidden_states

                self.log(
                    f"{logging_prefix}/vl_embeds_mean",
                    actions["vl_embeds_mean"],
                    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/vl_embeds_std",
                    actions["vl_embeds_std"],
                    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/vl_embeds_norm",
                    actions["vl_embeds_norm"],
                    on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
                )

                _, details = self.agent.action_head.get_grpo_reward(
                    actions,
                    tokens_list=tokens_list,
                    return_details=True,
                )
                all_nc.append(details["no_at_fault_collisions"].cpu())
                all_dac.append(details["drivable_area_compliance"].cpu())
                del actions, details

        torch.cuda.empty_cache()

        nc_tensor = torch.stack(all_nc, dim=1).float().to(self.device)
        dac_tensor = torch.stack(all_dac, dim=1).float().to(self.device)

        return RolloutBundle(
            all_gen_output=all_gen_output,
            pixel_values_cat=pixel_values_cat,
            questions=questions,
            num_patches_list=num_patches_list,
            history_trajectory=history_trajectory,
            diff_input=diff_input,
            diff_dtype=diff_dtype,
            nc_tensor=nc_tensor,
            dac_tensor=dac_tensor,
        )

    def _log_safety_rollout_metrics(
        self,
        nc_tensor: torch.Tensor,
        dac_tensor: torch.Tensor,
        logging_prefix: str,
    ) -> None:
        """Log shared NC/DAC statistics from rollout phase."""
        self.log(
            f"{logging_prefix}/nc_fail_rate",
            (nc_tensor < PDM_PERFECT_SCORE).float().mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
        )
        self.log(
            f"{logging_prefix}/dac_fail_rate",
            (dac_tensor < PDM_PERFECT_SCORE).float().mean(),
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
        )

    def _compute_mean_response_log_prob(
        self,
        b: int,
        g: int,
        bundle: RolloutBundle,
    ) -> torch.Tensor:
        """Forward a single (b, g) sample and return mean token log-prob of the response."""
        gen_output_g = bundle.all_gen_output[g]
        gen_output_single = NegDriveGenOutput(
            full_ids=gen_output_g.full_ids[b:b + 1],
            attention_mask=gen_output_g.attention_mask[b:b + 1],
            response_start_idx=gen_output_g.response_start_idx,
            text_actions=None,
        )

        start_patch = sum(bundle.num_patches_list[:b])
        end_patch = start_patch + bundle.num_patches_list[b]
        pv_single = bundle.pixel_values_cat[start_patch:end_patch]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            token_lp_b, eos_mask_b = compute_response_logprobs_tokens(
                model=self.agent.vlm.model,
                pixel_values=pv_single,
                generation_output=gen_output_single,
            )

        num_tokens_b = eos_mask_b.sum().clamp(min=1)
        return (token_lp_b * eos_mask_b).sum() / num_tokens_b

    def _run_distributed_manual_updates(
        self,
        update_samples: List[Optional[Tuple]],
        bundle: RolloutBundle,
        logging_prefix: str,
        loss_log_key: str,
        empty_skip_message: str,
    ) -> bool:
        """
        Run per-sample manual backward passes with DDP synchronization.

        Each item in update_samples is either None (dummy backward) or
        (b, g, advantage_scalar).
        Returns True if the optimizer step should be skipped.
        """
        num_updates_local = sum(1 for sample in update_samples if sample is not None)

        any_updates_tensor = torch.tensor(float(num_updates_local > 0), device=self.device)
        torch.distributed.all_reduce(any_updates_tensor, op=torch.distributed.ReduceOp.SUM)

        if any_updates_tensor.item() == 0:
            torch.cuda.empty_cache()
            print(empty_skip_message)
            return True

        max_updates_tensor = torch.tensor(float(num_updates_local), device=self.device)
        torch.distributed.all_reduce(max_updates_tensor, op=torch.distributed.ReduceOp.MAX)
        num_iterations = int(max_updates_tensor.item())

        padded_samples = list(update_samples)
        while len(padded_samples) < num_iterations:
            padded_samples.append(None)

        global_denominator_tensor = torch.tensor(float(num_updates_local), device=self.device)
        torch.distributed.all_reduce(global_denominator_tensor, op=torch.distributed.ReduceOp.SUM)
        global_denominator = max(global_denominator_tensor.item(), 1.0)

        total_pg_loss = 0.0
        for sample in padded_samples:
            if sample is None:
                dummy = sum(
                    p.sum() * 0.0 for p in self.agent.vlm.parameters() if p.requires_grad
                )
                self.manual_backward(dummy)
                torch.cuda.empty_cache()
                continue

            b, g, advantage = sample
            mean_log_prob_b = self._compute_mean_response_log_prob(b, g, bundle)
            loss_b = -(advantage * mean_log_prob_b) / global_denominator
            self.manual_backward(loss_b)
            total_pg_loss += loss_b.item()
            del mean_log_prob_b, loss_b
            torch.cuda.empty_cache()

        self.log(
            loss_log_key,
            total_pg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return False

    def _cleanup_rollout_bundle(self, bundle: RolloutBundle) -> None:
        del bundle.all_gen_output
        del bundle.pixel_values_cat
        del bundle.questions
        del bundle.num_patches_list
        del bundle.history_trajectory
        del bundle.diff_input
        del bundle.nc_tensor
        del bundle.dac_tensor
        torch.cuda.empty_cache()

    def _training_step_nsr(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ) -> bool:
        """NSR: update only on NC/DAC failure rollouts with fixed advantage -1."""
        features, _targets, tokens_list = batch
        bundle = self._collect_rollouts(features, tokens_list, logging_prefix)

        nc_tensor = bundle.nc_tensor
        dac_tensor = bundle.dac_tensor
        failure_mask = (nc_tensor < PDM_PERFECT_SCORE) | (dac_tensor < PDM_PERFECT_SCORE)

        num_failures = failure_mask.sum().item()
        print(f"负样本数目：{num_failures}")
        self.log(
            f"{logging_prefix}/num_failures",
            int(num_failures),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_safety_rollout_metrics(nc_tensor, dac_tensor, logging_prefix)

        failure_samples: List[Optional[Tuple[int, int, float]]] = []
        batch_size = failure_mask.shape[0]
        for g in range(self.G):
            for b in range(batch_size):
                if failure_mask[b, g]:
                    failure_samples.append((b, g, NSR_FAILURE_ADVANTAGE))

        skipped = self._run_distributed_manual_updates(
            update_samples=failure_samples,
            bundle=bundle,
            logging_prefix=logging_prefix,
            loss_log_key=f"{logging_prefix}/pg_loss",
            empty_skip_message="没有负样本",
        )
        self._cleanup_rollout_bundle(bundle)
        return skipped

    def _training_step_grpo(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ) -> bool:
        """GRPO: group-relative updates using binary NC/DAC rewards (+1 / -1)."""
        features, _targets, tokens_list = batch
        bundle = self._collect_rollouts(features, tokens_list, logging_prefix)

        nc_tensor = bundle.nc_tensor
        dac_tensor = bundle.dac_tensor
        rewards = compute_binary_safety_rewards(nc_tensor, dac_tensor)

        advantages, valid_group_mask = compute_grpo_group_advantages(
            rewards,
            eps=self.grpo_advantage_eps,
            skip_zero_std_groups=self.grpo_skip_zero_std_groups,
            clip_lower_q=self.grpo_clip_lower_q,
            clip_upper_q=self.grpo_clip_upper_q,
        )

        num_valid_groups = int(valid_group_mask.sum().item())
        num_zero_std_groups = int((~valid_group_mask).sum().item())
        self.log(
            f"{logging_prefix}/grpo_valid_groups",
            num_valid_groups,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            f"{logging_prefix}/grpo_zero_std_groups",
            num_zero_std_groups,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            f"{logging_prefix}/grpo_mean_reward",
            rewards.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self._log_safety_rollout_metrics(nc_tensor, dac_tensor, logging_prefix)

        update_samples: List[Optional[Tuple[int, int, float]]] = []
        batch_size, group_size = advantages.shape
        for b in range(batch_size):
            if not valid_group_mask[b]:
                continue
            for g in range(group_size):
                adv = advantages[b, g].item()
                if abs(adv) <= self.grpo_advantage_eps:
                    continue
                update_samples.append((b, g, adv))

        num_updates = len(update_samples)
        print(f"GRPO 更新样本数目：{num_updates}")
        self.log(
            f"{logging_prefix}/grpo_num_updates",
            num_updates,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        skipped = self._run_distributed_manual_updates(
            update_samples=update_samples,
            bundle=bundle,
            logging_prefix=logging_prefix,
            loss_log_key=f"{logging_prefix}/grpo_pg_loss",
            empty_skip_message="没有 GRPO 更新样本",
        )
        self._cleanup_rollout_bundle(bundle)
        return skipped

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



# Save checkpoint + LoRA after every validation, named by training step (iter_{step}).
class IterLoRAModelCheckpoint(Callback):
    def __init__(self, dirpath: Optional[str] = None):
        self.dirpath = dirpath

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return

        # Baseline validation before training (global_step=0): log only, no checkpoint.
        if trainer.global_step == 0:
            return

        dirpath = self.dirpath or os.path.join(trainer.default_root_dir, "checkpoints")
        os.makedirs(dirpath, exist_ok=True)

        step = trainer.global_step
        ckpt_name = f"iter_{step}"
        # ckpt_path = os.path.join(dirpath, f"{ckpt_name}.ckpt")
        # trainer.save_checkpoint(ckpt_path)

        lora_dir = os.path.join(dirpath, f"{ckpt_name}_lora")
        os.makedirs(lora_dir, exist_ok=True)
        pl_module.agent.vlm.model.language_model.save_pretrained(lora_dir)

        print(f"[LoRA SAVED] {lora_dir}")