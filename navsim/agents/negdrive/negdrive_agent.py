# -*- encoding: utf-8 -*-
'''
@File    :   negdrive_agent.py
@Time    :   2026/01/04 10:32:54
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   DOING
@Desc    :   None
'''

import pdb

from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image 
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs

from .negdrive_features import NegDriveFeatureBuilder, NegDriveTrajectoryTargetBuilder

from .negdrive_backbone import NegDriveBackbone
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig



class NegDriveAgent(AbstractAgent):

    def __init__(   # TODO: init 的参数根据具体情况设定
        self,
        *,
        trajectory_sampling: TrajectorySampling,    # TODO

        # ========== VLM (POLICY) ==========
        vlm_path: str,
        vlm_type: str = "internvl",
        vlm_size: str = "large",
        train_vlm: bool = True,                 
        cache_hidden_state: bool = False,       # TODO must be False for RL
        cache_mode: bool = False,   # TODO cache_mode ? 

        # ========== DIFFUSION (DECODER) ==========
        dit_type: str = "small",
        sampling_method: str = "ddim",
        freeze_diffusion: bool = True,           
        diff_path: Optional[str] = None,

        # ========== RL / GRPO ==========
        use_grpo: bool = True,
        metric_cache_path: Optional[str] = None,    # TODO 原本给 diffu 的，可能不需要
        reward_scale: float = 1.0,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,

        # ========== OPTIM ==========
        lr: float = 1e-5,

        # ========== RUNTIME ==========
        device: Optional[str] = None,
    ):
        super().__init__()

        # -----------------------
        # core attributes
        # -----------------------
        self._trajectory_sampling = trajectory_sampling     # TODO
        self.device = device or f"cuda:{int(os.getenv('LOCAL_RANK', 0))}"

        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state

        # -----------------------
        # VLM (policy network)
        # -----------------------
        if self.cache_hidden_state or self.cache_mode:  # TODO
            raise ValueError(
                "cache_hidden_state=True or cache_mode=True is incompatible with VLM RL training "
            )
        
        self.vlm_path = vlm_path
        self.vlm_type = vlm_type    # TODO
        self.vlm_size = vlm_size    # TODO
        self.train_vlm = train_vlm
        self.grpo = use_grpo    # TODO

        self.vlm = NegDriveBackbone(     # TODO ori: ReCogDriveBackbone
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
        )
        
        # -----------------------
        # Diffusion planner (frozen)
        # -----------------------
        self.freeze_diffusion = freeze_diffusion
        self.dit_type = dit_type

        # self.metric_cache_path = metric_cache_path TODO 

        if self.freeze_diffusion:
            input_dim = 1536 if vlm_size == "large" else 384
            cfg = make_diffusion_planner_config(   
                    self.dit_type, 
                    action_dim=3, 
                    action_horizon=8, 
                    grpo=False, 
                    input_embedding_dim=input_dim,
                    sampling_method=sampling_method
                    )
            print("检查Diff-01: Config 成功")
            
            self.action_head = ReCogDriveDiffusionPlanner(cfg).to(self.device)
            print("检查Diff-02: 初始化")

            for p in self.action_head.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError

        # # optional planner checkpoint
        # self.checkpoint_path = diff_checkpoint_path  # TODO
        # # if checkpoint_path: 
        # #     self._load_planner_checkpoint(checkpoint_path)  # TODO

        # # -----------------------
        # # GRPO / RL parameters
        # # -----------------------
        # self.reward_scale = reward_scale
        # self.entropy_coef = entropy_coef
        # self.kl_coef = kl_coef

        # self.reference_policy_checkpoint = reference_policy_checkpoint
        # self.metric_cache_path = metric_cache_path  # TODO

        # -----------------------
        # optimizer
        # -----------------------
        self._lr = lr

        # # -----------------------
        # # others TOOD
        # # -----------------------
        # self.num_inference_samples = 1
        # self.inference_selection_mode = "median"


    def name(self) -> str:
        return self.__class__.__name__


    def initialize(self) -> None:   # TODO 
        """
        Initialize agent components from checkpoints.

        Semantics:
        - VLM checkpoint → trainable policy
        - Diffusion checkpoint → frozen decoder
        - GRPO reference policy → loaded separately
        """
        pass
        # if self.checkpoint_path:
        #     ckpt = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
        #     model_dict = self.state_dict()
        #     filtered_ckpt = {}
        #     for k, v in ckpt.items():
        #         k2 = k[len("agent."):] if k.startswith("agent.") else k
        #         if k2 in model_dict and v.shape == model_dict[k2].shape:
        #             filtered_ckpt[k2] = v
        #     self.load_state_dict(filtered_ckpt, strict=False)


    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])


    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [NegDriveTrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]


    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [NegDriveFeatureBuilder(   
            cache_hidden_state=self.cache_hidden_state,
            # model_type=self.vlm_type,  # TODO 
            # checkpoint_path=self.vlm_path,
            model_type=None,
            checkpoint_path=None,
            device=self.device,
            cache_mode=self.cache_mode,
        )]


    def forward(self, 
                features: Dict[str, torch.Tensor],
                targets = None, 
                tokens_list = None
                ) -> Dict[str, torch.Tensor]:
        
        dtype = next(self.vlm.parameters()).type()
        

                # -------------------------------------------------
        # Build diffusion inputs (NO GRAD)
        # -------------------------------------------------
        diff_dtype = next(self.action_head.parameters()).dtype
        action_inputs = BatchFeature({
            "state": input_state.to(diff_dtype),
            "his_traj": history_trajectory_reshaped.to(diff_dtype),
            "status_feature": status_feature.to(diff_dtype)
        }
        )
        with torch.no_grad():
            actions = self.action_head.get_action(
                last_hidden_state.to(diff_dtype), 
                action_inputs
            )        
    #     if self.training and not self.grpo:   # TODO 有必要再弄可配置的
    #         action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
    #         return self.action_head(last_hidden_state, action_inputs)
    #     elif self.training and self.grpo:
    #         action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
    #         return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
    #     else: 
    #         action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
    #         return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

        # -------------------------------------------------
        # TRAINING: GRPO loss on VLM
        # -------------------------------------------------  
        if self.training and self.grpo:     # TODO self.training flag 在哪里 tag 
            rewards = self._compute_vlm_rlvr_reward(    # TODO compute rlvr reward, 结合 ne reinforce 
                traj_outputs = actions,
                targets = targets,
            )

            # TODO 拿出去 positive 的

            grpo_loss = self._compute_vlm_grpo_loss(    # TODO compute GRPO 
                log_probs = log_probs,
                rewards = rewards,
                entropy = entropy,
                tokens = tokens_list
            )

            return {    # TODO return 的一致性
            "loss": grpo_loss,
            "reward": rewards.mean(),
            "policy_loss": grpo_loss,
            "entropy": entropy.mean(),
            "pred_traj": actions["pred_traj"],
            }

        elif self.training and not self.grpo:
            raise NotImplementedError

        # -------------------------------------------------
        # Eval
        # -------------------------------------------------         
        return actions  # TODO 检查跟原输出的一致性


    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
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


    def compute_trajectory(self, 
                           agent_input: AgentInput  # TODO AgentInput
                           ) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():    # TODO get_feature_builders()
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)    # TODO Trajectory


    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, 
                     features: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor], 
                     predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        For pl;  
        
        TODO check 
        """
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss 
        else:
            return torch.nn.functional.l1_loss(
                predictions["pred_traj"],
                targets["trajectory"]
            )


    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        """for pl
        """
        optimizer_cfg = DictConfig(dict(type="AdamW", 
                                        lr=self._lr, 
                                        weight_decay=1e-4, 
                                        betas=(0.9, 0.95))
                                        )
        optimizer = build_from_configs(optim, 
                                       optimizer_cfg, 
                                       params=self.vlm.parameters())    
        # TODO this is for full VLM tuning,
        # for LoRA , adapter ?  
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer,    # TODO 这个是啥东西
                                    lr=self._lr, 
                                    min_lr=0.0, 
                                    epochs=10, 
                                    warmup_epochs=0
                                    )
        else:
            raise NotImplementedError
            # scheduler = WarmupCosLR(optimizer=optimizer,    # TODO 另外的
            #                         lr=self._lr, 
            #                         min_lr=1e-6, 
            #                         epochs=200, 
            #                         warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}



def make_diffusion_planner_config(  # TODO change hyper
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig (our diffusion planner head) object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(     
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
