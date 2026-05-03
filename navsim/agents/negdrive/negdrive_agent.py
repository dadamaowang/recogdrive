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
from .negdrive_actionhead import NegDriveDiffusionPlannerConfig, NegDriveDiffusionPlanner



class NegDriveAgent(AbstractAgent):

    def __init__(   
        self,
        *,
        trajectory_sampling: TrajectorySampling,    # TODO

        metric_cache_path: Optional[str] = None,    # for validation 


        # ========== VLM (POLICY) ==========
        vlm_path: str,
        vlm_type: str = "internvl",
        vlm_size: str = "large",       
        vlm_lr: float = 1e-5,        

        # ========== DIFFUSION (DECODER) ==========
        dit_type: str = "small",
        diff_sampling_method: str = "ddim",
        freeze_diffusion: bool = True,           
        diff_path: Optional[str] = None,

        # ========== RL / GRPO ==========
        per_sample_rollout: int = 4,
        reward_scale: float = 1.0,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,


        # ========== RUNTIME ==========
        device: Optional[str] = None,
    ):
        super().__init__()

        # -----------------------
        # core attributes
        # -----------------------
        self._trajectory_sampling = trajectory_sampling     
        self.metric_cache_path = metric_cache_path
        self.device = device or f"cuda:{int(os.getenv('LOCAL_RANK', 0))}"


        # -----------------------
        # VLM (policy network)
        # -----------------------        
        self.vlm_path = vlm_path
        self.vlm_type = vlm_type   
        self.vlm_size = vlm_size    

        self.vlm = NegDriveBackbone(     # TODO ori: ReCogDriveBackbone
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
        )

        
        # -----------------------
        # Diffusion planner (frozen)
        # -----------------------
        self.diff_path = diff_path
        self.freeze_diffusion = freeze_diffusion
        self.dit_type = dit_type
        self.diff_sampling_method = diff_sampling_method

        input_embedding_dim = 384 if self.dit_type == "small" else 1536
        cfg = make_diffusion_planner_config(
            self.dit_type,
            action_dim=3, 
            action_horizon=8,
            grpo=False,
            input_embedding_dim=input_embedding_dim,
            sampling_method=self.diff_sampling_method
        )
        cfg.vlm_size = self.vlm_size 
        cfg.metric_cache_path = self.metric_cache_path

        self.action_head =  NegDriveDiffusionPlanner(cfg).to(self.device)

        if self.freeze_diffusion:
            diff_ckpt = torch.load(self.diff_path, map_location="cpu", weights_only=False)
            state_dict = diff_ckpt["state_dict"]

            # strip "agent.action_head" prefix
            prefix = "agent.action_head."
            cleaned_state_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            missing, unexpected = self.action_head.load_state_dict(
                cleaned_state_dict, strict=False
            )
            real_missing = [k for k in missing if not k.startswith("old_policy")]
            real_unexpected = [k for k in unexpected if not k.startswith("old_policy")]

            assert not real_missing,    f"Missing keys in diffusion planner: {missing}"
            assert not real_unexpected, f"Unexpected keys in diffusion planner: {unexpected}"
            print(f"Diffusion planner loaded successfully ({len(cleaned_state_dict)} keys).")

            for p in self.action_head.parameters():
                p.requires_grad = False

            diff_dtype = torch.bfloat16   # TODO on H800
            self.action_head = self.action_head.to(dtype=diff_dtype)
            
        else:
            raise NotImplementedError

        self._verify_model_dtype(self.vlm, "VLM")
        self._verify_model_dtype(self.action_head, "Diffusion Planner")


        # -----------------------
        # GRPO / RL parameters
        # -----------------------
        self._lr = vlm_lr
        self.per_sample_rollout = per_sample_rollout



        # self.reward_scale = reward_scale
        # self.entropy_coef = entropy_coef
        # self.kl_coef = kl_coef
        # self.reference_policy_checkpoint = reference_policy_checkpoint

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

    def _verify_model_dtype(
        self,
        model,
        model_name,
        allowed_fp32_keywords=None,
        verbose=False,
    ):
        """
        Check whether any FLOAT32 parameters are trainable unexpectedly.

        Typical expected fp32 trainable params:
        - layer norms
        - biases
        - positional embeddings
        - some LoRA params (depending on implementation)

        Args:
            model: nn.Module
            allowed_fp32_keywords: list[str]
                parameter name substrings allowed to stay fp32
            verbose: bool

        Returns:
            suspicious_params: list[dict]
        """

        if allowed_fp32_keywords is None:
            allowed_fp32_keywords = [
                # norm
                "norm",
                "ln",
                "layernorm",

                # bias
                "bias",

                # positional embeddings
                "position_embedding",
                "pos_embed",
                "position_ids",

                # embeddings sometimes intentionally fp32
                "embedding",

                # LoRA (some implementations keep fp32)
                "lora_",
            ]

        suspicious_params = []

        total_params = 0
        total_trainable = 0
        total_fp32_trainable = 0

        print("\n" + "=" * 80)
        print(f"{model_name}-模型精度检查")
        print("=" * 80)

        for name, param in model.named_parameters():

            total_params += param.numel()

            if param.requires_grad:
                total_trainable += param.numel()

            # only care about trainable fp32 params
            if param.requires_grad and param.dtype == torch.float32:

                total_fp32_trainable += param.numel()

                allowed = any(
                    kw.lower() in name.lower()
                    for kw in allowed_fp32_keywords
                )

                info = {
                    "name": name,
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype),
                    "allowed": allowed,
                    "numel": param.numel(),
                }

                if not allowed:
                    suspicious_params.append(info)

                if verbose:
                    status = "OK_ALLOWED" if allowed else "SUSPICIOUS"

                    print(
                        f"[{status}] "
                        f"{name:<100} "
                        f"shape={str(tuple(param.shape)):<25} "
                        f"dtype={param.dtype}"
                    )

        print("\n" + "-" * 80)
        print(f"Total params:                {total_params:,}")
        print(f"Total trainable params:      {total_trainable:,}")
        print(f"Trainable fp32 params:       {total_fp32_trainable:,}")
        print(f"Suspicious fp32 params:      {len(suspicious_params)}")
        print("-" * 80)

        if len(suspicious_params) == 0:
            print("✅ No suspicious trainable fp32 params found.")
        else:
            print("❌ Found suspicious trainable fp32 params!")

        return suspicious_params


    def get_sensor_config(self) -> SensorConfig:
        """暂时不用管"""
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])


    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """暂时不用管"""
        return [NegDriveTrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]


    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """test 的时候可能先 cache, 看情况；"""
        return [NegDriveFeatureBuilder(   
            # cache_hidden_state=False,
        )]

    def forward(self, 
                features: Dict[str, torch.Tensor],
                targets = None, 
                tokens_list = None
                ) -> Dict[str, torch.Tensor]:
        
        pass


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

        scheduler = WarmupCosLR(optimizer=optimizer,  
                                    lr=self._lr, 
                                    min_lr=0.0, 
                                    epochs=10, 
                                    warmup_epochs=0
                                    )       
        
        # if self.grpo:
        #     scheduler = WarmupCosLR(optimizer=optimizer,    # TODO 这个是啥东西
        #                             lr=self._lr, 
        #                             min_lr=0.0, 
        #                             epochs=10, 
        #                             warmup_epochs=0
        #                             )
        # else:
        #     raise NotImplementedError
        #     # scheduler = WarmupCosLR(optimizer=optimizer,    # TODO 另外的
        #     #                         lr=self._lr, 
        #     #                         min_lr=1e-6, 
        #     #                         epochs=200, 
        #     #                         warmup_epochs=3)
            
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
) -> NegDriveDiffusionPlannerConfig:
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

    config = NegDriveDiffusionPlannerConfig(     
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
