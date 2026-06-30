# -*- encoding: utf-8 -*-
'''
@File    :   negdrive_agent.py
@Time    :   2026/01/04 10:32:54
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   【正在优化】
@Desc    :   
'''

# import pdb
import sys 

from typing import Any, List, Dict, Optional, Tuple, Union, Literal
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


from PIL import Image
import base64
import io

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
        trajectory_sampling: TrajectorySampling,    # for building targets 
        metric_cache_path: Optional[str] = None,    # for validation 
        mode: str = "train",   # control certain behaviors (e.g. caching, rollout, etc.) based on mode

        # ========== VLM (POLICY) ==========
        vlm_path: str,
        vlm_lora_path: Optional[str] = None,  
        vlm_type: str = "internvl",
        vlm_size: str = "large",       

        max_text_tokens: int = 128,   # max generated text tokens
        rollout_temperature: float = 0.7,   # training rollout sampling temperature
        val_do_sample: bool = False,   # validation: false = greedy decoding
        pass_cot_token_only: bool = False,  # if True, pass only cot token's last hidden states to planner

        # ========== Optimizers and Schedulers ==========
        vlm_lr: float = 1e-5,
        opt_type: str = "AdamW",
        opt_weight_decay: float = 0.01,
        opt_eps: float = 1e-8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,

        # ========== DIFFUSION (DECODER) ==========
        dit_type: str = "small",
        diff_sampling_method: str = "ddim",
        freeze_diffusion: bool = True,           
        diff_path: Optional[str] = None,

        # ========== RL ==========
        rl_algorithm: str = "nsr",   # nsr | grpo
        grpo_cfg: Optional[Dict[str, Any]] = None,
        per_sample_rollout: int = 4,
        bag_g: int = 4,   # Best-of-G reward
        max_padding_len: int = 2800,   # max token length after padding (for diffusion input) 
        reward_scale: float = 1.0,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,

        # ========== RUNTIME ==========
        device: Optional[str] = None,
    ):
        """【正在优化】
        
        """
        
        super().__init__()

        # -----------------------
        # core attributes
        # -----------------------
        self._trajectory_sampling = trajectory_sampling     
        self.metric_cache_path = metric_cache_path
        self.device = device or f"cuda:{int(os.getenv('LOCAL_RANK', 0))}"

        self.mode = mode

        """NOTE VLM + Diffusion Planner 
        (1) dit_type : main differenct: input_dim, output_dim 
            small : input_embedding_dim = 384
            large : input_embedding_dim = 1536

            dit_type is decoupled with vlm_size

        (2) vlm_size: 
            small: hidden_embedding_dim = 1536 
            large: hidden_embedding_dim = 3584
        """

        # -----------------------
        # VLM (policy network)
        # -----------------------        
        self.vlm_path = vlm_path
        self.vlm_lora_path = vlm_lora_path
        self.vlm_type = vlm_type   
        self.vlm_size = vlm_size    

        self.max_padding_len = max_padding_len

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if self.mode == "train" or self.mode == "recogdrive_eval":  # TODO

            mode = "train" if self.mode == "train" else "eval"

            self.vlm = NegDriveBackbone(    
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                vlm_lora_path=self.vlm_lora_path,
                device=self.device,
                max_padding_len=self.max_padding_len,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                mode=mode,
            )
            self.vlm = self.vlm.to(self.device)


        elif self.mode == "eval": 

            self.vlm = NegDriveBackbone(    
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                vlm_lora_path=self.vlm_lora_path,
                device=self.device,
                max_padding_len=self.max_padding_len,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                mode=mode,
            )
            self.vlm = self.vlm.to(self.device)




        # -----------------------
        # vlm setting
        # -----------------------
        self.max_text_tokens = max_text_tokens
        self.rollout_temperature = rollout_temperature
        self.val_do_sample = val_do_sample

        self.pass_cot_token_only = pass_cot_token_only

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
            
        else:
            raise NotImplementedError


        # -----------------------
        # Load Model Final Check
        # -----------------------        
        # self._verify_model_dtype(self.vlm, "VLM")
        # self._verify_model_dtype(self.action_head, "Diffusion Planner")


        # -----------------------
        # training parameters 
        # -----------------------
        self._lr = vlm_lr

        self.rl_algorithm = rl_algorithm.lower()
        if self.rl_algorithm not in ("nsr", "grpo"):
            raise ValueError(
                f"Unknown rl_algorithm: {rl_algorithm!r}. Expected 'nsr' or 'grpo'."
            )
        self.grpo_cfg = dict(grpo_cfg) if grpo_cfg is not None else {}

        self.per_sample_rollout = per_sample_rollout
        self.bag_g = bag_g
        

        self.opt_type = opt_type
        self.opt_weight_decay = opt_weight_decay
        self.opt_eps = opt_eps

        self.total_training_steps = None   # set by lightning module at runtime


        # self.reward_scale = reward_scale
        # self.entropy_coef = entropy_coef
        # self.kl_coef = kl_coef
        # self.reference_policy_checkpoint = reference_policy_checkpoint




        # # -----------------------
        # # others TOOD
        # # -----------------------
        # self.num_inference_samples = 1
        # self.inference_selection_mode = "median"

        if self.mode == "eval":
            print("Agent is in eval mode.")
            self.eval()


    def initialize(self) -> None:
        """for hydra to initialize, no use. 【D】"""
        pass


    def name(self) -> str:
        """【D】"""
        return self.__class__.__name__
    

    def set_total_training_steps(self, total_steps: int):
        self.total_training_steps = total_steps
        print(f"Total training steps {total_steps}")


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
            raise RuntimeError("Found suspicious trainable fp32 params!")

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


    def compute_trajectory_recogdrive(
            self, 
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pixel_values_cat, questions, num_patches_list, history_trajectory = self.unpack_features(features)
                diff_dtype, diff_input = self.get_diff_input(features, history_trajectory)

                outputs = self.vlm(pixel_values_cat, questions, num_patches_list=num_patches_list)
                last_hidden_state = outputs.hidden_states[-1]

                status_feature = features["status_feature"].cuda()
                if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
                if last_hidden_state.ndim == 2: last_hidden_state = last_hidden_state.unsqueeze(0)

            predictions = self.action_head.get_action(last_hidden_state.to(diff_dtype), diff_input)

            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        
        return Trajectory(poses)    

    
    def compute_traj_cot(self, agent_input: AgentInput, cot_text:str) -> Trajectory:
        """
        NOTE 给 diffusion planner 的 last_hidden_states 不包括 prompt 
        
        TODO w/o image tokens 设置
        
        """
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():    
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        pixel_values_cat, questions, num_patches_list, history_trajectory = self.unpack_features_cot_prompt(features)
        diff_dtype, diff_input = self.get_diff_input(features, history_trajectory)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.pass_cot_token_only:
                    fwd_output = self.vlm.forward_cot_only(cot_text)
                else:
                    # create new prompt list(cot list)
                    batch_size = len(questions)
                    if batch_size > 1:
                        cot_list = []
                        for b in batch_size:
                            cot_list.append(f"{cot_text}")
                    else:
                        cot_list = [cot_text]

                    fwd_output = self.vlm.forward_with_ids_cot_and_image(
                        pixel_values_cat, cot_list, num_patches_list
                    )

                last_hidden_states = fwd_output.hidden_states[-1].clone()
                if last_hidden_states.ndim == 2: 
                    last_hidden_states = last_hidden_states.unsqueeze(0)

            predictions = self.action_head.get_action(last_hidden_states.to(diff_dtype), diff_input)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        
        return Trajectory(poses)   


    def unpack_features_for_vllm_service(self, agent_input: AgentInput):
        """
        vLLM service online inference input prepare

        Args:
            features: Dictionary containing raw feature tensors.

        """
        # =========================
        # unpack agent_input
        # =========================

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():    
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        image_path_tensor = features["image_path_tensor"]
        if image_path_tensor.ndim == 1: image_path_tensor = image_path_tensor.unsqueeze(0)
        image_paths = decode_paths_from_tensor(image_path_tensor)
        
        history_trajectory = features["history_trajectory"].cuda()
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)

        high_command_one_hot = features["high_command_one_hot"].cuda()
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)
        
        navigation_commands = ['turn left', 'go straight', 'turn right']
        command_indices = torch.argmax(high_command_one_hot, dim=-1)
        command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

        """
        Build per-sample prompts and corresponding HTTP payloads compatible
        with the online vLLM chat/completions API (InternVL).
        batch_size == 1
        
        """            
        # =========================
        # prepare text 
        # =========================   

        history_trajectory_sample = history_trajectory[0]
        command_str_sample = command_str_list[0]

        history_str = ' '.join([
            f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
            f'{format_number(history_trajectory_sample[j, 1].item())}, '
            f'{format_number(history_trajectory_sample[j, 2].item())})'
            for j in range(history_trajectory_sample.shape[0])
        ])

        reasoning_framework = (
            "Before providing the trajectory, follow this hierarchical cognitive process:\n"
            "1. Foundational Perception: describe critical static and dynamic elements (traffic lights, vehicles, obstacles).\n"
            "2. Dynamic Understanding: analyze movement and intent of surrounding agents relative to your path.\n"
            "3. Planning & Reasoning: formulate your high-level strategy and explain the causal reason for your decision.\n"
            "4. Advanced Reasoning: briefly consider a counterfactual to ensure safety margins.\n"
        )

        prompt = (
            "You are an advanced autonomous driving cognitive agent. Based on the provided front camera view, "
            "historical context, and navigation command, perform a step-by-step reasoning analysis followed by trajectory planning.\n\n"
            "Inputs:\n"
            f"- Historical motion (last 4 timesteps): {history_str}\n"
            f"- Navigation target: [{command_str_sample.upper()}]\n\n"
            "Instructions:\n"
            f"{reasoning_framework}"
        )

        output_requirements = (
            "Output Format:\n"
            "1) Reasoning Trace: a concise paragraph with the 4-level analysis.\n"
            "2) Trajectory: predict 8 future waypoints encapsulated in [PT, ...].\n"
            "   - Each point: (x:float, y:float, heading:float)\n"
            "   - Maintain 2 decimal places of precision.\n"
        )

        full_text = f"{prompt}\n{output_requirements}"
        text_content = {"type": "text", "text": full_text}


        # =========================
        # prepare image
        # =========================

        # Encode the corresponding front camera image as data URI (base64)
        try:
            img_path = image_paths[0]
            with open(img_path, "rb") as f:
                img = Image.open(f)
                # img_resized = img.resize((960, 540))  # TODO inf时降低分辨率试试？
                img_resized = img

                buffer = io.BytesIO()
                img_resized.save(buffer, format="JPEG")
                buffer.seek(0)

                img_b64 = base64.b64encode(buffer.read()).decode("utf-8")

            image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}


        except Exception:
            # fallback to omit image if encoding fails
            print(f"图像加载失败！图像路径：{img_path}")
            image_content = None
            return None
            
        vllm_input_messages = [
            {"role": "system", "content": [{"type": "text", "text": self.vlm.model.system_message}]},
            {"role": "user", "content": [text_content, image_content]}

        ]

        # payload = {
        #     "model": model_name,  
        #     "messages": [
        #         {
        #             "role": "system",
        #             "content": [
        #                 {"type": "text", "text": "Hello, Who are you"}
        #             ]
        #         }
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        #                 {"type": "text", "text": "Hello, Who are you"}
        #             ]
        #         }
        #     ],
        #     "max_tokens": max_tokens,
        #     "temperature": temperature,  
        #     # "extra_body": {
        #     #     "lora_request": {
        #     #         "lora_name": lora_name,
        #     #         "lora_path": lora_path
        #     #     }
        #     # }
        # }


        return vllm_input_messages



    def unpack_features_cot_prompt(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str], List[int]]:
        """
        unpack_features, 自定义 CoT Prompt

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

            # Define the hierarchical reasoning framework
            reasoning_framework = (
                "Before providing the trajectory, follow this hierarchical cognitive process:\n"
                "1. **Foundational Perception**: Describe the critical static and dynamic elements visible (traffic lights, specific vehicles, obstacles).\n"
                "2. **Dynamic Understanding**: Analyze the movement and intent of surrounding agents relative to your path.\n"
                "3. **Planning & Reasoning**: Formulate your high-level strategy and explain the causal reason for your decision.\n"
                "4. **Advanced Reasoning**: Briefly consider a counterfactual (e.g., 'If the lead car accelerates, I will...') to ensure safety margins.\n"
            )

            prompt = (
                "<image>\n"
                "You are an advanced autonomous driving cognitive agent. Based on the provided front camera view, "
                "historical context, and navigation command, perform a step-by-step reasoning analysis followed by trajectory planning.\n\n"
                "### Inputs:\n"
                f"- Historical motion (last 4 timesteps): {history_str}\n"
                f"- Navigation target: [{command_str_sample.upper()}]\n\n"
                "### Instructions:\n"
                f"{reasoning_framework}"
            )
                                    
            output_requirements = (
                "\n### Output Format:\n"
                "1. Reasoning Trace: Write your 4-level analysis as a concise paragraph.\n"
                "2. Trajectory: Predict 8 future waypoints encapsulated in [PT, ...].\n"
                "- Each point: (x:float, y:float, heading:float)\n"
                "- Maintain 2 decimal places of precision.\n"
                "- Example: [PT, (1.20, 0.50, 0.05), ...]"
            )

            questions.append(f"{prompt}{output_requirements}")
        
        return pixel_values_cat, questions, num_patches_list, history_trajectory



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


    def get_diff_input(self, features: Dict[str, torch.Tensor], history_trajectory: torch.Tensor) -> BatchFeature:
        """
        Prepare input for the Diffusion-based planner.

        """
        # extract and prepare planner input
        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        state_input = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        diff_dtype = next(self.action_head.parameters()).dtype
        diff_input = BatchFeature({
                "state": state_input.to(diff_dtype),
                "his_traj": history_trajectory_reshaped.to(diff_dtype),
                "status_feature": status_feature.to(diff_dtype)
            }
        )
        return diff_dtype, diff_input


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
        """Get Optimizer and Scheduler for Negdrive VLM LoRA Fine-tuneing 
        """

        trainable_params = [p for p in self.vlm.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self._lr,
            betas=(0.9, 0.95),
            weight_decay=self.opt_weight_decay,
            eps=self.opt_eps,
            fused=True  # for H800
        )

        # optimizer_cfg = DictConfig(dict(type="AdamW", 
        #                                 lr=self._lr, 
        #                                 weight_decay=self.opt_weight_decay, 
        #                                 betas=(0.9, 0.95))
        #                                 )
        
        # optimizer = build_from_configs(optim, 
        #                                optimizer_cfg, 
        #                                params=trainable_params)    

        total_steps = self.total_training_steps
        warmup_steps = int(total_steps * 0.05)
        print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=5e-6)
            ],
            milestones=[warmup_steps]
        )

        # scheduler = WarmupCosLR(optimizer=optimizer,  
        #                             lr=self._lr, 
        #                             min_lr=0.0, 
        #                             epochs=10, 
        #                             warmup_epochs=0
        #                             )       
                    
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}



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




def make_diffusion_planner_config(  
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
