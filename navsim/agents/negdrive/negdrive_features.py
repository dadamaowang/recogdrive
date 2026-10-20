

from typing import Dict, Optional
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .negdrive_backbone import NegDriveBackbone
from .utils.internvl_preprocess import load_image


def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class NegDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                #  cache_hidden_state: bool = False,
                #   vlm_path: Optional[str] = None,
                 ):
        """
        Initializes the negdrive feature builder.
 
        """
        super().__init__()

    def get_unique_name(self) -> str:  
        return "negdrive_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        status_feature = torch.cat([
            high_command_one_hot.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)

        image_path = str(cameras[-1].cam_f0.image)            
        path_as_ordinals = [ord(char) for char in image_path]          
        path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)            
        return {
            "history_trajectory": history_trajectory.cpu(),
            "high_command_one_hot": high_command_one_hot.cpu(),
            "status_feature": status_feature.cpu(),
            "image_path_tensor": path_tensor.cpu(),
        }



class NegDriveTrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
