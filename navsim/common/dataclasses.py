from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import os

import numpy as np
import numpy.typing as npt
from PIL import Image
from pyquaternion import Quaternion


from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")


@dataclass
class Camera:
    """Camera dataclass for image and parameters."""

    image: Optional[npt.NDArray[np.float32]] = None

    # 激光雷达的旋转/平移（外参）和相机自身的内参/畸变参数
    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_translation: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float32]] = None
    distortion: Optional[npt.NDArray[np.float32]] = None


@dataclass
class Cameras:
    """Multi-camera dataclass."""

    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_camera_dict(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
        load_image_path: bool = False 
    ) -> Cameras:
        """
        Load camera dataclass from dictionary.
        :param sensor_blobs_path: root directory of sensor data.
        :param camera_dict: dictionary containing camera specifications.
        :param sensor_names: list of camera identifiers to include.
        :return: Cameras dataclass.
        """

        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                if not load_image_path:
                    data_dict[camera_identifier] = Camera(
                        image=np.array(Image.open(image_path)),
                        sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                        sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                        intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                        distortion=camera_dict[camera_name]["distortion"],
                    )
                else:
                    data_dict[camera_identifier] = Camera(
                        image=image_path,
                        sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                        sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                        intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                        distortion=camera_dict[camera_name]["distortion"],
                    )
            else:
                data_dict[camera_identifier] = Camera()  # empty camera

        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@dataclass
class Lidar:
    """Lidar point cloud dataclass."""

    # NOTE:
    # merged lidar point cloud as (6,n) float32 array with n points
    # first axis: (x, y, z, intensity, ring, lidar_id), see LidarIndex
    lidar_pc: Optional[npt.NDArray[np.float32]] = None

    @staticmethod
    def _load_bytes(lidar_path: Path) -> BinaryIO:
        """Helper static method to load lidar point cloud stream."""
        # 二进制读取模式 ("rb") 打开文件
        
        with open(lidar_path, "rb") as fp:
            """
            fp.read(): 一次性读取文件的全部二进制内容。
            io.BytesIO(...): 将这些二进制内容封装成一个在内存中的二进制流对象。
            这样做的好处是，下游的库（比如 nuPlan 的点云解析库）可以直接从这个内存流中读取数据，
            而不需要再次操作磁盘文件，效率更高。
            """
            return io.BytesIO(fp.read())

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, lidar_path: Path, sensor_names: List[str]) -> Lidar:
        """
        Loads lidar point cloud dataclass in log loading.
        :param sensor_blobs_path: root directory to sensor data
        :param lidar_path: relative lidar path from logs.
        :param sensor_names: list of sensor identifiers to load`
        :return: lidar point cloud dataclass
        """

        # NOTE: this could be extended to load specific LiDARs in the merged pc
        
        if "lidar_pc" in sensor_names:
            global_lidar_path = sensor_blobs_path / lidar_path
            """
            LidarPointCloud.from_buffer(..., "pcd"): 调用 nuplan 库提供的 LidarPointCloud 类的 from_buffer 方法。
                这个方法会解析内存中的二进制流（并告诉它格式是 .pcd），将其转换成一个 LidarPointCloud 对象。
            .points: 从 LidarPointCloud 对象中，提取出最终的、我们需要的 NumPy 数组
            """
            lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
            return Lidar(lidar_pc)
        return Lidar()  # empty lidar


@dataclass
class EgoStatus:
    """Ego vehicle status dataclass."""

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    in_global_frame: bool = False  # False for AgentInput


@dataclass
class AgentInput:
    """Dataclass for agent inputs with current and past ego statuses and sensors."""

    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
        load_image_path: bool = False 
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]), np.array(global_ego_poses, dtype=np.float64)
        )

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        for frame_idx in range(num_history_frames):

            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            # sensors to load
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                    load_image_path=load_image_path,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                    sensor_names=sensor_names,
                )
            )

        return AgentInput(ego_statuses, cameras, lidars)


@dataclass
class Annotations:
    """Dataclass of annotations (e.g. bounding boxes) per frame."""

    # npt: numpy.typing -numpy的类型提示，表示这是一个numpy数组
    """
    boxes: npt.NDArray[np.float32]:
    存储场景中所有物体的三维边界框（3D Bounding Boxes）。
    这是一个 NumPy 数组，形状通常是 (N, 7)，其中 N 是物体的数量，7 代表每个物体的 (x, y, z, length, width, height, heading)。
    names: List[str]:
    存储每个边界框对应的类别名称。
    这是一个字符串列表，长度为 N。例如 ['vehicle.car', 'pedestrian.adult', 'vehicle.bicycle']。
    velocity_3d: npt.NDArray[np.float32]:
    存储每个物体的三维速度。
    这是一个 NumPy 数组，形状为 (N, 3)，代表每个物体的 (vx, vy, vz)。
    instance_tokens: List[str]:
    存储每个物体的实例ID（Instance ID）。
    这是一个在整个数据集中都唯一的ID，用来标识一个独一无二的物体（比如，“在波士顿某条路上的这辆特定的丰田凯美瑞”）。
    track_tokens: List[str]:
    存储每个物体的追踪ID（Track ID）。
    这是一个在当前场景（Scene）内唯一的ID，用来在连续的帧之间追踪同一个物体。例如，第1帧中 track_token 为 "abc" 的汽车，在第2帧中 track_token 同样为 "abc"。
    """
    boxes: npt.NDArray[np.float32]
    names: List[str]
    velocity_3d: npt.NDArray[np.float32]
    instance_tokens: List[str]
    track_tokens: List[str]

    def __post_init__(self):
        # vars(self) 获取当前对象的所有属性和值，返回一个字典
        annotation_lengths: Dict[str, int] = {
            attribute_name: len(attribute) for attribute_name, attribute in vars(self).items()
        }
        assert (
            len(set(annotation_lengths.values())) == 1
        ), f"Annotations expects all attributes to have equal length, but got {annotation_lengths}"


@dataclass
class Trajectory:
    """Trajectory dataclass in NAVSIM."""

    poses: npt.NDArray[np.float32]  # local coordinates
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    """Dataclass of scene metadata (e.g. location) per scene."""
    # Metadata about the scene, such as its origin, map, and frame counts.
    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int


@dataclass
class Frame:
    """Frame dataclass with privileged information."""

    token: str
    timestamp: int # represent the time of the frame
    roadblock_ids: List[str] # List of roadblock IDs present in the frame
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Lidar
    cameras: Cameras


@dataclass
class Scene:
    """Scene dataclass defining a single sample in NAVSIM."""

    # Ground truth information
    scene_metadata: SceneMetadata
    map_api: AbstractMap
    frames: List[Frame]

    def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts future trajectory of the human operator in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_future_frames

        start_frame_idx = self.scene_metadata.num_history_frames - 1

        global_ego_poses = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_trajectory_frames + 1):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            # StateSE2(*global_ego_poses[0]) 代表将 global_ego_poses 中的第一个元素作为参考点
            StateSE2(*global_ego_poses[0]), np.array(global_ego_poses[1:], dtype=np.float64)
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_history_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts past trajectory of ego vehicles in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_history_frames

        global_ego_poses = []
        for frame_idx in range(num_trajectory_frames):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[-1])
        local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_agent_input(self) -> AgentInput:
        """
        Extracts agents input dataclass (without privileged information) from scene.
        :return: agent input dataclass
        """

        local_ego_poses = self.get_history_trajectory().poses
        ego_statuses: List[EgoStatus] = []
        cameras: List[Cameras] = []
        lidars: List[Lidar] = []

        for frame_idx in range(self.scene_metadata.num_history_frames):
            frame_ego_status = self.frames[frame_idx].ego_status

            ego_statuses.append(
                EgoStatus(
                    ego_pose=local_ego_poses[frame_idx],
                    ego_velocity=frame_ego_status.ego_velocity,
                    ego_acceleration=frame_ego_status.ego_acceleration,
                    driving_command=frame_ego_status.driving_command,
                )
            )
            cameras.append(self.frames[frame_idx].cameras)
            lidars.append(self.frames[frame_idx].lidar)

        return AgentInput(ego_statuses, cameras, lidars)

    @classmethod
    def _build_map_api(cls, map_name: str) -> AbstractMap:
        """Helper classmethod to load map api from name."""
        assert map_name in MAP_LOCATIONS, f"The map name {map_name} is invalid, must be in {MAP_LOCATIONS}"
        return get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)

    @classmethod
    def _build_annotations(cls, scene_frame: Dict) -> Annotations:
        """Helper classmethod to load annotation dataclass from logs."""
        return Annotations(
            boxes=scene_frame["anns"]["gt_boxes"],
            names=scene_frame["anns"]["gt_names"],
            velocity_3d=scene_frame["anns"]["gt_velocity_3d"],
            instance_tokens=scene_frame["anns"]["instance_tokens"],
            track_tokens=scene_frame["anns"]["track_tokens"],
        )

    @classmethod
    def _build_ego_status(cls, scene_frame: Dict) -> EgoStatus:
        """Helper classmethod to load ego status dataclass from logs."""
        ego_translation = scene_frame["ego2global_translation"]
        ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )
        ego_dynamic_state = scene_frame["ego_dynamic_state"]
        return EgoStatus(
            ego_pose=global_ego_pose,
            ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
            ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
            driving_command=scene_frame["driving_command"],
            in_global_frame=True,
        )

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
        load_image_path: bool = False 
    ) -> Scene:
        """
        Load scene dataclass from scene dictionary list (for log loading).
        :param scene_dict_list: list of scene frames (in logs)
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of past and current frames to load
        :param num_future_frames: number of future frames to load
        :param sensor_config: sensor config dataclass
        :return: scene dataclass
        """
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        map_api = cls._build_map_api(scene_metadata.map_name)

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = cls._build_ego_status(scene_dict_list[frame_idx])
            annotations = cls._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)

            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=scene_dict_list[frame_idx]["cams"],
                sensor_names=sensor_names,
                load_image_path=load_image_path,
            )

            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                sensor_names=sensor_names,
            )

            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
            )
            frames.append(frame)

        return Scene(scene_metadata=scene_metadata, map_api=map_api, frames=frames)


@dataclass
class SceneFilter:
    """Scene filtering configuration for scene loading."""

    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: Optional[int] = None # interval between frames, defaults to None
    has_route: bool = True # whether to filter scenes with no route

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    # TODO: expand filter options

    def __post_init__(self):

        if self.frame_interval is None:
            self.frame_interval = self.num_frames

        assert self.num_history_frames >= 1, "SceneFilter: num_history_frames must greater equal one."
        assert self.num_future_frames >= 0, "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."

    @property
    def num_frames(self) -> int:
        """
        :return: total number for frames for scenes to extract.
        """
        return self.num_history_frames + self.num_future_frames


@dataclass
class SensorConfig:
    """Configuration dataclass of agent sensors for memory management."""

    # Config values of sensors are either
    # - bool: Whether to load history or not
    # - List[int]: For loading specific history steps
    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]

    def get_sensors_at_iteration(self, iteration: int) -> List[str]:
        """
        Creates a list of sensor identifiers given iteration.
        :param iteration: integer indicating the history iteration.
        :return: list of sensor identifiers to load.
        """
        sensors_at_iteration: List[str] = []
        for sensor_name, sensor_include in asdict(self).items():
            if isinstance(sensor_include, bool) and sensor_include:
                sensors_at_iteration.append(sensor_name)
            elif isinstance(sensor_include, list) and iteration in sensor_include:
                sensors_at_iteration.append(sensor_name)
        return sensors_at_iteration

    @classmethod
    def build_all_sensors(cls, include: Union[bool, List[int]] = True) -> SensorConfig:
        """
        Classmethod to load all sensors with the same specification.
        :param include: boolean or integers for sensors to include, defaults to True
        :return: sensor configuration dataclass
        """
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=include,
            cam_l2=include,
            cam_r0=include,
            cam_r1=include,
            cam_r2=include,
            cam_b0=include,
            lidar_pc=include,
        )

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        """
        Classmethod to load no sensors.
        :return: sensor configuration dataclass
        """
        return cls.build_all_sensors(include=False)


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    """
    no_at_fault_collisions: the
    """
    no_at_fault_collisions: float
    drivable_area_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    comfort: float
    driving_direction_compliance: float

    score: float

@dataclass
class PDMResults_safe:
    """Helper dataclass to record PDM results."""

    """
    no_at_fault_collisions: the
    """
    no_at_fault_collisions: float
    drivable_area_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    comfort: float
    driving_direction_compliance: float

    score: float
    reward_score: float
    cost_score: float

