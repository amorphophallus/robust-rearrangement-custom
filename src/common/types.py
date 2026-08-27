import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Literal


class Observation(TypedDict):
    color_image1: np.ndarray
    color_image2: np.ndarray
    robot_state: dict
    image_size: Tuple[int]
    parts_poses: np.ndarray
    skill: Optional[str]
    guidance_point: Optional[np.ndarray]
    guidance_point_clean: Optional[np.ndarray]
    guidance_pose: Optional[np.ndarray]
    guidance_pose_clean: Optional[np.ndarray]
    guidance_frame: str
    guidance_gripper_width: Optional[float]
    guidance_point_2d: Dict[str, Optional[np.ndarray]]
    grasp_annotation_2d: Dict[str, Optional[dict]]


class Trajectory(TypedDict):
    observations: List[Observation]
    actions: List[np.ndarray]
    rewards: List[float]
    camera_info: Dict[str, Any]
    skills: List[str]
    success: bool
    furniture: str
    error: bool
    error_description: str
    annotation_source: str
    vlm_model_revision: Optional[str]
    guidance_frame: str
    guidance_schema_version: int


# Make type for the encoder name choices
EncoderName = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "spatial_softmax",
    "dino",
    "mae",
    "voltron",
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "vip",
    "r3m_18",
    "r3m_34",
    "r3m_50",
]

TaskName = Literal[
    "one_leg",
    "lamp",
    "round_table",
    "desk",
    "square_table",
    "cabinet",
    "chair",
    "stool",
]


Controllers = Literal["sim", "real"]

Domains = Literal["sim", "real"]

DemoSources = Literal["scripted", "rollout", "teleop", "augmentation"]

Randomness = Literal["low", "med", "high"]

DemoStatus = Literal["success", "failure", "partial_success"]
