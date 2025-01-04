from .datatypes import (
    Chunk,
    GripperAction,
    Observation,
)

import enum
import torch

from collections.abc import Mapping
from pathlib import Path
from torchvision.transforms import v2

class Camera(enum.Flag):
    FRONT = enum.auto()
    MOUNT = enum.auto()

    @property
    def single(self) -> bool:
        return bool(self ^ (Camera.FRONT | Camera.MOUNT))

    @property
    def name(self) -> str:
        if not self.single:
            return "both"
        return super().name.lower()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory: Path,
        transforms: Mapping[str, v2.Transform],
        camera: Camera,
        dtype: torch.dtype,
        device: torch.device):

        self.observations = []
        h5_files = sorted(
            directory.glob("*.h5"),
            key=lambda x: int(x.stem.split("_")[-1]), # ascending order of suffix number
        )
        for file in h5_files:
            for obs in Chunk(file)[:]:
                for key, transform in transforms.items():
                    setattr(obs, key, transform(getattr(obs, key)))
                self.observations.append(obs)

        self.camera = camera
        self.dtype = dtype
        self.device = device

    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, j: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not (0 <= j < len(self)):
            raise IndexError(f"Index {j} out-of-bounds for container of length {len(self)}")
        obs: Observation = self.observations[j]

        # Observations
        images: list[torch.Tensor] = []
        if self.camera & Camera.FRONT:
            images.append(obs.front_cam_ob)
        if self.camera & Camera.MOUNT:
            images.append(obs.mount_cam_ob)
        images: torch.Tensor = torch.concat(images, dim=0)
        images = images.type(self.dtype)
        dynamics = torch.concat(
            (
                obs.ee_cartesian_pos_ob,
                obs.ee_cartesian_vel_ob,
                obs.joint_pos_ob
            ), dim=0)
        dynamics = dynamics.type(self.dtype)
        
        # Actions
        delta_dynamics = obs.actions[:3]
        delta_dynamics = delta_dynamics.type(self.dtype)
        delta_dynamics = delta_dynamics.to(self.device)
        one_hot = torch.nn.functional.one_hot(
            torch.tensor(obs.actions[3].item(), dtype=torch.long, device=self.device),
            num_classes=len(GripperAction),
        )
        one_hot = one_hot.type(self.dtype)
        actions = torch.concat((delta_dynamics, one_hot), dim=0)

        images = images.to(self.device)
        dynamics = dynamics.to(self.device)
        return (images, dynamics), actions
