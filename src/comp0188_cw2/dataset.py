from .datatypes import (
    Chunk,
    GripperAction,
    Observation,
)

import bisect
import torch

from pathlib import Path

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, directory: Path, transforms: dict, dev: torch.device):
        self.transforms = transforms
        self.device = dev
        h5_files = sorted(
            directory.glob("*.h5"),
            key=lambda x: int(x.stem.split("_")[-1]), # ascending order of suffix number
        )
        self.chunks = tuple(Chunk(file) for file in h5_files)
        acc = 0
        self.cumulative_lengths = tuple(acc := acc + len(c) for c in self.chunks)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, j: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not (0 <= j < len(self)):
            raise IndexError(f"Index {j} out-of-bounds for container of length {len(self)}")
        chunk_index: int = bisect.bisect(self.cumulative_lengths, j)
        if chunk_index:
            j -= self.cumulative_lengths[chunk_index - 1]
        obs: Observation = self.chunks[chunk_index][j]

        if self.transforms:
            for key, transform in self.transforms.items():
                setattr(obs, key, transform(getattr(obs, key)))

        # Observations
        images = torch.concat((obs.front_cam_ob, obs.mount_cam_ob), dim=0)
        dynamics = torch.concat(
            (
                obs.ee_cartesian_pos_ob,
                obs.ee_cartesian_vel_ob,
                obs.joint_pos_ob
            ), dim=0)
        
        # Actions
        delta_dynamics = obs.actions[:3].to(self.device)
        actions = torch.concat(
            (
                delta_dynamics,
                torch.nn.functional.one_hot(
                    torch.tensor(obs.actions[3].item(), dtype=torch.long, device=self.device),
                    num_classes=len(GripperAction),
                ).type(torch.float16)
            ), dim=0
        )

        images = images.to(self.device)
        dynamics = dynamics.to(self.device)
        actions = actions.to(self.device)
        return (images, dynamics), actions
