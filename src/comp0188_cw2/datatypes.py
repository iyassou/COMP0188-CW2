import dataclasses
import enum
import h5py
import numpy as np
import numpy.typing as npt
import typing

from pathlib import Path

class Stepper(typing.Protocol):
    """torch.optim.lr_scheduler._LRScheduler superset"""
    def step(self):
        ...

V2              = typing.Annotated[npt.NDArray[np.float16], typing.Literal[2]]
V4              = typing.Annotated[npt.NDArray[np.float16], typing.Literal[4]]
V6              = typing.Annotated[npt.NDArray[np.float16], typing.Literal[6]]
V7              = typing.Annotated[npt.NDArray[np.float16], typing.Literal[7]]
Grayscale       = typing.Annotated[npt.NDArray[np.float16], typing.Literal["H", "W"]]

class GripperAction(enum.Enum):
    OPEN     = 0
    STATIC   = 1
    CLOSE    = 2

# No "prompts" or "rewards" in the preprocessed versions we're working with.
_ObservationBaseAttribute = typing.Literal[
    "actions", "ee_cartesian_pos_ob", "ee_cartesian_vel_ob", "front_cam_ob",
    "joint_pos_ob", "mount_cam_ob", "terminals"
]
_ObservationBaseTypes = V4, V7, V6, Grayscale, V2, Grayscale, np.bool_
_ObservationBase = dataclasses.make_dataclass(
    "_ObservationBase", zip(typing.get_args(_ObservationBaseAttribute), _ObservationBaseTypes)
)

@dataclasses.dataclass
class Observation(_ObservationBase):
    @property
    def gripper_action(self) -> GripperAction:
        return GripperAction(self.actions[3])

@dataclasses.dataclass
class Chunk:
    file: Path
    n: typing.Optional[int] = None

    def __len__(self) -> int:
        if self.n is None:
            with h5py.File(self.file, "r") as f:
                # Assuming all keys have the same number of entries
                self.n = len(f[next(iter(f.keys()))])
        return self.n
    
    def __getitem__(self, j: typing.Union[int, slice]) -> typing.Union[Observation, typing.List[Observation]]:
        if not isinstance(j, (int, slice)):
            raise TypeError(f"Unsupported index type: {repr(type(j))}")
        
        if isinstance(j, int):
            if j < 0:
                j %= len(self)
        else:
            start, stop, step = j.start, j.stop, j.step
            if start is not None and start < 0:
                start %= len(self)
            if stop is not None and stop < 0:
                stop %= len(self)
            start = start or 0
            stop = stop or len(self)
            step = step or 1
            if start > stop:
                raise IndexError(f"Invalid slice {j}")
            j = range(start, stop, step)

        with h5py.File(self.file, "r") as f:
            data = {k: f[k][j] for k in typing.get_args(_ObservationBaseAttribute)}

        if isinstance(j, int):
            return Observation(**data)
        j = range(len(j))
        return [Observation(**{k: v[m] for k, v in data.items()}) for m in j]
