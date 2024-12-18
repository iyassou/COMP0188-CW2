import numpy as np
import torch

def cnn_block(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: int,
    padding: int,
    dilation: int,
    max_pool2d_kernel_size: tuple[int, int],
    dtype: type,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    return (
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dtype=dtype,
            device=device
        ),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=max_pool2d_kernel_size)
    )

def dense_block(
    in_features: int,
    out_features: int,
    dropout: float,
    batch_norm: bool,
    activation: torch.nn.Module,
    dtype: type,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    if dropout and batch_norm:
        raise ValueError("are you sure about that?")
    return tuple(
        filter(
            lambda x: x is not None,
            (
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    dtype=dtype,
                    device=device,
                ),
                torch.nn.BatchNorm1d(out_features) if batch_norm else None,
                torch.nn.Dropout(dropout) if dropout else None,
                activation,
            )
        )
    )

class BaselineModel(torch.nn.Module):
    def __init__(self, dev: torch.device, dtype: type=np.float16):
        super().__init__()
        self.device = dev

        channels = 2, 8, 16, 32
        self.joint_cnn_encoder = torch.nn.Sequential(
            *(
                module
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
                for module in cnn_block(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                    max_pool2d_kernel_size=(2, 2), dtype=dtype, device=dev,
                )
            ),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(256, dtype=dtype, device=dev),
            torch.nn.Linear(256, 128, dtype=dtype, device=dev),
        )

        dyn_features = 15, 256, 128
        self.dynamics_encoder = torch.nn.Sequential(
            module
            for in_features, out_features in zip(dyn_features[:-1], dyn_features[1:])
            for module in dense_block(
                in_features=in_features, out_features=out_features,
                dropout=None, batch_norm=False, activation=None,
                dtype=dtype, device=dev,
            )
        )

        fl_features = 128, 64, 32, 6
        self.fusion_layer = torch.nn.Sequential(
            module
            for in_features, out_features in zip(fl_features[:-1], fl_features[1:])
            for module in dense_block(
                in_features=in_features, out_features=out_features,
                dropout=None, batch_norm=False, activation=None,
                dtype=dtype, device=dev,
            )
        )
    
    def forward(self, images: torch.Tensor, dynamics: torch.Tensor) -> torch.Tensor:
        """Accepts the concatenated front camera and mounted camera observations,
        and the robot arm's dynamics. Outputs a 6-dimensional prediction of the
        position-velocity component and the one-hot encoding of the gripper's action.
        
        Parameters
        ----------
        images: torch.Tensor
            Shape [batch_size, 448, 224], stems from:
                torch.concat((front_cam_ob, mount_cam_ob), dim=0)
        dynamics: torch.Tensor
            Shape [batch_size, 15], stems from:
                torch.concat((ee_cartesian_pos_ob, ee_cartesian_vel_ob, joint_pos_ob), dim=0)
        
        Returns
        -------
        torch.Tensor
            Shape [batch_size, 6]
        """
        images = images.to(self.device)
        dynamics = dynamics.to(self.device)
        x = self.joint_cnn_encoder(images)
        y = self.dynamics_encoder(dynamics)
        return self.fusion_layer(x + y)
