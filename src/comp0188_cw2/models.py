import itertools
import torch

def Linear_block(
    in_features: int,
    out_features: int,
    dropout: float,
    batch_norm: bool,
    activation: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
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
                torch.nn.BatchNorm1d(
                    out_features,
                    eps=torch.finfo(dtype).tiny,
                    dtype=dtype,
                    device=device,
                ) if batch_norm else None,
                torch.nn.Dropout(dropout) if dropout else None,
                activation,
            )
        )
    )

def Conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: int,
    padding: int,
    dilation: int,
    activation: torch.nn.Module,
    max_pool2d_kernel_size: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    return tuple(
        filter(
            lambda x: x is not None,
            (
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
                activation,
                torch.nn.MaxPool2d(
                    kernel_size=max_pool2d_kernel_size
                ) if max_pool2d_kernel_size is not None else None,
            )
        )
    )

def ConvTranspose2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    activation: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    return tuple(
        filter(
            lambda x: x is not None,
            (
                torch.nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    dtype=dtype,
                    device=device
                ),
                activation,
            )
        )
    )

def chain_Linear_blocks(
    features: tuple[int, ...],
    dropout: float,
    batch_norm: bool,
    activation: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    LAST_LAYER = max(0, len(features) - 2)
    return tuple(
        module
        for J, (in_features, out_features) in enumerate(itertools.pairwise(features))
        for module in Linear_block(
            in_features=in_features, out_features=out_features,
            
            dropout=dropout if J != LAST_LAYER else None,
            batch_norm=batch_norm if J != LAST_LAYER else False,
            activation=activation if J != LAST_LAYER else None,
            
            dtype=dtype, device=device,
        )
    )

def chain_Conv2d_blocks(
    channels: tuple[int, ...],
    kernel_size: tuple[int, int],
    stride: int,
    padding: int,
    dilation: int,
    activation: torch.nn.Module,
    max_pool2d_kernel_size: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    return tuple(
        module
        for in_channels, out_channels in itertools.pairwise(channels)
        for module in Conv2d_block(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            activation=activation,
            max_pool2d_kernel_size=max_pool2d_kernel_size,
            dtype=dtype, device=device,
        )
    )

def chain_ConvTranspose2d_blocks(
    channels: tuple[int, ...],
    kernel_size: tuple[int, int],
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    activation: torch.nn.Module,
    dtype: torch.dtype,
    device: torch.device) -> tuple[torch.nn.Module, ...]:
    LAST_LAYER = max(0, len(channels) - 2)
    return tuple(
        module
        for J, (in_channels, out_channels) in enumerate(itertools.pairwise(channels))
        for module in ConvTranspose2d_block(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation,
            activation=activation if J != LAST_LAYER else torch.nn.Sigmoid(),
            dtype=dtype, device=device,
        )
    )

class BaselineModelArchitecture(torch.nn.Module):
    def __init__(
            self,
            joint_cnn_channels: tuple[int, ...],
            dynamics_features: tuple[int, ...],
            fusion_layer_features: tuple[int, ...],

            dropout: float,
            batch_norm: bool,
            activation: torch.nn.Module,

            device: torch.device,
            dtype: torch.dtype):
        super().__init__()
        self.device = device

        self.joint_cnn_encoder = torch.nn.Sequential(
            *chain_Conv2d_blocks(
                channels=joint_cnn_channels,
                kernel_size=(3, 3), stride=1, padding=1, dilation=1,
                activation=torch.nn.ReLU(), max_pool2d_kernel_size=(2, 2),
                dtype=dtype, device=device,
            ),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(256, dtype=dtype, device=device),
            torch.nn.Linear(256, 128, dtype=dtype, device=device),
        )

        self.dynamics_encoder = torch.nn.Sequential(
            *chain_Linear_blocks(
                features=dynamics_features,
                dropout=dropout, batch_norm=batch_norm, activation=activation,
                dtype=dtype, device=device,
            )
        )

        self.fusion_layer = torch.nn.Sequential(
            *chain_Linear_blocks(
                features=fusion_layer_features,
                dropout=dropout, batch_norm=batch_norm, activation=activation,
                dtype=dtype, device=device,
            )
        )

    def forward(self, X: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Accepts the concatenated front camera and mounted camera observations,
        and the robot arm's dynamics. Outputs a 6-dimensional prediction of the
        position-velocity component and the one-hot encoding of the gripper's action.
        
        Parameters
        ----------
        X: tuple[torch.Tensor, torch.Tensor]
            Contains two tensors:
                1) `images`, with shape [batch_size, 2, 224, 224], stemming from:
                    torch.concat((front_cam_ob, mount_cam_ob), dim=0)
                2) `dynamics`, with shape [batch_size, 15], stemming from:
                    torch.concat((ee_cartesian_pos_ob, ee_cartesian_vel_ob, joint_pos_ob), dim=0)
        
        Returns
        -------
        torch.Tensor
            Shape [batch_size, 6]"""
        images, dynamics = X
        images = images.to(self.device)
        dynamics = dynamics.to(self.device)
        x = self.joint_cnn_encoder(images)
        y = self.dynamics_encoder(dynamics)
        return self.fusion_layer(x + y)

class VanillaBaselineModel(BaselineModelArchitecture):
    JOINT_CNN_CHANNELS: tuple[int, ...] = (2, 8, 16, 32)
    DYNAMICS_FEATURES: tuple[int, ...] = (15, 256, 128)
    FUSION_LAYER_FEATURES: tuple[int, ...] = (128, 64, 32, 6)
    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            joint_cnn_channels=VanillaBaselineModel.JOINT_CNN_CHANNELS,
            dynamics_features=VanillaBaselineModel.DYNAMICS_FEATURES,
            fusion_layer_features=VanillaBaselineModel.FUSION_LAYER_FEATURES,
            dropout=None,
            batch_norm=False,
            activation=None,
            
            device=device,
            dtype=dtype
        )

class VariationalAutoEncoder(torch.nn.Module):
    def __init__(
            self,
            latent_space_dimensions: int,
            device: torch.device,
            dtype: torch.dtype):
        super().__init__()
        self.channels = 2, 32, 64, 128, 256, 512
        # NOTE: stride=2 so image dimensions go: 224 => 112 => 56 => 28 => 14 => 7
        self.last_image_dimensions = 7
        self.latent_space_dimensions = latent_space_dimensions
        self.device = device
        self.dtype = dtype

        self.encoder = torch.nn.Sequential(
            *chain_Conv2d_blocks(
                channels=self.channels,
                kernel_size=(3, 3), stride=2, padding=1, dilation=1,
                activation=torch.nn.LeakyReLU(), max_pool2d_kernel_size=None,
                dtype=dtype, device=device,
            ),
            torch.nn.Flatten(),
        )
        output_dimensions = self.channels[-1] * self.last_image_dimensions ** 2
        self.mean = torch.nn.Linear(output_dimensions, latent_space_dimensions, dtype=dtype, device=device)
        # NOTE: log(variance) for numerical stability.
        self.log_variance = torch.nn.Linear(output_dimensions, latent_space_dimensions, dtype=dtype, device=device)
        self.latent_space_decoder = torch.nn.Linear(latent_space_dimensions, output_dimensions, dtype=dtype, device=device)
        self.decoder = torch.nn.Sequential(
            *chain_ConvTranspose2d_blocks(
                channels=self.channels[::-1],
                kernel_size=(3, 3), stride=2, padding=1, output_padding=1,
                dilation=1, activation=torch.nn.LeakyReLU(),
                dtype=dtype, device=device,
            )
        )

    def reparametrise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrisation trick."""
        # NOTE: exp(0.5 * log(var)) == exp(log( sqrt(var) ))
        standard_deviation: torch.Tensor = torch.exp(0.5 * logvar)
        # NOTE: `torch.randn_like(x)` takes `dtype` and `device`` into account.
        epsilon: torch.Tensor = torch.randn_like(standard_deviation)
        return mu + standard_deviation * epsilon
    
    def reshape_latent_space_vector(self, z: torch.Tensor) -> torch.Tensor:
        """Reshapes a latent space vector for use by the latent space decoder.
        
        Parameters
        ----------
        z: torch.Tensor
            Shape [batch_size, self.channels[-1] * self.last_image_dimensions ** 2]

        Returns
        -------
        torch.Tensor
            Shape [batch_size, self.channels[-1], self.last_image_dimensions, self.last_image_dimensions]"""
        return z.view(
            z.size(0),
            self.channels[-1],
            self.last_image_dimensions,
            self.last_image_dimensions
        )
    
    def forward(self, X: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accepts the concatenated front camera and mounted camera observations and the robot
        arm's dynamics. Returns the beta-VAE loss and latent space parameters.
        
        Parameters
        ----------
        X: tuple[torch.Tensor, torch.Tensor]
            (0) `images`, with shape        [batch_size, 2, 224, 224]
            (1) `dynamics`, with shape      [batch_size, 15]

        Notes
        -----
        DOES NOT USE THE DYNAMICS DATA! It's just convenient to keep the `forward` function signatures
        the same and drop the unused data on a per-case basis rather than change everything elsewhere.
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (0) VAE's reconstruction, with shape            [batch_size, 2, 224, 224]
            (1) Latent space's mean, with shape             [batch_size, latent_space_dimensions]
            (2) Log of latent space's variance, with shape  [batch_size, latent_space_dimensions]"""
        images, _ = X
        encoded = self.encoder(images)
        mu = self.mean(encoded)
        logvar = self.log_variance(encoded)
        z = self.reparametrise(mu, logvar)
        z = self.latent_space_decoder(z)
        z = self.reshape_latent_space_vector(z)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
