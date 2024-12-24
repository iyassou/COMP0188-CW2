import torch
import typing

BetaVAEReconstructionLoss = typing.Literal["mse", "bce"]

def kl_divergence_special_case(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Returns the KL divergence between an arbitrary Gaussian and a normal Gaussian.
    Computes the closed-form.
    
    Parameters
    ----------
    mu: torch.Tensor
        The (not necessarily) non-normal Gaussian's mean, with shape:
            [batch_size, x]
    logvar: torch.Tensor
        The logarithm of the (not necessarily) non-normal Gaussian's variance, with shape:
            [batch_size, x]

    Notes
    -----
    Batch normalises the KL divergence.
            
    Returns
    -------
    torch.Tensor"""
    batch_size = mu.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_batch_norm = kl / batch_size
    return kl_batch_norm

class BalancedMSECrossEntropyLoss(torch.nn.modules.loss._Loss):
    """Balanced loss, equally weighting the MSE loss of the position-velocity
    component and the cross-entropy loss of the gripper component."""
    def __init__(self):
        super().__init__()
        self.pos_vel_loss = torch.nn.MSELoss(reduction="mean")
        self.gripper_loss = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def forward(
            self, prediction: torch.Tensor, actual: torch.Tensor,
        ) -> torch.Tensor:
        """`prediction` and `actual` are 6-dimensional tensors, containing the
        robot arm's position-velocity components and the one-hot encoding of
        the gripper's action.
        
        Parameters
        ----------
        prediction: torch.Tensor
            Shape [batch_size, 6]
        actual: torch.Tensor
            Shape [batch_size, 6]

        Returns
        -------
        torch.Tensor"""
        pv_loss = self.pos_vel_loss(prediction[:, :3], actual[:, :3])
        g_loss = self.gripper_loss(prediction[:, 3:], actual[:, 3:])
        return torch.mean(pv_loss + g_loss)

class BetaVAELoss(torch.nn.modules.loss._Loss):
    """Parametrised loss function for beta variational autoencoders.
    Returns the sum of the reconstruction loss and the beta-weighted KL divergence."""
    def __init__(self, beta: float, reconstruction_loss: BetaVAEReconstructionLoss):
        super().__init__()
        if reconstruction_loss not in typing.get_args(BetaVAEReconstructionLoss):
            raise ValueError(
                f"unrecognised `reconstruction_loss` {repr(reconstruction_loss)}, "
                f"must be one of: {', '.join(map(repr, typing.get_args(BetaVAEReconstructionLoss)))}"
            )
        
        self.beta = beta
        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")
        else:
            self.reconstruction_loss = torch.nn.BCELoss(reduction="mean")

    def forward(
            self, 
            prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            actual: tuple[torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
        """Accepts a tuple comprising the reconstruction and two latent space distribution
        parameters (mean and log of the variance), and the input data fed to the model.
        
        Parameters
        ----------
        prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (0) VAE's reconstruction, with shape            [batch_size, 2, 224, 224]
            (1) Latent space's mean, with shape             [batch_size, latent_space_dimensions]
            (2) Log of latent space's variance, with shape  [batch_size, latent_space_dimensions]
        actual: tuple[torch.Tensor, torch.Tensor]
            (0) `images`, with shape                        [batch_size, 2, 224, 224]
            (1) `dynamics`, with shape                      [batch_size, 15]

        Notes
        -----
        DOES NOT USE THE DYNAMICS DATA! It's just convenient to keep the `forward` function signatures
        the same and drop the unused data on a per-case basis rather than change everything elsewhere.
        
        Uses the closed-form of the KL divergence between an arbitrary Gaussian and a normal Gaussian.

        Returns
        -------
        torch.Tensor
            Sum of the reconstruction loss and the beta-weighted KL divergence"""
        reconstruction, mu, logvar = prediction
        images, _ = actual
        r_loss = self.reconstruction_loss(reconstruction, images)
        kl_loss = kl_divergence_special_case(mu, logvar)
        return r_loss + self.beta * kl_loss
