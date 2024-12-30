import torch
import typing

BetaVAEReconstructionLoss = typing.Literal["mse", "bce"]
BetaVAEReduction = typing.Literal["mean", "sum"]

def kl_divergence_special_case(mu: torch.Tensor, logvar: torch.Tensor, reduction: BetaVAEReduction) -> torch.Tensor:
    """Returns the closed form KL divergence between an arbitrary Gaussian and a normal Gaussian.
    
    Parameters
    ----------
    mu: torch.Tensor
        The (not necessarily) non-normal Gaussian's mean, shape:
                [batch_size, x]
    logvar: torch.Tensor
        The logarithm of the (not necessarily) non-normal Gaussian's variance, shape:
                [batch_size, x]
    
    Returns
    -------
    torch.Tensor"""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "mean":
        kl = kl / mu.size(0)
    return kl

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
    def __init__(self, reconstruction_loss: BetaVAEReconstructionLoss, reduction: BetaVAEReduction):
        super().__init__()
        if reconstruction_loss not in typing.get_args(BetaVAEReconstructionLoss):
            raise ValueError(
                f"unrecognised `reconstruction_loss` {repr(reconstruction_loss)}, "
                f"must be one of: {', '.join(map(repr, typing.get_args(BetaVAEReconstructionLoss)))}"
            )
        if reduction not in typing.get_args(BetaVAEReduction):
            raise ValueError(
                f"unrecognised `reduction` {repr(reconstruction_loss)}, "
                f"must be one of: {', '.join(map(repr, typing.get_args(BetaVAEReduction)))}"
            )
        
        self.beta = torch.tensor(0.0, requires_grad=False)
        self.reduction = reduction
        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction=reduction)
        else:
            self.reconstruction_loss = torch.nn.BCELoss(reduction=reduction)

    def forward(
            self, 
            prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            actual: tuple[torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
        """Accepts a tuple comprising the reconstruction and latent space vectors' parameters
        (mean and log of the variance), and the input data fed to the model.
        
        Parameters
        ----------
        prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (0) VAE's reconstruction, with shape                    [batch_size, 2, 224, 224]
            (1) Latent space vectors' means, with shape             [batch_size, latent_space_dimension]
            (2) Log of latent space vectors' variances, with shape  [batch_size, latent_space_dimension]
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
        recon, mu, logvar = prediction
        images, _ = actual
        r_loss = self.reconstruction_loss(recon, images)
        kl_loss = kl_divergence_special_case(mu, logvar, self.reduction)
        return r_loss + self.beta * kl_loss
