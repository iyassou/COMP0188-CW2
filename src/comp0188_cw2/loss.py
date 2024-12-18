import torch

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