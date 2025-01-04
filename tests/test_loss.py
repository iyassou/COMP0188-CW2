import comp0188_cw2.loss as L

import pytest
import torch

N = 10
B = 7

@pytest.mark.parametrize(
    "mu, var, reduction, expected",
    [
        (0, 1, "sum", torch.tensor(0.)),
        (0, 1, "mean", torch.tensor(0.)),
        (1, 2, "sum", B * N / 2 * (analytical := 2 - torch.tensor(2).log())),
        (1, 2, "mean", N / 2 * analytical),
    ]
)
def test_kl_divergence(mu, var, reduction, expected):
    mu = torch.tensor(mu).repeat(B, N)
    logvar = torch.tensor(var).repeat(B, N).log()
    actual = L.kl_divergence_special_case(mu, logvar, reduction)
    assert torch.isclose(expected, actual).all()
