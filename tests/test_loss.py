import comp0188_cw2.loss as L

import math
import torch

N = 10
B = 7

def test_kl_divergence_standard_normal():
    mu = torch.zeros((B, N))
    logvar = torch.zeros((B, N))
    expected: dict[L.BetaVAEReduction, torch.Tensor] = {
        "sum": torch.tensor(0.0),
        "mean": torch.tensor(0.0),
    }
    for reduction, expect in expected.items():
        actual = L.kl_divergence_special_case(mu, logvar, reduction)
        assert actual == expect, f"{reduction=}"

def test_kl_divergence_normal_1_2():
    # mu = 1, var = 2
    mu = torch.ones((B, N))
    logvar = torch.log(torch.tile(torch.tensor(2), (B, N)))
    analytical = 2 - torch.tensor(math.log(2))
    truth: dict[L.BetaVAEReduction, torch.Tensor] = {
        "sum":  B * analytical * N / 2,
        "mean":     analytical * N / 2,
    }
    for reduction, expected in truth.items():
        actual = L.kl_divergence_special_case(mu, logvar, reduction)
        assert torch.isclose(actual, expected), f"{reduction=}"
