import comp0188_cw2.metrics as M

import math
import pytest
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = torch.finfo(DTYPE).eps
N = 10

def random_integer(low=10, high=100) -> torch.Tensor:
    return torch.randint(low=low, high=high, size=(1,), dtype=DTYPE, device=DEVICE)

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            torch.ones(1, N, dtype=DTYPE, device=DEVICE),
            torch.ones(1, N, dtype=DTYPE, device=DEVICE),
            1,
        ),
        (
            torch.ones(3, N, dtype=DTYPE, device=DEVICE),
            torch.ones(3, N, dtype=DTYPE, device=DEVICE),
            1,
        ),
        (
            a := torch.nn.functional.normalize(
                torch.ones(1, N, dtype=DTYPE, device=DEVICE),
                eps=0
            ),
            -a,
            -1,
        ),
        (
            a,
            torch.tensor(
                [x for _ in range(N // 2) for x in (1, 0)],
                dtype=DTYPE, device=DEVICE,
            ).reshape(1, N),
            math.cos(math.radians(45)),
        ),
        (
            a,
            torch.tensor(
                [x for _ in range(N // 2) for x in (1, -1)],
                dtype=DTYPE, device=DEVICE,
            ).reshape(1, N),
            0,
        ),
    ]
)
def test_cosine_similarity(a, b, expected):
    cossim = M.CosineSimilarity(eps=EPS, device=DEVICE)
    cossim.update(a, b)
    actual = cossim.compute().type(DTYPE)
    assert pytest.approx(expected) == actual.item()

@pytest.mark.parametrize(
    "average, channels, a, b, expected",
    [
        (None, N, a := torch.ones(N), a, torch.zeros_like(a)),
        ("macro", None, a, a, torch.zeros(1)),
        ("macro", None, a, (x := random_integer()) * a, x - 1),
        (None, N, a, a + x, x)
    ]
)
def test_mean_absolute_error(average, channels, a, b, expected):
    mae = M.MeanAbsoluteError(average=average, channels=channels, device=DEVICE)
    mae.update(a, b)
    actual = mae.compute()
    assert torch.isclose(expected, actual).all()

@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.ones(N), torch.ones(1)),
        (torch.zeros(N), torch.zeros(1)),
        ((rand := random_integer()) * torch.ones(N), rand),
    ]
)
def test_meanf32(x, expected):
    mf32 = M.MeanFloat32(device=DEVICE)
    mf32.update(x)
    actual = mf32.compute()
    assert torch.isclose(expected, actual).all()

@pytest.mark.parametrize(
    "kernel_size, x, y, expected",
    [
        (
            7,
            x := torch.rand(1, 2, 10, 10, dtype=DTYPE, device=DEVICE),
            x,
            torch.ones(2, dtype=DTYPE, device=DEVICE),
        ),
        (
            7,
            x := torch.rand(1, 1, 10, 10, dtype=DTYPE, device=DEVICE),
            x,
            torch.ones(1, dtype=DTYPE, device=DEVICE),
        ),
        (
            3,
            x := torch.zeros(1, 2, 10, 10, dtype=DTYPE, device=DEVICE),
            torch.ones_like(x),
            torch.zeros(2, dtype=DTYPE, device=DEVICE),
        ),
    ]
)
def test_structural_similarity(kernel_size, x, y, expected):
    ss = M.StructuralSimilarity(
        kernel_size=kernel_size, channels=x.size(1), value_range=1., device=DEVICE
    )
    ss.update(x, y)
    actual = ss.compute()
    assert torch.isclose(expected, actual, atol=1e-4).all()
