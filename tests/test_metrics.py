import comp0188_cw2.metrics as M

import math
import pytest
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = torch.finfo(DTYPE).eps
N = 10

@pytest.fixture
def cosine_similarity():
    return M.CosineSimilarity(eps=EPS, device=DEVICE)

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            torch.ones((1, N), dtype=DTYPE, device=DEVICE),
            torch.ones((1, N), dtype=DTYPE, device=DEVICE),
            1,
        ),
        (
            torch.ones((3, N), dtype=DTYPE, device=DEVICE),
            torch.ones((3, N), dtype=DTYPE, device=DEVICE),
            1,
        ),
        (
            a := torch.nn.functional.normalize(
                torch.ones((1, N), dtype=DTYPE, device=DEVICE),
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
            ).reshape((1, N)),
            math.cos(math.radians(45)),
        ),
        (
            a,
            torch.tensor(
                [x for _ in range(N // 2) for x in (1, -1)],
                dtype=DTYPE, device=DEVICE,
            ).reshape((1, N)),
            0,
        ),
    ]
)
def test_cosine_similarity(a, b, expected, cosine_similarity):
    cosine_similarity.update(a, b)
    actual = cosine_similarity.compute().type(DTYPE)
    assert pytest.approx(expected) == actual.item()

@pytest.mark.parametrize(
    "average, a, b, expected",
    [
        (
            None,
            a := torch.ones((1, N), dtype=DTYPE, device=DEVICE),
            a,
            torch.zeros_like(a),
        ),
        (
            "macro",
            a,
            a,
            torch.tensor(0.0, dtype=DTYPE, device=DEVICE),
        ),
        (
            None,
            a,
            a + (
                salt := torch.randint(
                    low=10, high=100, size=(1,), dtype=DTYPE, device=DEVICE)
            ),
            torch.tile(salt, (1, N)),
        )
    ]
)
def test_mean_absolute_error(average, a, b, expected):
    mae = M.MeanAbsoluteError(average=average, device=DEVICE)
    mae.update(a, b)
    actual = mae.compute()
    assert torch.isclose(expected, actual).all()

@pytest.mark.parametrize(
    "kernel_size, x, y, expected",
    [
        (
            7,
            x := torch.rand((1, 2, 10, 10), dtype=DTYPE, device=DEVICE),
            x,
            torch.ones(2, dtype=DTYPE, device=DEVICE),
        ),
        (
            7,
            x := torch.rand((1, 1, 10, 10), dtype=DTYPE, device=DEVICE),
            x,
            torch.ones(1, dtype=DTYPE, device=DEVICE),
        ),
        (
            3,
            x := torch.zeros((1, 2, 10, 10), dtype=DTYPE, device=DEVICE),
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
