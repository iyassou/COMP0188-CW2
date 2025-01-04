import comp0188_cw2.utils as U

import math
import pytest
import torch

@pytest.mark.parametrize(
    "start, end, steps, exception",
    [
        (1, -1, 1, ValueError), # start greater than end
        (0, 1, 0, ValueError),  # bad steps
        (0, 1, -1, ValueError), # bad steps
    ]
)
def test_cosine_annealer_bad_inputs(start, end, steps, exception):
    with pytest.raises(exception):
        U.CosineAnnealer(param=torch.tensor(0.0), start=start, end=end, steps=steps)

@pytest.mark.parametrize(
    "start, end, steps",
    [
        (0, 10, 1),
        (5, 10, 20),
        (0, 1, 100),
        (0, 2, 100),
        (0, 100, 100),
    ]
)
def test_cosine_annealer(start, end, steps):
    def cosine_func(step: int, total_steps: int):
        step %= total_steps
        return (math.cos(math.pi * (step / total_steps - 1.)) + 1.) / 2.
    
    x = torch.tensor(0.0)
    annealer = U.CosineAnnealer(param=x, start=start, end=end, steps=steps)
    assert pytest.approx(start) == x.item()

    for step in range(10 * steps):
        expected = start + cosine_func(step, steps) * (end - start)
        assert pytest.approx(expected, abs=1e-5) == x.item()
        annealer.step()
