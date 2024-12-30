import comp0188_cw2.utils as U

import math
import pytest
import torch

def test_cosine_annealer_bad_inputs():
    # start greater than end
    with pytest.raises(ValueError):
        U.CosineAnnealer(
            param=torch.tensor(0.0), start=1, end=-1, steps=1
        )
    # bad_steps
    bad_steps = 0, -1
    for steps in bad_steps:
        with pytest.raises(ValueError):
            U.CosineAnnealer(
                param=torch.tensor(0.0), start=0, end=1, steps=steps,
            )

def test_cosine_annealer_initialisation():
    x = torch.tensor(0.0)
    start = 55
    assert x.item() != start
    _ = U.CosineAnnealer(x, start=start, end=start + 1, steps=1)
    assert x.item() == start

def test_cosine_annealer_stepping():
    x = torch.tensor(0.0)
    start = 5
    end = 10
    steps = 20
    def cosine_func(step: int, total_steps: int):
        step %= total_steps
        return (math.cos(math.pi * (step / total_steps - 1.)) + 1.) / 2.
    expected = [
        start + cosine_func(step, steps) * (end - start)
        for step in range(2 * steps)
    ]
    annealer = U.CosineAnnealer(x, start=start, end=end, steps=steps)
    for expect in expected:
        assert x.item() == pytest.approx(expect)
        annealer.step()
