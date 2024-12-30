import torch

class CosineAnnealer:
    def __init__(self, param: torch.Tensor, start: float, end: float, steps: int):
        if start > end:
            raise ValueError(f"cannot have {start=} > {end=}")
        if steps < 1:
            raise ValueError(f"need steps > 1, got {steps=}")
        self.param = param
        self.start = start
        self.end = end
        self.steps = steps

        self.param.fill_(start)
        self._step = torch.tensor(0)

    def step(self):
        self._step = (self._step + 1) % self.steps
        fraction = (torch.cos(torch.pi * (self._step / self.steps - 1.)) + 1.) / 2.
        self.param.fill_(self.start + fraction * (self.end - self.start))
