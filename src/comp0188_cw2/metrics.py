import piqa
import torch
import torcheval.metrics

from typing import get_args, Iterable, Literal, Optional

Averaging = Literal[None, "macro"]

class CosineSimilarity(torcheval.metrics.Metric[torch.Tensor]):
    def __init__(
            self,
            eps: Optional[float]=1e-8,
            device: Optional[torch.device]=None):
        super().__init__(device=device)
        self.eps = eps
        self._add_state("cossim_sum", torch.tensor(0.0, device=device))
        self._add_state("count", torch.tensor(0, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "CosineSimilarity":
        input = input.to(self.device)
        target = target.to(self.device)
        cos = torch.nn.functional.cosine_similarity(input, target, dim=1, eps=self.eps)
        self.cossim_sum += cos.sum()
        self.count += cos.numel()
        return self

    def compute(self) -> torch.Tensor:
        return self.cossim_sum / self.count

    def merge_state(self, metrics: Iterable["CosineSimilarity"]):
        for metric in metrics:
            self.cossim_sum += metric.cossim_sum.to(self.device)
            self.count += metric.count.to(self.device)
        return self
    
class MeanAbsoluteError(torcheval.metrics.Metric[torch.Tensor]):
    def __init__(
            self,
            average: Optional[Averaging]=None,
            channels: Optional[int]=None,
            device: Optional[torch.device]=None):
        if average not in get_args(Averaging):
            raise ValueError(
                f"unrecognised `average` {repr(average)}, "
                f"must be one of: {', '.join(map(repr, get_args(Averaging)))}"
            )
        super().__init__(device=device)
        if average is None and channels is None:
            raise ValueError("must supply `channels` if `average` is None")
        self.average = average
        self._add_state("mae_sum", torch.zeros(channels or 1, device=device))
        self._add_state("count", torch.zeros(channels or 1, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "MeanAbsoluteError":
        input = input.to(self.device)
        target = target.to(self.device)
        mae = torch.abs(input - target)
        self.mae_sum += mae.sum(0)
        self.count += mae.size(0)
        return self
    
    def compute(self) -> torch.Tensor:
        return self.mae_sum / self.count
    
    def merge_state(self, metrics: Iterable["MeanAbsoluteError"]):
        for metric in metrics:
            self.mae_sum += metric.mae_sum.to(self.device)
            self.count += metric.count.to(self.device)
        return self

class MeanFloat32(torcheval.metrics.Metric[torch.Tensor]):
    """No float64 support on MPS so torcheval.metrics.Mean raises a TypeError"""
    def __init__(self, device: torch.device):
        super().__init__(device=device)
        self._add_state("total", torch.tensor(0.0, device=device, dtype=torch.float32))
        self._add_state("count", torch.tensor(0.0, device=device, dtype=torch.float32))

    def update(self, input: torch.Tensor) -> "MeanFloat32":
        self.total += input.sum(0)
        self.count += input.numel()
        return self

    def compute(self) -> torch.Tensor:
        return self.total / self.count

    def merge_state(self, metrics: Iterable["MeanFloat32"]) -> "MeanFloat32":
        for metric in metrics:
            self.total += metric.total.to(self.device)
            self.count += metric.count.to(self.device)
        return self

class StructuralSimilarity(torcheval.metrics.Metric[torch.Tensor]):
    def __init__(self, kernel_size: int, channels: int, value_range: float=1., device: Optional[torch.device]=None) -> None:
        super().__init__(device=device)
        self.kernel = piqa.ssim.gaussian_kernel(kernel_size).repeat(channels, 1, 1)
        self.kernel = self.kernel.to(device)
        self.value_range = value_range
        self._add_state("ssim_sum", torch.zeros(channels, device=device))
        self._add_state("count", torch.zeros(channels, dtype=int, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "StructuralSimilarity":
        input = input.to(self.device)
        target = target.to(self.device)
        ssim, _ = piqa.ssim.ssim(input, target, self.kernel, channel_avg=False, value_range=self.value_range)
        self.ssim_sum += ssim.sum(0)
        self.count += ssim.size(0)
        return self

    def compute(self) -> torch.Tensor:
        return self.ssim_sum / self.count
    
    def merge_state(self, metrics: Iterable["StructuralSimilarity"]):
        for metric in metrics:
            self.ssim_sum += metric.ssim_sum.to(self.device)
            self.count += metric.count.to(self.device)
        return self