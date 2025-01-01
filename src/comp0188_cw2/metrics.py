import piqa
import torch
import torcheval.metrics

from typing import (
    Iterable,
    Optional,
)

class CosineSimilarity(torcheval.metrics.Metric[torch.Tensor]):
    def __init__(
            self,
            eps: Optional[float]=1e-8,
            device: Optional[torch.device]=None):
        super().__init__(device=device)
        self.eps = eps
        self._add_state("sum_cosine_similarity", torch.tensor(0.0, device=device))
        self._add_state("count", torch.tensor(0, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "CosineSimilarity":
        input = input.to(self.device)
        target = target.to(self.device)
        cos = torch.nn.functional.cosine_similarity(input, target, dim=1, eps=self.eps)
        self.sum_cosine_similarity += cos.sum()
        self.count += cos.numel()
        return self

    def compute(self) -> torch.Tensor:
        return self.sum_cosine_similarity / self.count

    def merge_state(self, metrics: Iterable["CosineSimilarity"]):
        for metric in metrics:
            self.sum_cosine_similarity += metric.sum_cosine_similarity.to(self.device)
            self.count += metric.count.to(self.device)
        return self
    
class MeanAbsoluteError(torcheval.metrics.Metric[torch.Tensor]):
    def __init__(
            self,
            average: Optional[str]=None,
            device: Optional[torch.device]=None):
        if average is not None and average != "macro":
            raise ValueError(f"unknown average value: {average}")
        super().__init__(device=device)
        self.average = average
        self._add_state("sum_mae", torch.tensor(0.0, device=device))
        self._add_state("count", torch.tensor(0.0, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "MeanAbsoluteError":
        input = input.to(self.device)
        target = target.to(self.device)
        mae = torch.abs(input - target)
        if self.average is None:
            # bad hack because cba
            if self.sum_mae.numel() != mae.size(1):
                self.sum_mae = mae.sum(dim=0)
            else:
                self.sum_mae += mae.sum(dim=0)
            self.count += mae.size(0)
        else:
            # average == "macro"
            self.sum_mae += mae.sum()
            self.count += mae.numel()
        return self
    
    def compute(self) -> torch.Tensor:
        return self.sum_mae / self.count
    
    def merge_state(self, metrics: Iterable["MeanAbsoluteError"]):
        for metric in metrics:
            self.sum_mae += metric.sum_mae.to(self.device)
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

    def compute(self) -> torch.Tensor:
        return self.ssim_sum / self.count
    
    def merge_state(self, metrics: Iterable["StructuralSimilarity"]):
        for metric in metrics:
            self.ssim_sum += metric.ssim_sum.to(self.device)
            self.count += metric.count.to(self.device)
        return self