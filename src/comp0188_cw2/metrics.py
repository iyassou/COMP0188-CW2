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
        self._add_state("count", torch.tensor(0.0, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "CosineSimilarity":
        input = input.to(self.device)
        target = target.to(self.device)
        cos = torch.nn.functional.cosine_similarity(input, target, dim=1, eps=self.eps)
        if self.sum_cosine_similarity.ndim != cos.ndim:
            self.sum_cosine_similarity = cos
        else:
            self.sum_cosine_similarity += cos
        self.count += 1
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
        super().__init__(device=device)
        self.average = average
        self._add_state("sum_mae", torch.tensor(0.0, device=device))
        self._add_state("count", torch.tensor(0.0, device=device))

    def update(self, input: torch.Tensor, target: torch.Tensor) -> "MeanAbsoluteError":
        input = input.to(self.device)
        target = target.to(self.device)
        mae = torch.abs(input - target)
        if self.average == "macro":
            mae = mae.mean(dim=1)
        if self.sum_mae.ndim != mae.ndim:
            self.sum_mae = mae
        else:
            self.sum_mae += mae
        self.count += 1
        return self
    
    def compute(self) -> torch.Tensor:
        return self.sum_mae / self.count
    
    def merge_state(self, metrics: Iterable["MeanAbsoluteError"]):
        for metric in metrics:
            self.sum_mae += metric.sum_mae.to(self.device)
            self.count += metric.count.to(self.device)
        return self
