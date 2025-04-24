import torch
import torch.nn as nn


class GiniLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        n = x.numel()
        if n == 0:
            raise ValueError("GiniLoss: input tensor must have at least one element.")

        min_x = torch.min(x)
        if min_x < 0:
            x = x - min_x

        mu = torch.mean(x) + self.eps

        diff = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
        sum_diff = torch.sum(diff)

        gini = sum_diff / (2 * (n ** 2) * mu)
        return gini
