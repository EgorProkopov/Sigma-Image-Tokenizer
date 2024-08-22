import scipy.linalg as linalg
import torch
import torch.nn as nn


class SVDVanilla:
    def __call__(self, x):
        U, s, VT = linalg.svd(x, compute_uv=True)
        return U, s, VT


class SVDTokenizer(nn.Module):
    def __init__(self, svd_algorithm, input_dim, model_dim, bias=False):
        super().__init__()
        self.svd = svd_algorithm
        self.projection = nn.Linear(in_features=input_dim, out_features=model_dim, bias=bias)

    def forward(self, x):
        # что-то вроде псевдокода, я хз пока как это нормально написать
        x = torch.apply(lambda a: self.svd(x), x)
        U, s, VT = x[:, 0], x[:, 1], x[:, 2]
        tokens = self.projection(torch.cat(U, s, VT))
        return tokens
