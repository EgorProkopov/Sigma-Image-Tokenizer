import torch
import torch.nn as nn


class SVDTokenizer(nn.Module):
    def __init__(self, dispersion=0.9):
        super().__init__()

        self.dispersion = dispersion

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        U, S, Vh = torch.linalg.svd(image, full_matrices=True)
        V = Vh.mH

        S_squared = torch.square(S)
        S_squared = S_squared.sum(dim=0, keepdim=True)

        S_cumsum = torch.cumsum(S_squared, dim=1)
        total_sum = S_squared.sum()
        threshold = self.dispersion * total_sum

        mask = (S_cumsum >= threshold).squeeze()
        mask = torch.where(mask)[0]

        if mask.numel() > 0:
            rank = mask[0].item() + 1
        else:
            rank = S_squared.shape[1]

        approx_U, approx_S, approx_V = U[:, :, :rank], S[:, :rank], V[:, :, :rank]

        return approx_U, approx_S, approx_V, rank

    @staticmethod
    def reconstruct_image(approx_U, approx_S, approx_V):
        US = approx_U * approx_S.unsqueeze(1)  # Shape: [channels, m, rank]
        Vt = approx_V.transpose(1, 2)
        approx_image = torch.bmm(US, Vt)

        return approx_image
