import math

import torch
import torch.nn as nn


class ViTDetokenizer(nn.Module):
    def __init__(
            self,
            patch_size: int = 16,
            out_channels: int = 3,
            embed_dim: int = 768,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size

        self.projector = nn.Linear(
            in_features=embed_dim,
            out_features=out_channels * (patch_size ** 2)
        )

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        flat_patches = self.projector(x)
        patches_seq = flat_patches.view(
            batch_size, num_tokens,
            self.out_channels,
            self.patch_size,
            self.patch_size
        )

        return patches_seq

    def reconstruct_image(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels, patch_size, patch_size = patches.shape

        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Wrong num of patches: {num_patches}")

        patches = patches.view(batch_size, grid_size, grid_size, channels, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)

        images = patches.contiguous().view(
            batch_size,
            channels,
            grid_size * patch_size, grid_size * patch_size
        )
        return images



