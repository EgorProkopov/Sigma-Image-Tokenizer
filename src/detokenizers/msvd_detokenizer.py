import math
import torch
import torch.nn as nn


class MSVDSigmoidGatingDetokenizer(nn.Module):
    def __init__(
            self,
            out_channels: int =3,
            pixel_shuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768
    ):
        super().__init__()

        self.out_channels = out_channels
        self.pixel_shuffle_scale_factors = pixel_shuffle_scale_factors
        self.embedding_dim = embedding_dim

        self.decoder = nn.ModuleList()
        current_channels = embedding_dim
        for scale in self.pixel_shuffle_scale_factors:
            self.decoder.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                )
            )
            self.decoder.append(nn.InstanceNorm2d(current_channels))
            self.decoder.append(nn.LeakyReLU())
            self.decoder.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                )
            )
            self.decoder.append(nn.InstanceNorm2d(current_channels))
            self.decoder.append(nn.LeakyReLU())
            self.decoder.append(nn.PixelShuffle(upscale_factor=scale))

            current_channels = int(current_channels // (scale ** 2))

        self.decoder = nn.Sequential(*self.decoder)

        self.projection = nn.Conv2d(
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )

    def _get_raw_image(self, tokens):
        batch_size, seq_len, token_size = tokens.shape

        sqrt_seq_len = int(math.sqrt(seq_len))
        if sqrt_seq_len * sqrt_seq_len != seq_len:
            raise ValueError(f"seq_len={seq_len} not a square")

        tokens = tokens.permute(0, 2, 1)
        raw_tensor = tokens.reshape(batch_size, token_size, sqrt_seq_len, sqrt_seq_len)

        raw_image = self.decoder(raw_tensor)

        return raw_image

    def forward(self, tokens):
        raw_image = self._get_raw_image(tokens)
        image = self.projection(raw_image)

        return {"image": image}


if __name__ == "__main__":
    batch = torch.randn(8, 256, 768)
    detokenizer = MSVDSigmoidGatingDetokenizer()
    out = detokenizer(batch)
    print(out['image'].shape)
