import torch
import torch.nn as nn

from src.tokenizers.modified_svd_tokenizer import MSVDSigmoidGatingTokenizer
from src.detokenizers.msvd_detokenizer import MSVDSigmoidGatingDetokenizer


class MSVDAutoencoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_chanels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            pixel_shuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            selection_mode: str = "full",
            top_k: int = 128,
            dispersion: float = 0.9,
    ):
        super().__init__()

        self.tokenizer = MSVDSigmoidGatingTokenizer(
            in_channels=in_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            selection_mode=selection_mode,
            dispersion=dispersion,
            top_k=top_k
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(embedding_dim)
        )

        self.detokenizer = MSVDSigmoidGatingDetokenizer(
            out_channels=out_chanels,
            pixel_shuffle_scale_factors=pixel_shuffle_scale_factors,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        msvd_output = self.tokenizer(x)
        tokens = msvd_output["tokens"]
        scores = msvd_output["scores"]
        tokens = tokens[:, 1:]

        # print(f"tokens shape: {tokens.shape}")
        proj_tokens = self.bottleneck(tokens)
        detokenizer_output = self.detokenizer(proj_tokens)
        return detokenizer_output

    @torch.no_grad()
    def generate(self, tokens):
        output = self.detokenizer(tokens)
        return output


if __name__ == "__main__":
    batch = torch.randn(8, 3, 256, 256)

    model = MSVDAutoencoder()
    output = model(batch)
    print(output['image'].shape)

    batch_tokens = torch.randn(8, 256, 768)
    output = model.generate(batch_tokens)
    print(output['image'])

