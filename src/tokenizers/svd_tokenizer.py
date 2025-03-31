import torch
import torch.nn as nn

from src.tokenizers.positional_encoding import PositionalEncoding


class SVDTokenizer(nn.Module):
    def __init__(self, image_size, embedding_dim, dispersion=0.9):
        super().__init__()

        self.dispersion = dispersion

        self.embedder = nn.Linear(image_size * 2, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def get_approx_svd(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # with torch.no_grad():
        # Shapes: [channels,image_size, rank] and [channels, rank]
        approx_U, approx_S, approx_V, _ = self.get_approx_svd(image)

        approx_U = approx_U.transpose(1, 2)
        approx_V = approx_V.transpose(1, 2)

        approx_S_root = torch.sqrt(approx_S)

        # print(approx_U.shape)
        # print(approx_S_root.shape)
        # print(approx_V.shape)

        approx_US = approx_U * approx_S_root.unsqueeze(-1)  # Shape: [channels, rank, image_size]
        approx_SV = approx_V * approx_S_root.unsqueeze(-1)  # Shape: [channels, rank, image_size]

        approx_US = approx_US.transpose(0, 1)  # Shape: [rank, channels, image_size]
        approx_SV = approx_SV.transpose(0, 1)  # Shape: [rank, channels, image_size]

        raw_embedding = torch.cat([approx_US, approx_SV], dim=-1)  # Shape: [rank, channels, image_size * 2]
        channels, rank, image_size = raw_embedding.shape
        raw_embedding = torch.reshape(raw_embedding, (channels * rank, image_size))

        embeddings = self.embedder(raw_embedding)
        print(self.cls_token.shape)
        print(embeddings.shape)
        embeddings = torch.cat([self.cls_token, embeddings], dim=0)
        positional_embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings + positional_embeddings
        return embeddings


    @staticmethod
    def reconstruct_image(approx_U, approx_S, approx_V):
        US = approx_U * approx_S.unsqueeze(1)  # Shape: [channels, image_size, rank]
        Vt = approx_V.transpose(1, 2)
        approx_image = torch.bmm(US, Vt)

        return approx_image
