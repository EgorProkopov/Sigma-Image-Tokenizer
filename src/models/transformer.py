import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import TransformerEncoder
from src.tokenizers.vit_tokenization import ViTTokenization


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 200,
    ):
        super().__init__()

        self.tokenization = ViTTokenization(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, n_classes),
            nn.Sigmoid()
        )

    def forward(self, tensor):
        x = self.tokenization(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits

    