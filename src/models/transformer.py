import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import TransformerEncoder
from src.tokenizers.vit_tokenization import ViTTokenization
from src.tokenizers.svd_tokenizer import SVDTokenizer

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 1024,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = ViTTokenization(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embedding_dim
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits


class SVDViT(nn.Module):
    def __init__(
            self,
            image_size: int = 512,
            embedding_dim: int = 768,
            dispersion: float = 0.999,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = SVDTokenizer(
            image_size=image_size, embedding_dim=embedding_dim, dispersion=dispersion
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits