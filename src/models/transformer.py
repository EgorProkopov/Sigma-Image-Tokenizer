import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import TransformerEncoder
from src.tokenizers.modified_fft_tokenizer import MFFTTokenizer
from src.tokenizers.positional_encoding import PositionalEncoding
from src.tokenizers.vit_tokenization import ViTTokenization
from src.tokenizers.svd_tokenizer import SVDLinearTokenizer, SVDSquareTokenizer
from src.tokenizers.fft_tokenizer import FFTTokenizer
from src.tokenizers.modified_svd_tokenizer import MSVDNoScorerTokenizer, MSVDSigmoidGatingTokenizer
from src.tokenizers.wavelet_tokenizer import WaveletTokenizer


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
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits


class SVDLinearViT(nn.Module):
    def __init__(
            self,
            image_size: int = 512,
            num_channels: int = 3,
            embedding_dim: int = 768,
            dispersion: float = 0.999,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = SVDLinearTokenizer(
            image_size=image_size, num_channels=num_channels, embedding_dim=embedding_dim, dispersion=dispersion
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits


class SVDSquareViT(nn.Module):
    def __init__(
            self,
            image_size: int = 512,
            num_channels: int = 3,
            embedding_dim: int = 768,
            dispersion: float = 0.999,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = SVDSquareTokenizer(
            image_size=image_size, num_channels=num_channels, embedding_dim=embedding_dim, dispersion=dispersion
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits


class FFTViT(nn.Module):
    def __init__(
            self,
            image_size: int = 224,
            patch_size: int = 16,
            num_channels: int = 3,
            num_bins: int = 16,
            filter_size: int = 64,
            embedding_dim: int = 768,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = FFTTokenizer(
            image_size=image_size, num_channels=num_channels, num_bins=num_bins, embedding_dim=embedding_dim, filter_size=filter_size
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits


class MSVDNoScorerViT(nn.Module):
    def __init__(
            self,
            num_channels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            dispersion: float = 0.900,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = MSVDNoScorerTokenizer(
            in_channels=num_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            dispersion=dispersion,
            embedding_dim=embedding_dim,
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        x = self.tokenizer(tensor)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits


class MSVDSigmoidGatingViT(nn.Module):
    def __init__(
            self,
            num_channels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            selection_mode: str = "full",
            top_k: int = 25,
            dispersion: float = 0.900,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = MSVDSigmoidGatingTokenizer(
            in_channels=num_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            selection_mode=selection_mode,
            dispersion=dispersion,
            top_k=top_k
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        msvd_output = self.tokenizer(tensor)
        tokens = msvd_output["tokens"]
        scores = msvd_output["scores"]
        x = self.transformer_encoder(tokens)
        x = x[:, 0]
        logits = self.classifier(x)
        return {"logits": logits, "scores": scores}


class MFFTViT(nn.Module):
    def __init__(
            self,
            num_channels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            filter_size: int = 128,
            energy_ratio: float = 0.900,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,
    ):
        super().__init__()

        self.tokenizer = MFFTTokenizer(
            in_channels=num_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            filter_size=filter_size,
            energy_ratio=energy_ratio
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, tensor):
        mfft_output = self.tokenizer(tensor)
        tokens = mfft_output["tokens"]
        filter_size = mfft_output["filter_size"]
        x = self.transformer_encoder(tokens)
        x = x[:, 0]
        logits = self.classifier(x)
        return {"logits": logits, "filter_size": filter_size}


class MFFTViTRegression(nn.Module):
    def __init__(
            self,
            num_channels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            filter_size: int = 128,
            energy_ratio: float = 0.900,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
    ):
        super().__init__()

        self.tokenizer = MFFTTokenizer(
            in_channels=num_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            filter_size=filter_size,
            energy_ratio=energy_ratio
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, tensor):
        mfft_output = self.tokenizer(tensor)
        tokens = mfft_output["tokens"]
        filter_size = mfft_output["filter_size"]
        x = self.transformer_encoder(tokens)
        x = x[:, 0]
        logits = self.classifier(x)
        return {"logits": logits, "filter_size": filter_size}


class WaveletViT(nn.Module):
    def __init__(
            self,
            # параметры токенизатора
            wavelet: str = 'haar',
            bit_planes: int = 4,
            final_threshold: float = 2 ** -3,
            # параметры трансформера
            embedding_dim: int = 768,
            qkv_dim: int = 64,
            mlp_hidden_size: int = 1024,
            n_layers: int = 12,
            n_heads: int = 12,
            n_classes: int = 1000,

            max_seq_len: int = 1024,
    ):

        super().__init__()

        self.tokenizer = WaveletTokenizer(
            wavelet=wavelet,
            bit_planes=bit_planes,
            final_threshold=final_threshold
        )
        self.vocab_size =  self.tokenizer.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

        self.max_seq_len = max_seq_len

    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = tokens.to(x.device)

        B, L = tokens.shape
        if L > self.max_seq_len:
            tokens = tokens[:, :self.max_seq_len]

        tokens = self.embedding(tokens)

        cls = self.cls_token.to(tokens.device).expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.positional_encoding(tokens)

        encoded = self.transformer_encoder(tokens)
        cls_token = encoded[:, 0, :]
        logits = self.classifier(cls_token)

        return logits
