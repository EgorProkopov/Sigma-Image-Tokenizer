import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.scale = 1 / (qkv_dim ** 0.5)
        self.q_linear = nn.Linear(embed_dim, qkv_dim)
        self.k_linear = nn.Linear(embed_dim, qkv_dim)
        self.v_linear = nn.Linear(embed_dim, qkv_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.attns = torch.nn.ModuleList([
            ScaledDotProductAttention(embed_dim=embed_dim, qkv_dim=qkv_dim, dropout_rate=dropout_rate)
            for _ in range(n_heads)
        ])

        self.projection = torch.nn.Linear(n_heads * qkv_dim, embed_dim)

    def forward(self, x):
        heads_output = [attn(x) for attn in self.attns]
        concatenated = torch.cat(heads_output, dim=-1)
        output = self.projection(concatenated)
        return output


class FFNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_size: int = 3072,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.mlp(x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            n_heads: int = 12,
            qkv_dim: int = 64,
            embed_dim: int = 768,
            mlp_hidden_size: int = 3072,
            attn_p: float = 0.1,
            mlp_p: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, qkv_dim=qkv_dim, dropout_rate=attn_p)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FFNN(embed_dim=embed_dim, mlp_hidden_size=mlp_hidden_size, dropout_rate=mlp_p)

    def forward(self, x):
        attn_output = self.attention(self.norm1(x)) + x
        output = self.mlp(self.norm2(attn_output)) + attn_output
        return output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            *[EncoderBlock(
                n_heads=n_heads,
                qkv_dim=qkv_dim,
                embed_dim=embed_dim,
                mlp_hidden_size=mlp_hidden_size
            ) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.encoder(x)
        x = x[:, 0]
        return x


class TransformerClassifierModel(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12,
        n_classes: int = 1_000,
    ):
        super().__init__()

        # Image Tokenizer
        self.tokenizer = None
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, image):
        x = self.tokenizer(image)
        x = self.transformer_encoder(x)
        logits = self.classifier(x)
        return logits
    