import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import FFNN


class DecoderScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.scale = 1 / (qkv_dim ** 0.5)
        self.query_proj = nn.Linear(embed_dim, qkv_dim)
        self.key_proj = nn.Linear(embed_dim, qkv_dim)
        self.value_proj = nn.Linear(embed_dim, qkv_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ V


class DecoderMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.attns = nn.ModuleList([
            DecoderScaledDotProductAttention(embed_dim, qkv_dim, dropout_rate)
            for _ in range(n_heads)
        ])
        self.projection = nn.Linear(n_heads * qkv_dim, embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        heads_output = [attn(queries, keys, values, attn_mask) for attn in self.attns]
        concat = torch.cat(heads_output, dim=-1)
        return self.projection(concat)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        self_attn_p: float = 0.1,
        mlp_p: float = 0.1,
    ):
        super().__init__()
        self.self_attn = DecoderMultiHeadAttention(n_heads, embed_dim, qkv_dim, self_attn_p)
        self.norm_self = nn.LayerNorm(embed_dim)
        
        self.ffn = FFNN(embed_dim, mlp_hidden_size, mlp_p)
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        T = target.size(1)
        mask = torch.tril(torch.ones(T, T, device=target.device)).bool()

        res = target
        x = self.norm_self(target)
        x = self.self_attn(x, x, x, mask) + res

        res = x
        x = self.norm_ffn(x)
        x = self.ffn(x) + res

        return x


class TransformerDecoderCABlock(nn.Module):
    def __init__(
        self,
        n_heads: int = 12,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        self_attn_p: float = 0.1,
        cross_attn_p: float = 0.1,
        mlp_p: float = 0.1,
    ):
        super().__init__()
        self.self_attn = DecoderMultiHeadAttention(n_heads, embed_dim, qkv_dim, self_attn_p)
        self.norm_self = nn.LayerNorm(embed_dim)

        self.cross_attn = DecoderMultiHeadAttention(n_heads, embed_dim, qkv_dim, cross_attn_p)
        self.norm_cross = nn.LayerNorm(embed_dim)

        self.ffn = FFNN(embed_dim, mlp_hidden_size, mlp_p)
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        T = target.size(1)
        causal_mask = torch.tril(torch.ones(T, T, device=target.device)).bool()

        res = target
        x = self.norm_self(target)
        x = self.self_attn(x, x, x, causal_mask) + res

        res = x
        x = self.norm_cross(x)
        x = self.cross_attn(x, memory, memory) + res

        res = x
        x = self.norm_ffn(x)
        x = self.ffn(x) + res
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 3072,
        n_layers: int = 12,
        n_heads: int = 12
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                n_heads, embed_dim, qkv_dim, mlp_hidden_size
            ) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    B, T_tgt, T_src, D = 2, 10, 15, 768
    tgt = torch.randn(B, T_tgt, D)
    mem = torch.randn(B, T_src, D)
    decoder = TransformerDecoder()
    out = decoder(tgt, mem)
    print(out.shape)
