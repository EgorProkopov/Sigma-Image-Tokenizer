import torch
import torch.nn as nn

from src.detokenizers.vit_detokenizer import ViTDetokenizer
from src.tokenizers.vit_tokenization import ViTTokenization
from src.models.decoder_modules import TransformerDecoder


class ViTTransformerDecoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 768,
        qkv_dim: int = 64,
        mlp_hidden_size: int = 1024,
        n_layers: int = 12,
        n_heads: int = 12
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.tokenizer = ViTTokenization(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embedding_dim
        )
        self.transformer_decoder = TransformerDecoder(
            embed_dim=embedding_dim,
            qkv_dim=qkv_dim,
            mlp_hidden_size=mlp_hidden_size,
            n_layers=n_layers,
            n_heads=n_heads
        )
        self.detokenizer = ViTDetokenizer(
            patch_size=patch_size,
            out_channels=in_channels,
            embed_dim=embedding_dim
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing forward:
         - images: [B, C, H, W]
         - returns recon: [B, C, H, W]
        """
        # 1) токенизируем
        embeds = self.tokenizer(images)             # [B, P+1, E]
        # 2) вход модели — все, кроме последнего эмбеддинга
        inp = embeds[:, :-1, :]                     # [B, P, E]
        # 3) декодируем
        decoded = self.transformer_decoder(inp)     # [B, P, E]
        # 4) реконструкция
        patches = self.detokenizer(decoded)         # [B, P, C, ps, ps]
        recon = self.detokenizer.reconstruct_image(patches)  # [B, C, H, W]
        return recon

    @torch.no_grad()
    def generate_image(self, device: torch.device = None):
        """
        Генерация изображения патч за патчем. Возвращает список промежуточных изображений [step0, step1, ..., final].
        """
        if device is None:
            device = next(self.parameters()).device
        num_patches = (self.image_size // self.patch_size) ** 2
        E = self.embedding_dim

        # 1) Инициализируем CLS-токен случайно
        seq = torch.randn(1, 1, E, device=device)  # [1,1,E]
        images = []

        for step in range(num_patches):
            # 2) Добавляем positional embeddings и прогоняем через декодер
            pos_emb = self.tokenizer.pos_embeddings[:, : seq.size(1), :].to(device)
            inp = seq + pos_emb  # [1, cur_len, E]
            decoded = self.transformer_decoder(inp)  # [1, cur_len, E]

            # 3) Получаем embedding следующего патча и добавляем к seq
            next_emb = decoded[:, -1:, :].clone()  # [1,1,E]
            seq = torch.cat([seq, next_emb], dim=1)  # [1, cur_len+1, E]

            # 4) Падим до длины P+1 и отбрасываем CLS для детокенизации
            pad_len = (num_patches + 1) - seq.size(1)
            if pad_len > 0:
                pad = torch.zeros(1, pad_len, E, device=device)
                seq_padded = torch.cat([seq, pad], dim=1)
            else:
                seq_padded = seq

            patch_seq = seq_padded[:, 1:, :]  # [1, num_patches, E]

            # 5) Получаем патчи и реконструируем изображение
            patches = self.detokenizer(patch_seq)  # [1, num_patches, C, ps, ps]
            img = self.detokenizer.reconstruct_image(patches)  # [1, C, H, W]
            images.append(img.squeeze(0))  # [C, H, W]

        return images
