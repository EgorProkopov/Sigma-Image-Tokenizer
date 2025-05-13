import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import lightning.pytorch as pl

from src.models.decoder_transformer import ViTTransformerDecoder


class CustomDecoderLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        lr: float,
        num_generate: int = 2,
        log_step: int = 100
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_generate = num_generate
        self.log_step = log_step

        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, _ = batch                              # [B, C, H, W]
        recon = self.forward(images)                   # [B, C, H, W]
        loss = self.criterion(recon, images)           # nn.MSELoss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        recon = self.forward(images)
        loss = self.criterion(recon, images)
        self.log("val_loss", loss, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, "generated")
        os.makedirs(save_dir, exist_ok=True)

        for img_idx in range(self.num_generate):
            images_seq = self.model.generate_image(device=self.device)
            final_img = images_seq[-1]             # [C, H, W]
            mn, mx = final_img.min(), final_img.max()
            img_norm = (final_img - mn) / (mx - mn + 1e-8)
            fname = f"epoch{self.current_epoch}_img{img_idx}.png"
            save_image(img_norm, os.path.join(save_dir, fname))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ViTDecoderLightningModule(CustomDecoderLightningModule):
    def __init__(
            self,
            model_hparams,
            criterion,
            lr,
            log_step=100
    ):
        model = ViTTransformerDecoder(
            image_size=model_hparams["image_size"],
            patch_size=model_hparams["patch_size"],
            in_channels=model_hparams["in_channels"],
            embedding_dim=model_hparams["embedding_dim"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
        )
        super().__init__(model, criterion, lr, log_step=log_step)
        self.save_hyperparameters()

