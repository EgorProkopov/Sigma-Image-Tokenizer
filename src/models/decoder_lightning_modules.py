import os

import torch
from torchvision.utils import save_image
import lightning.pytorch as pl

from src.models.decoder_transformer import ViTTransformerDecoder


class CustomDecoderLightningModule(pl.LightningModule):
    def __init__(
            self,
            model,
            criterion,
            lr,
            log_step=100
    ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.log_step = log_step

        self.num_generate = 2

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, _ = batch

        outputs = self.forward(images)
        predicted_patches = outputs["patches"]
        predicted_images = self.model.detokenizer.reconstruct_image(predicted_patches)
        loss = self.criterion(predicted_images, images)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch

        outputs = self.forward(images)
        predicted_patches = outputs["patches"]
        predicted_images = self.model.detokenizer.reconstruct_image(predicted_patches)

        loss = self.criterion(predicted_images, images)

        self.log("val_loss", loss, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, "generated")
        os.makedirs(save_dir, exist_ok=True)

        for img_idx in range(self.num_generate):
            images_seq = self.model.generate_image(device=self.device)
            final_img = images_seq[-1]  # Tensor [C, H, W]

            img_min, img_max = final_img.min(), final_img.max()
            img_norm = (final_img - img_min) / (img_max - img_min + 1e-8)

            fname = f"epoch{self.current_epoch}_img{img_idx}.png"
            path = os.path.join(save_dir, fname)
            save_image(img_norm, path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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

