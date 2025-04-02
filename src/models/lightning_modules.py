import torch
import lightning.pytorch as pl

from src.models.transformer import VisionTransformer
from src.models.transformer import SVDViT


class ViTLightingModule(pl.LightningModule):
    def __init__(self, model_hparams, criterion, lr):
        super().__init__()
        self.save_hyperparameters(model_hparams)
        self.model = VisionTransformer(
            image_size=self.hparams["image_size"],
            patch_size=self.hparams["patch_size"],
            in_channels=self.hparams["in_channels"],
            embedding_dim=self.hparams["embedding_dim"],
            qkv_dim=self.hparams["qkv_dim"],
            mlp_hidden_size=self.hparams["mlp_hidden_size"],
            n_layers=self.hparams["n_layers"],
            n_heads=self.hparams["n_heads"],
            n_classes=self.hparams["n_classes"]
        )

        self.criterion=criterion
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label_encoded"]

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SVDViTLightingModule(pl.LightningModule):
    def __init__(self, model_hparams, criterion, lr):
        super().__init__()
        self.save_hyperparameters(model_hparams)
        self.model = SVDViT(
            image_size=self.hparams["image_size"],
            embedding_dim=self.hparams["embedding_dim"],
            dispersion=self.hparams["dispersion"],
            qkv_dim=self.hparams["qkv_dim"],
            mlp_hidden_size=self.hparams["mlp_hidden_size"],
            n_layers=self.hparams["n_layers"],
            n_heads=self.hparams["n_heads"],
            n_classes=self.hparams["n_classes"]
        )

        self.criterion=criterion
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label_encoded"]

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

