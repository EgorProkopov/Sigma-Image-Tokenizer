import os

import torch
import torch.nn as nn
import torchvision.utils as vutils
import lightning.pytorch as pl

from src.tokenizers.modified_svd_tokenizer import MSVDSigmoidGatingTokenizer
from src.detokenizers.msvd_detokenizer import MSVDSigmoidGatingDetokenizer


class MSVDAutoencoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_chanels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            pixel_shuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            selection_mode: str = "full",
            top_k: int = 128,
            dispersion: float = 0.9,
    ):
        super().__init__()

        self.tokenizer = MSVDSigmoidGatingTokenizer(
            in_channels=in_channels,
            pixel_unshuffle_scale_factors=pixel_unshuffle_scale_factors,
            embedding_dim=embedding_dim,
            selection_mode=selection_mode,
            dispersion=dispersion,
            top_k=top_k
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(embedding_dim)
        )

        self.detokenizer = MSVDSigmoidGatingDetokenizer(
            out_channels=out_chanels,
            pixel_shuffle_scale_factors=pixel_shuffle_scale_factors,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        msvd_output = self.tokenizer(x)
        tokens = msvd_output["tokens"]
        scores = msvd_output["scores"]
        tokens = tokens[:, 1:]

        # print(f"tokens shape: {tokens.shape}")
        proj_tokens = self.bottleneck(tokens)
        detokenizer_output = self.detokenizer(proj_tokens)
        return detokenizer_output

    @torch.no_grad()
    def generate(self, tokens):
        output = self.detokenizer(tokens)
        return output


class MSVDAutoencoderLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_hparams, criterion, lr, log_step=1000
    ):
        super().__init__()
        self.save_hyperparameters()

        self.log_dir    = "lightning_logs"
        self.images_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.tokenizer   = MSVDSigmoidGatingTokenizer(
            in_channels=model_hparams["in_channels"],
            pixel_unshuffle_scale_factors=model_hparams["pixel_unshuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"],
            selection_mode=model_hparams["selection_mode"],
            dispersion=model_hparams["dispersion"],
            top_k=model_hparams["top_k"]
        )
        self.bottleneck  = nn.Sequential(
            nn.Linear(model_hparams["embedding_dim"],
                      model_hparams["embedding_dim"]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
        )
        self.detokenizer = MSVDSigmoidGatingDetokenizer(
            out_channels=model_hparams["out_channels"],
            pixel_shuffle_scale_factors=model_hparams["pixel_shuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"]
        )

        self.criterion     = criterion
        self.lr            = lr
        self.log_step      = log_step
        self.train_accum_loss = 0.0
        self.val_accum_loss   = 0.0

        self._val_proj_tokens = None

    def forward(self, x):
        msvd_output = self.tokenizer(x)
        tokens = msvd_output["tokens"][:, 1:]
        proj_tokens = self.bottleneck(tokens)
        return self.detokenizer(proj_tokens)

    @torch.no_grad()
    def generate(self, proj_tokens):
        return self.detokenizer(proj_tokens)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out   = self(x)
        img   = out["image"]
        loss  = self.criterion(img, x)
        self.train_accum_loss += loss.item()

        if (batch_idx + 1) % self.log_step == 0:
            avg = self.train_accum_loss / self.log_step
            self.train_accum_loss = 0.0
            self.log("train_loss", avg, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out      = self(x)
        img      = out["image"]
        val_loss = self.criterion(img, x)
        self.val_accum_loss += val_loss.item()

        if (batch_idx + 1) % self.log_step == 0:
            avg = self.val_accum_loss / self.log_step
            self.val_accum_loss = 0.0
            self.log("val_loss", avg, prog_bar=True)

        if batch_idx == 0 and self._val_proj_tokens is None:
            tok    = self.tokenizer(x)["tokens"][:, 1:]
            proj   = self.bottleneck(tok)
            self._val_proj_tokens = proj[:2].detach()

        return val_loss

    def on_validation_epoch_end(self):
        if self._val_proj_tokens is not None:
            gen_inputs = torch.randn(self._val_proj_tokens.shape, device=self._val_proj_tokens.device )
            output = self.generate(gen_inputs)
            imgs   = output["image"]
            fname  = os.path.join(
                self.images_dir,
                f"reconstructions_epoch_{self.current_epoch}.png"
            )
            vutils.save_image(
                imgs,
                fname,
                nrow=2,
                normalize=False,
                value_range=(0, 1)
            )
            print(self._val_proj_tokens)
            self._val_proj_tokens = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    batch = torch.randn(8, 3, 256, 256)

    model = MSVDAutoencoder()
    output = model(batch)
    print(output['image'].shape)

    batch_tokens = torch.randn(8, 256, 768)
    output = model.generate(batch_tokens)
    print(output['image'].shape)

