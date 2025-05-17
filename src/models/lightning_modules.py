import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

from src.models.losses import GatedLoss
from src.models.transformer import VisionTransformer, MFFTViT, WaveletViT, MFFTViTRegression
from src.models.transformer import SVDLinearViT, SVDSquareViT
from src.models.transformer import FFTViT
from src.models.transformer import MSVDNoScorerViT, MSVDSigmoidGatingViT


class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, criterion, lr, n_classes, log_step=1000):
        """
        Base class for custom lightning models

        Args:
         - model:  ML model
         - criterion: loss function
         - lr: learning rate
         - n_classes: num_classes
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.log_step = log_step

        self.train_accuracy = Accuracy(num_classes=n_classes, task="multiclass")
        self.train_precision = Precision(num_classes=n_classes, average='macro', task="multiclass")
        self.train_recall = Recall(num_classes=n_classes, average='macro', task="multiclass")
        self.train_f1 = F1Score(num_classes=n_classes, average='macro', task="multiclass")

        self.val_accuracy = Accuracy(num_classes=n_classes, task="multiclass")
        self.val_precision = Precision(num_classes=n_classes, average='macro', task="multiclass")
        self.val_recall = Recall(num_classes=n_classes, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=n_classes, average='macro', task="multiclass")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)

        preds = torch.argmax(outputs, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        if self.global_step % self.log_step == 0 and self.global_step != 0:
            acc = self.train_accuracy.compute()
            prec = self.train_precision.compute()
            rec = self.train_recall.compute()
            f1 = self.train_f1.compute()

            self.log("train_accuracy", acc, prog_bar=True)
            self.log("train_precision", prec, prog_bar=True)
            self.log("train_recall", rec, prog_bar=True)
            self.log("train_f1", f1, prog_bar=True)

            self.train_accuracy.reset()
            self.train_precision.reset()
            self.train_recall.reset()
            self.train_f1.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_accuracy.compute()
        prec = self.val_precision.compute()
        rec = self.val_recall.compute()
        f1 = self.val_f1.compute()

        self.log("val_accuracy_epoch", acc, prog_bar=True)
        self.log("val_precision_epoch", prec, prog_bar=True)
        self.log("val_recall_epoch", rec, prog_bar=True)
        self.log("val_f1_epoch", f1, prog_bar=True)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class CustomRegressionLightningModule(pl.LightningModule):
    def __init__(self, model, criterion, lr, log_step=1000):
        """
        Base class for custom lightning models

        Args:
         - model:  ML model
         - criterion: loss function
         - lr: learning rate
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.log_step = log_step

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)

        self.log("val_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class ViTLightingModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = VisionTransformer(
            image_size=model_hparams["image_size"],
            patch_size=model_hparams["patch_size"],
            in_channels=model_hparams["in_channels"],
            embedding_dim=model_hparams["embedding_dim"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()


class SVDLinearViTLightingModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = SVDLinearViT(
            image_size=model_hparams["image_size"],
            embedding_dim=model_hparams["embedding_dim"],
            dispersion=model_hparams["dispersion"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()


class SVDSquareViTLightingModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = SVDSquareViT(
            image_size=model_hparams["image_size"],
            embedding_dim=model_hparams["embedding_dim"],
            dispersion=model_hparams["dispersion"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()


class FFTViTLightingModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = FFTViT(
            image_size=model_hparams["image_size"],
            embedding_dim=model_hparams["embedding_dim"],
            filter_size=model_hparams["filter_size"],
            num_bins=model_hparams["num_bins"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()


class MSVDNoScorerViTLightingModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = MSVDNoScorerViT(
            num_channels=model_hparams["num_channels"],
            pixel_unshuffle_scale_factors=model_hparams["pixel_unshuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"],
            dispersion=model_hparams["dispersion"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()


class MSVDSigmoidGatingViTLightningModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = MSVDSigmoidGatingViT(
            num_channels=model_hparams["num_channels"],
            pixel_unshuffle_scale_factors=model_hparams["pixel_unshuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"],
            selection_mode=model_hparams["selection_mode"],
            top_k=model_hparams["top_k"],
            dispersion=model_hparams["dispersion"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)

        self.alpha = model_hparams["alpha"]
        self.gated_loss = GatedLoss()

        self.save_hyperparameters()

    def forward(self, x):
        msvd_output = self.model(x)
        return msvd_output

    def training_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        msvd_output = self.forward(images)
        logits = msvd_output["logits"]
        scores = msvd_output["scores"]
        gates = torch.nn.functional.sigmoid(scores)

        ce_loss = self.criterion(logits, labels)
        auxiliary_loss = self.alpha * self.gated_loss(gates)
        loss = ce_loss + auxiliary_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("auxiliary_loss", auxiliary_loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        if self.global_step % self.log_step == 0 and self.global_step != 0:
            acc = self.train_accuracy.compute()
            prec = self.train_precision.compute()
            rec = self.train_recall.compute()
            f1 = self.train_f1.compute()

            self.log("train_accuracy", acc, prog_bar=True)
            self.log("train_precision", prec, prog_bar=True)
            self.log("train_recall", rec, prog_bar=True)
            self.log("train_f1", f1, prog_bar=True)

            self.train_accuracy.reset()
            self.train_precision.reset()
            self.train_recall.reset()
            self.train_f1.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        msvd_output = self.forward(images)
        logits = msvd_output["logits"]
        scores = msvd_output["scores"]
        gates = torch.nn.functional.sigmoid(scores)

        ce_loss = self.criterion(logits, labels)
        auxiliary_loss = self.alpha * self.gated_loss(gates)
        loss = ce_loss + auxiliary_loss

        preds = torch.argmax(logits, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=False)
        return loss


class MFFTViTLightningModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = MFFTViT(
            num_channels=model_hparams["num_channels"],
            pixel_unshuffle_scale_factors=model_hparams["pixel_unshuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"],
            filter_size=model_hparams["filter_size"],
            energy_ratio=model_hparams["energy_ratio"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()

    def forward(self, x):
        mfft_output = self.model(x)
        return mfft_output

    def training_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        msvd_output = self.forward(images)
        logits = msvd_output["logits"]
        filter_size = msvd_output["filter_size"]

        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_filter_size", filter_size, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        if self.global_step % self.log_step == 0 and self.global_step != 0:
            acc = self.train_accuracy.compute()
            prec = self.train_precision.compute()
            rec = self.train_recall.compute()
            f1 = self.train_f1.compute()

            self.log("train_accuracy", acc, prog_bar=True)
            self.log("train_precision", prec, prog_bar=True)
            self.log("train_recall", rec, prog_bar=True)
            self.log("train_f1", f1, prog_bar=True)

            self.train_accuracy.reset()
            self.train_precision.reset()
            self.train_recall.reset()
            self.train_f1.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        # images = batch["image"]
        # labels = batch["label_encoded"]

        images, labels = batch

        msvd_output = self.forward(images)
        logits = msvd_output["logits"]
        filter_size = msvd_output["filter_size"]

        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        # labels = torch.argmax(labels, dim=1)

        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_filter_size", filter_size, prog_bar=True)
        return loss


class MFFTViTRegressionLightningModule(CustomRegressionLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = MFFTViTRegression(
            num_channels=model_hparams["num_channels"],
            pixel_unshuffle_scale_factors=model_hparams["pixel_unshuffle_scale_factors"],
            embedding_dim=model_hparams["embedding_dim"],
            filter_size=model_hparams["filter_size"],
            energy_ratio=model_hparams["energy_ratio"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
        )
        super().__init__(model, criterion, lr, log_step=log_step)
        self.save_hyperparameters()

        self.train_loss_accum = 0.0
        self.val_loss_accum = 0.0

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()

        outputs = self.forward(images)
        preds = outputs["logits"]
        batch_size, _ = preds.shape
        preds = torch.reshape(preds, (batch_size, ))

        loss = self.criterion(preds, labels)
        self.train_loss_accum += loss.item()


        if (batch_idx + 1) % self.log_step == 0:
            avg_loss = self.train_loss_accum / self.log_step
            self.log("train_loss", avg_loss, prog_bar=True)
            self.train_loss_accum = 0.0

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.float()

        outputs = self.forward(images)
        preds = outputs["logits"]

        batch_size, _ = preds.shape
        preds = torch.reshape(preds, (batch_size,))
        loss = self.criterion(preds, labels)
        self.val_loss_accum += loss.item()

        if (batch_idx + 1) % self.log_step == 0:
            avg_loss = self.val_loss_accum / self.log_step
            self.log("val_loss", avg_loss, prog_bar=True)
            self.val_loss_accum = 0.0

        return loss


class WaveletViTLightningModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = WaveletViT(
            wavelet=model_hparams["wavelet"],
            bit_planes=model_hparams["bit_planes"],
            final_threshold=model_hparams["final_threshold"],
            embedding_dim=model_hparams["embedding_dim"],
            qkv_dim=model_hparams["qkv_dim"],
            mlp_hidden_size=model_hparams["mlp_hidden_size"],
            n_layers=model_hparams["n_layers"],
            n_heads=model_hparams["n_heads"],
            n_classes=model_hparams["n_classes"],
            max_seq_len=model_hparams["max_seq_len"]
        )
        super().__init__(model, criterion, lr, n_classes=model_hparams["n_classes"], log_step=log_step)
        self.save_hyperparameters()
