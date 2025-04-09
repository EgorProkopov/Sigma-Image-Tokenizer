import torch
import lightning.pytorch as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

from src.models.transformer import VisionTransformer
from src.models.transformer import SVDViT


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

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_accuracy.compute()
        prec = self.val_precision.compute()
        rec = self.val_recall.compute()
        f1 = self.val_f1.compute()

        self.log("val_accuracy_epoch", acc)
        self.log("val_precision_epoch", prec)
        self.log("val_recall_epoch", rec)
        self.log("val_f1_epoch", f1)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

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

class SVDViTLightingModule(CustomLightningModule):
    def __init__(self, model_hparams, criterion, lr, log_step=1000):
        model = SVDViT(
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

