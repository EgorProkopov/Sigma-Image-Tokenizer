from omegaconf import DictConfig

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets


from src.models.decoder_lightning_modules import CustomDecoderLightningModule


def train_model(
        model: CustomDecoderLightningModule,
        config: DictConfig,
        transform: torchvision.transforms.transforms.Compose
):
    model_hparams = config["model_hparams"]
    train_hparams = config["train_hparams"]

    train_dataset = datasets.ImageFolder(r"F:\research\data\intel_multiclass\seg_train", transform=transform)
    val_dataset = datasets.ImageFolder(r"F:\research\data\intel_multiclass\seg_test", transform=transform)

    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Размер тренировочного датасета: {train_size}")
    print(f"Размер валидационного датасета: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_hparams["train_batch_size"],
        shuffle=True,
        num_workers=train_hparams["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_hparams["val_batch_size"],
        shuffle=False,
        num_workers=train_hparams["num_workers"]
    )

    if train_hparams.get("accelerator") == "cuda":
        torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        max_epochs=train_hparams["max_epoch"],
        accelerator=train_hparams["accelerator"],
        log_every_n_steps=100
    )

    trainer.fit(model, train_loader, val_loader)
