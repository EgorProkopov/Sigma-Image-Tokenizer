import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


def train_model(model, config):
    model_hparams = config["model_hparams"]
    train_hparams = config["train_hparams"]

    transform = transforms.Compose([
        transforms.Resize((
            int(model_hparams["image_size"] * 1.25),
            int(model_hparams["image_size"] * 1.25)
        )),
        transforms.RandomCrop(model_hparams["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Размер тренировочного датасета: {train_size}")
    print(f"Размер валидационного датасета: {val_size}")

    # Создаем DataLoader-ы
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

    # Дополнительная настройка для CUDA (если требуется)
    if train_hparams.get("accelerator") == "cuda":
        torch.set_float32_matmul_precision('medium')

    # Создаем Trainer
    trainer = pl.Trainer(
        max_epochs=train_hparams["max_epoch"],
        accelerator=train_hparams["accelerator"],
        log_every_n_steps=100
    )

    # Обучаем модель
    trainer.fit(model, train_loader, val_loader)