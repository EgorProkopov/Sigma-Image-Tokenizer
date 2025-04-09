import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

from omegaconf import OmegaConf

from src.data.tiny_imagenet_dataset import SmallImageNetTrainDataset
from src.models.lightning_modules import ViTLightingModule
from src.utils import set_seed


def main():
    set_seed(239)
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\vit_train.yaml")

    model_hparams = config["model_hparams"]

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

    train_dataset = SmallImageNetTrainDataset(
        root_dir=config["dataset"]["train_root_dir"],
        classes_names_path=config["dataset"]["classes_names_path"],
        transform=transform
    )

    val_dataset = SmallImageNetTrainDataset(
        root_dir=config["dataset"]["val_root_dir"],
        classes_names_path=config["dataset"]["classes_names_path"],
        transform=transform
    )

    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    train_size = len(train_dataset)
    val_size = len(val_dataset)

    print(f"Размер тренировочного датасета: {train_size}")
    print(f"Размер валидационного датасета: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_hparams"]["train_batch_size"],
        shuffle=True,
        num_workers=config["train_hparams"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train_hparams"]["val_batch_size"],
        shuffle=False,
        num_workers=config["train_hparams"]["num_workers"]
    )

    criterion = torch.nn.CrossEntropyLoss()
    model = ViTLightingModule(model_hparams, criterion, lr=config["train_hparams"]["lr"], log_step=1000)

    # if config["train_hparams"]["accelerator"] == "cuda":
    #     torch.set_float32_matmul_precision('medium')


    trainer = pl.Trainer(
        max_epochs=config["train_hparams"]["max_epoch"],
        devices=1 if torch.cuda.is_available() else None,
        accelerator=config["train_hparams"]["accelerator"],
        log_every_n_steps=1000
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()