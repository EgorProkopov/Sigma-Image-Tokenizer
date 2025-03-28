import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from omegaconf import OmegaConf

from src.data.tiny_imagenet_dataset import SmallImageNetTrainDataset
from src.models.lightning_modules import ViTLightingModule


def main():
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\vit_train.yaml")

    model_hparams = config["model_hparams"]

    transform = transforms.Compose([
        transforms.Resize((model_hparams["image_size"], model_hparams["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SmallImageNetTrainDataset(
        root_dir=config["train_root_dir"],
        classes_names_path=config["classes_names_path"],
        transform=transform
    )

    val_dataset = SmallImageNetTrainDataset(
        root_dir=config["val_root_dir"],
        classes_names_path=config["classes_names_path"],
        transform=transform
    )

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    criterion = torch.nn.CrossEntropyLoss()
    model = ViTLightingModule(model_hparams, criterion, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=config["max_epoch"],
        devices=1 if torch.cuda.is_available() else None,
        accelerator=config["accelerator"],
        log_every_n_steps=100
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()