import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from omegaconf import OmegaConf

from src.data.tiny_imagenet_dataset import SmallImageNetTrainDataset
from src.models.lightning_modules import SVDViTLightingModule


def main():
    config = OmegaConf.load(r"F:\\research\\Sigma-Image-Tokenizer\\configs\\svdvit_train.yaml")

    model_hparams = config["model_hparams"]

    transform = transforms.Compose([
        transforms.Resize((model_hparams["image_size"], model_hparams["image_size"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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

    train_size = len(train_dataset)
    val_size = len(val_dataset)

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
    model = SVDViTLightingModule(model_hparams, criterion, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=config["train_hparams"]["max_epoch"],
        devices=1 if torch.cuda.is_available() else None,
        accelerator=config["train_hparams"]["accelerator"],
        log_every_n_steps=100
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()