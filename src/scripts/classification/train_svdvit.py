import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from src.models.lightning_modules import SVDLinearViTLightingModule
from src.utils import set_seed
from src.scripts.classification.train import train_model

def main():
    set_seed(239)
    config = OmegaConf.load(r"/configs/svd_lin_vit_train.yaml")
    model_hparams = config["model_hparams"]

    criterion = torch.nn.CrossEntropyLoss()
    model = SVDLinearViTLightingModule(model_hparams, criterion, lr=config["train_hparams"]["lr"], log_step=config["logging"]["logging_step"])

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

    train_model(model, config, transform)

if __name__ == '__main__':
    main()

