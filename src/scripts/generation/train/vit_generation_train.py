import torch
from omegaconf import OmegaConf

import torchvision.transforms as transforms

from src.models.decoder_lightning_modules import ViTDecoderLightningModule
from src.utils import set_seed
from src.scripts.generation.train.train_generation import train_model


def main():
    set_seed(239)
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\vit_train.yaml")
    model_hparams = config["model_hparams"]

    criterion = torch.nn.MSELoss()
    model = ViTDecoderLightningModule(
        model_hparams, criterion,
        lr=config["train_hparams"]["lr"],
        log_step=config["logging"]["logging_step"]
    )

    image_size = 256

    transform = transforms.Compose([
        transforms.Resize((
            int(image_size * 1.25),
            int(image_size * 1.25)
        )),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_model(model, config, transform)

if __name__ == '__main__':
    main()