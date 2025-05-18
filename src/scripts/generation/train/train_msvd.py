import torch
from omegaconf import OmegaConf

import torchvision.transforms as transforms

from src.models.autoencoders import MSVDAutoencoderLightningModule
from src.utils import set_seed
from src.scripts.generation.train.train import train_model


def main():
    set_seed(239)
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\msvd_autoencoder.yaml")
    model_hparams = config["model_hparams"]

    criterion = torch.nn.MSELoss()
    model = MSVDAutoencoderLightningModule(model_hparams, criterion, lr=config["train_hparams"]["lr"], log_step=config["logging"]["logging_step"])

    image_size = 128

    transform = transforms.Compose([
        transforms.Resize((
            int(image_size),
            int(image_size)
        )),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
    ])

    train_model(model, config, transform)

if __name__ == '__main__':
    main()
