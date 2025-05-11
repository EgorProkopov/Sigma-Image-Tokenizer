import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from src.models.lightning_modules import WaveletViTLightningModule
from src.utils import set_seed
from src.scripts.train import train_model

def main():
    set_seed(239)
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\wavelet_vit_train.yaml")
    model_hparams = config["model_hparams"]

    criterion = torch.nn.CrossEntropyLoss()
    model = WaveletViTLightningModule(model_hparams, criterion, lr=config["train_hparams"]["lr"], log_step=config["logging"]["logging_step"])

    image_size = 32

    transform = transforms.Compose([
        transforms.Resize((
            int(image_size * 1.25),
            int(image_size * 1.25)
        )),
        transforms.RandomCrop(image_size),
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

