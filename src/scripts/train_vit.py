import torch
from omegaconf import OmegaConf

from src.models.lightning_modules import ViTLightingModule
from src.utils import set_seed
from src.scripts.train import train_model

def main():
    set_seed(239)
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\vit_train.yaml")
    model_hparams = config["model_hparams"]

    criterion = torch.nn.CrossEntropyLoss()
    model = ViTLightingModule(model_hparams, criterion, lr=config["train_hparams"]["lr"], log_step=config["logging"]["logging_step"])

    train_model(model, config)

if __name__ == '__main__':
    main()