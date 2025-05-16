from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms

from src.models.lightning_modules import MSVDSigmoidGatingViTLightningModule
from src.utils import set_seed
from src.scripts.classification.train import test_model

def main():
    set_seed(239)

    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\msvdvit_sigmoid_gating.yaml")

    model = MSVDSigmoidGatingViTLightningModule(
        config.model_hparams,
        torch.nn.CrossEntropyLoss(),
        lr=config.train_hparams.lr,
        log_step=config.logging.logging_step
    )

    ckpt_path = r"F:\research\Sigma-Image-Tokenizer\src\scripts\classification\lightning_logs\msvdvit_sigmoid_gating_00005\checkpoints\epoch=19-step=23400.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    orig_sd = ckpt.get("state_dict", ckpt)

    new_sd = {}
    for k, v in orig_sd.items():
        if k.startswith("model."):
            new_sd[k] = v
        else:
            new_sd["model." + k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    device = config.train_hparams.accelerator
    model = model.to(device).eval()

    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_model(model, config, transform)

if __name__ == "__main__":
    main()
