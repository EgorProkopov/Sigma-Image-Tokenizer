from omegaconf import OmegaConf
import lightning.pytorch as pl
import torch
import torchvision.transforms as transforms

from src.models.lightning_modules import MSVDSigmoidGatingViTLightningModule
from src.models.losses import GatedLoss
from src.utils import set_seed
from src.scripts.train import test_model

def main():
    # reproducibility
    set_seed(239)

    # 1) загрузка конфига
    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\msvdvit_sigmoid_gating.yaml")

    # 2) создаём модель «голой» (её __init__ задаёт структуру)
    model = MSVDSigmoidGatingViTLightningModule(
        config.model_hparams,
        torch.nn.CrossEntropyLoss(),
        lr=config.train_hparams.lr,
        log_step=config.logging.logging_step
    )

    # 3) читаем чекпоинт
    ckpt_path = r"F:\research\Sigma-Image-Tokenizer\src\scripts\lightning_logs\msvdvit_sigmoid_gating_00005\checkpoints\epoch=19-step=23400.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    orig_sd = ckpt.get("state_dict", ckpt)

    # 4) приводим ключи в соответствие self.model.<...>
    new_sd = {}
    for k, v in orig_sd.items():
        if k.startswith("model."):
            new_sd[k] = v
        else:
            new_sd["model." + k] = v

    # 5) загружаем веса (strict=False, чтобы увидеть реальные несовпадения)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    # 6) переводим на устройство и в eval
    device = config.train_hparams.accelerator
    model = model.to(device).eval()

    # 7) готовим трансформации
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 8) запускаем тестирование
    test_model(model, config, transform)

if __name__ == "__main__":
    main()
