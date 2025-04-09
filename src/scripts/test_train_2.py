import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from omegaconf import OmegaConf

from src.scripts.test_vit import VisionTransformer  # Импортируй свой ViT сюда
from src.data.tiny_imagenet_dataset import SmallImageNetTrainDataset


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        images = batch["image"]
        labels = batch["label_encoded"]

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"]
            labels = batch["label_encoded"]

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Гиперпараметры
    epochs = 10
    batch_size = 12
    learning_rate = 3e-4
    num_classes = 1000  # CIFAR-10

    # Аугментации и загрузка данных
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
    ])

    config = OmegaConf.load(r"F:\research\Sigma-Image-Tokenizer\configs\vit_train.yaml")

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


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Модель
    model = VisionTransformer(num_classes=num_classes)
    model.to(device)

    # Оптимизатор и лосс
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Сохраняем модель
    print("Training complete")


if __name__ == "__main__":
    main()
