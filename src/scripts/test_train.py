import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.scripts.test_vit import VisionTransformer
from src.data.tiny_imagenet_dataset import SmallImageNetTrainDataset


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters and configuration
    config = {
        "image_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "embedding_dim": 768,
        "qkv_dim": 64,
        "mlp_hidden_size": 1024,
        "n_layers": 12,
        "n_heads": 12,
        "n_classes": 1000,  # Update if your dataset has a different number of classes
        "lr": 1e-4,
        "batch_size": 32,
        "num_epochs": 20
    }

    # Paths to your training and validation data and the class names file
    train_root= r"F:\\research\data\\small_imagenet_object_loc\\train"
    val_root= r"F:\\research\data\\small_imagenet_object_loc\\val"
    classes_names_path= r"F:\\research\\data\\small_imagenet_object_loc\\classes_names.txt"

    # Define transforms (make sure PIL-based transforms come before ToTensor)
    transform = transforms.Compose([
        transforms.Resize((int(config["image_size"] * 1.2), int(config["image_size"] * 1.2))),
        transforms.RandomCrop(config["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create training and validation datasets
    train_dataset = SmallImageNetTrainDataset(
        root_dir=train_root,
        classes_names_path=classes_names_path,
        transform=transform
    )
    val_dataset = SmallImageNetTrainDataset(
        root_dir=val_root,
        classes_names_path=classes_names_path,
        transform=transform
    )

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # Instantiate the Vision Transformer model
    model = VisionTransformer(
        # image_size=config["image_size"],
        # patch_size=config["patch_size"],
        # in_channels=config["in_channels"],
        # embedding_dim=config["embedding_dim"],
        # qkv_dim=config["qkv_dim"],
        # mlp_hidden_size=config["mlp_hidden_size"],
        # n_layers=config["n_layers"],
        # n_heads=config["n_heads"],
        # n_classes=config["n_classes"]
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_val_acc = 0.0

    # Training and Validation Loop
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0

        for idx, batch in enumerate(train_loader):
            # Get images and labels from the batch dictionary
            images = batch["image"].to(device)             # Tensor shape: [batch_size, C, H, W]
            labels = batch["label_encoded"].to(device)       # Tensor shape: [batch_size]

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(images)  # Expected output shape: [batch_size, n_classes]
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            print(f"Итерация: {idx}, лосс: {loss.item()}")

        epoch_loss = running_loss / len(train_dataset)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label_encoded"].to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], "
              f"Train Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vit_model.pth")

    print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))


if __name__ == "__main__":
    main()
