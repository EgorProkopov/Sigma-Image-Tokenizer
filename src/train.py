from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, classification_report

from src.data.tiny_imagenet_dataset import SmallImageNetTrainDataset
from src.models.transformer import VisionTransformer
from src.utils import plot_metrics


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, device, epochs):
    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels, tags in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)

            predicted_encoded = torch.zeros_like(labels)
            predicted_encoded[predicted] = 1.0
            correct_train += (predicted_encoded == labels).sum().item()

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(valid_dataloader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


def test(model, test_dataloader, device):
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    test_accuracy = accuracy_score(true_labels, pred_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(classification_report(true_labels, pred_labels))


if __name__ == '__main__':
    train_dataset_dir = r"/Users/egorprokopov/Documents/ITMO/BachelorThesis/data/small_imagenet_object_loc/train"
    val_dataset_dir = r"/Users/egorprokopov/Documents/ITMO/BachelorThesis/data/small_imagenet_object_loc/val"
    classes_names_path = r"/Users/egorprokopov/Documents/ITMO/BachelorThesis/data/small_imagenet_object_loc/classes_names.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    train_dataset = SmallImageNetTrainDataset(
        root_dir=train_dataset_dir,
        classes_names_path=classes_names_path,
        transform=transform
    )
    val_dataset = SmallImageNetTrainDataset(
        root_dir=val_dataset_dir,
        classes_names_path=classes_names_path,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("mps")
    model = VisionTransformer(
        image_size = 224,
        patch_size = 16,
        in_channels = 3,
        embed_dim = 768,
        qkv_dim = 64,
        mlp_hidden_size = 3072,
        n_layers= 12,
        n_heads = 12,
        n_classes = 1000
    )
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train_loss, val_loss, train_acc, val_acc = train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=2
    )

    plot_metrics(train_loss, val_loss, train_acc, val_acc)
