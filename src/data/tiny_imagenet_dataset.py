import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class SmallImageNetTrainDataset(Dataset):
    def __init__(self, root_dir: str, classes_names_path: str, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory of the Small ImageNet dataset.
            classes_names_path (str): Path to the file containing the class names.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.classes_names_path = classes_names_path

        self.transform = transform
        self.to_tensor_transform = transforms.ToTensor()

        self.classes_tags = sorted(os.listdir(root_dir))
        if self.classes_tags[0] == '.DS_Store':
            self.classes_tags = self.classes_tags[1:]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes_tags)}
        self.image_paths, self.image_tags = self._get_paths()

        self.classes_names = self.get_names()


    def _get_paths(self):
        image_paths = []
        image_tags = []

        for cls_name in self.classes_tags:
            class_dir = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(class_dir):
                images = sorted(os.listdir(class_dir))
                for img_name in images:
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        image_tags.append(cls_name)

        return image_paths, image_tags

    def get_names(self):
        names = {}
        if os.path.exists(self.classes_names_path):
            with open(self.classes_names_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        class_id, class_name = parts
                        names[class_id] = class_name
        return names

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_tag = self.image_tags[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.to_tensor_transform(image)
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[image_tag]
        label_encoded = torch.zeros(len(self.classes_tags), dtype=torch.float32)
        label_encoded[label] = 1.0
        return image, label_encoded, image_tag


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SmallImageNetTrainDataset(
        root_dir=r"/Users/egorprokopov/Documents/ITMO/BachelorThesis/data/small_imagenet_object_loc/train",
        classes_names_path=r"/Users/egorprokopov/Documents/ITMO/BachelorThesis/data/small_imagenet_object_loc/classes_names.txt",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    images, labels, images_tags = next(iter(dataloader))
    # print(f"images num: {len(dataset)}")
    print(images_tags)
    print(labels)
