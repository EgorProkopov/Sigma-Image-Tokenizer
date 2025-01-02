import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory of the Tiny ImageNet dataset.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.data_dir = os.path.join(self.root_dir, "train")
        self.classes = sorted(os.listdir(self.data_dir))[1:]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.data_dir, cls_name, 'images')
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

