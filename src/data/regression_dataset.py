import os
from torch.utils.data import Dataset
import torchvision
from PIL import Image


class UTKFaceDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.filenames = os.listdir(self.dataset_dir)
        self.totensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(os.path.join(self.dataset_dir, img_name))
        split_var = img_name.split('_')
        age = float(int(split_var[0]))

        # image = self.totensor(image)

        if self.transform:
            image = self.transform(image)

        return image, age


if __name__ == "__main__":
    data_dir = r"F:\research\data\utk_faces\train"
    dataset = UTKFaceDataset(data_dir)
    print(dataset[100])

