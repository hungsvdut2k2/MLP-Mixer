import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import convert_category_to_label
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.categories = os.listdir(data_dir)
        self.labels = convert_category_to_label(self.categories)
        self.path_with_label = []
        for i in range(len(self.categories)):
            sub_dir_path = os.path.join(self.data_dir, self.categories[i])
            for image_path in os.listdir(sub_dir_path):
                self.path_with_label.append(
                    (os.path.join(sub_dir_path, image_path), self.labels[i])
                )

    def __len__(self):
        return len(self.path_with_label)

    def __getitem__(self, index):
        image = Image.open(self.path_with_label[index][0])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.path_with_label[index][1])
        return image, label
