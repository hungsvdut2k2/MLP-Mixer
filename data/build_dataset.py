import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from .image_dataset import ImageDataset
from PIL import Image


def calculate_mean_and_std(data_dir: str):
    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0
    for sub_dir in os.listdir(data_dir):
        sub_dir_full_path = os.path.join(data_dir, sub_dir)
        for image_path in os.listdir(sub_dir_full_path):
            image_full_path = os.path.join(sub_dir_full_path, image_path)
            image = np.array(Image.open(image_full_path))
            mean += np.mean(image, axis=(0, 1))
            std += np.std(image, axis=(0, 1))
            count += 1
    mean /= count * 255.0
    std /= count * 255.0
    return mean, std


def build_transforms(data_dir: str) -> transforms:
    means, stds = calculate_mean_and_std(data_dir)
    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[means[0], means[1], means[2]],
                std=[stds[0], stds[1], stds[2]],
            ),
        ]
    )
    return data_transforms


def build_dataloader(data_dir: str, batch_size: int) -> DataLoader:
    data_transforms = build_transforms(data_dir)
    image_dataset = ImageDataset(data_dir=data_dir, transform=data_transforms)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return image_dataloader
