import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets


class DummyDataset(Dataset):
    """Dummy dataset for testing when no real data is available."""

    def __init__(self, size=32, length=100):
        self.data = torch.randn(length, 3, 256, 256)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_image_dataloader(data_path, batch_size, image_size):
    """
    Creates a dataloader for a directory of images.

    Structure should be:
        primary_dir/
          class_A/
            img1.jpg
            ...
          class_B/
            ...

    Args:
        data_path: Path to ImageFolder dataset
        batch_size: Batch size for training
        image_size: Size to resize images to

    Returns:
        DataLoader with image dataset
    """
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found. Using Dummy Dataset.")
        dataset = DummyDataset(size=image_size, length=100)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Map to [-1, 1]
        ])
        dataset = datasets.ImageFolder(root=data_path, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
