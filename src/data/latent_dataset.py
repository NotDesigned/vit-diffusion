import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
import os


class LatentDataset(Dataset):
    """
    A dataset that pre-encodes images to latent space using a VAE.
    Caches latents to disk for faster subsequent training runs.

    Usage:
        dataset = LatentDataset(
            data_path='./data',
            vae=vae,
            image_size=256,
            cache_dir='./latent_cache',
            device='cuda'
        )
    """

    def __init__(
        self,
        data_path: str,
        vae,
        image_size: int = 256,
        cache_dir: str = None,
        device: str = 'cuda',
        force_recompute: bool = False
    ):
        """
        Args:
            data_path: Path to ImageFolder dataset
            vae: VAE model for encoding (should be on device)
            image_size: Size to resize images to before encoding
            cache_dir: Directory to cache encoded latents (if None, uses data_path/.latent_cache)
            device: Device to use for encoding
            force_recompute: If True, recompute latents even if cache exists
        """
        self.data_path = data_path
        self.device = device

        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(data_path, '.latent_cache')
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f'latents_{image_size}.pt')

        # Check if cache exists and is valid
        if os.path.exists(self.cache_file) and not force_recompute:
            print(f"Loading cached latents from {self.cache_file}")
            cache_data = torch.load(self.cache_file, map_location='cpu')
            self.latents = cache_data['latents']
            self.labels = cache_data.get('labels', None)
            print(f"Loaded {len(self.latents)} cached latents")
        else:
            # Need to encode the dataset
            print(f"Pre-encoding dataset to latent space...")
            self.latents, self.labels = self._encode_dataset(
                data_path, vae, image_size, device
            )

            # Save cache
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving latent cache to {self.cache_file}")
            torch.save({
                'latents': self.latents,
                'labels': self.labels,
                'image_size': image_size
            }, self.cache_file)

    def _encode_dataset(self, data_path, vae, image_size, device):
        """Encode entire dataset to latent space."""
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image_dataset = datasets.ImageFolder(root=data_path, transform=transform)

        # Use a larger batch size for encoding (more efficient)
        encode_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_latents = []
        all_labels = []

        vae.eval()
        # Use VAE's actual dtype for consistency
        vae_dtype = next(vae.parameters()).dtype
        with torch.no_grad():
            for images, labels in tqdm(encode_loader, desc="Encoding images"):
                images = images.to(device, dtype=vae_dtype)
                latents = vae.encode(images)
                # Store as float32 on CPU to save GPU memory
                all_latents.append(latents.cpu().float())
                all_labels.append(labels)

        return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        if self.labels is not None:
            return latent, self.labels[idx]
        return latent


def get_latent_dataloader(
    data_path: str,
    vae,
    batch_size: int,
    image_size: int = 256,
    cache_dir: str = None,
    device: str = 'cuda',
    force_recompute: bool = False,
    **dataloader_kwargs
):
    """
    Convenience function to create a DataLoader with pre-encoded latents.

    Args:
        data_path: Path to ImageFolder dataset
        vae: VAE model for encoding
        batch_size: Batch size for training
        image_size: Size to resize images to before encoding
        cache_dir: Directory to cache encoded latents
        device: Device to use for encoding
        force_recompute: If True, recompute latents even if cache exists
        **dataloader_kwargs: Additional kwargs for DataLoader

    Returns:
        DataLoader with pre-encoded latents
    """
    dataset = LatentDataset(
        data_path=data_path,
        vae=vae,
        image_size=image_size,
        cache_dir=cache_dir,
        device=device,
        force_recompute=force_recompute
    )

    default_kwargs = {
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True
    }
    default_kwargs.update(dataloader_kwargs)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        **default_kwargs
    )
