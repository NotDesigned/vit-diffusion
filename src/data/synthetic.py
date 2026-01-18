import torch
from torch.utils.data import Dataset
import numpy as np
import math

class SyntheticDataset(Dataset):
    """
    Synthetic datasets for testing topological diffusion.
    Returns (B, 2) data points reshaped to (B, C, 1, 1) for compatibility with DiT.
    """
    def __init__(self, size=50000, type='swiss_roll'):
        self.size = size
        self.type = type
        self.data = self._generate_data()

    def _generate_data(self):
        if self.type == 'swiss_roll':
            # 2D Swiss Roll
            n = self.size
            theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
            r_noise = 0.25 # Noise level
            
            x = theta * np.cos(theta)
            y = theta * np.sin(theta)
            data = np.stack([x, y], axis=1)
            
            # Normalize to roughly [-2, 2] then scale to [-1, 1]?
            data = (data - data.mean(0)) / data.std(0)
            data = data * 0.5 # Scale to fit comfortably in [-1, 1]
            
            # Add intrinsic manifold noise
            data += np.random.randn(n, 2) * 0.05
            
        elif self.type == '8gaussians':
            n = self.size
            scale = 2.
            centers = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(scale * x, scale * y) for x, y in centers]
            
            data = []
            for i in range(n):
                point = np.random.randn(2) * 0.05
                center = centers[np.random.choice(len(centers))]
                point[0] += center[0]
                point[1] += center[1]
                data.append(point)
            data = np.array(data)
            data /= 1.414 # Scale roughly to [-2, 2] range
            data *= 0.5 
            
        elif self.type == 'checkerboard':
            n = self.size
            x1 = np.random.rand(n) * 4 - 2
            x2_ = np.random.rand(n) - np.random.randint(0, 2, n) * 2
            x2 = x2_ + (np.floor(x1) % 2)
            data = np.stack([x1, x2], axis=1) * 2
            data = (data - data.mean(0)) / data.std(0) * 0.5
            
        elif self.type == 'olympic':
            # 5 Interlocking circles
            # Just rough centers
            centers = [(-2, 0), (-1, -1), (0, 0), (1, -1), (2, 0)]
            n = self.size
            data = []
            for i in range(n):
                c_idx = np.random.choice(len(centers))
                cx, cy = centers[c_idx]
                angle = np.random.rand() * 2 * np.pi
                r = 0.8 + np.random.randn() * 0.05
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                data.append([x, y])
            data = np.array(data)
            data = (data - data.mean(0)) / data.std(0) * 0.6
            
        else:
            # Simple Gaussian
            data = np.random.randn(self.size, 2) * 0.5

        return torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return as (C, H, W) where H=1, W=1
        return self.data[idx].view(2, 1, 1)
