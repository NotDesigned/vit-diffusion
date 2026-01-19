import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class AutoencoderWrapper(nn.Module):
    """
    Wrapper for a pretrained Autoencoder (VAE) from Diffusers.
    We generally freeze this model during training.
    """
    def __init__(self, model_key="stabilityai/sd-vae-ft-mse", dtype=None):
        """
        Args:
            model_key: HuggingFace model key for the VAE
            dtype: torch dtype (e.g., torch.float16, torch.bfloat16).
                   None = float32, 'auto' = match accelerator mixed precision
        """
        super().__init__()
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(model_key)

        if dtype is not None:
            self.vae.to(dtype=dtype)

        self.scale_factor = 0.18215 # Magic number for SD VAE
        
    def encode(self, img):
        """
        img: (B, C, H, W) in [0, 1] or [-1, 1]?
        Diffusers VAE expects [-1, 1] if not configured otherwise, roughly.
        """
        # Assume input is [-1, 1]
        with torch.no_grad():
            dist = self.vae.encode(img).latent_dist
            latents = dist.sample()
            latents = latents * self.scale_factor
        return latents

    def decode(self, latents):
        """
        latents: (B, 4, H, W)
        """
        latents = 1 / self.scale_factor * latents
        # Ensure dtype matches VAE model
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype)
        
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return image
    
    def get_latent_dim(self):
        # Typically 4 for SD VAE
        return self.vae.config.latent_channels
