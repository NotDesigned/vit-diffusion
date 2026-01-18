import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class AutoencoderWrapper(nn.Module):
    """
    Wrapper for a pretrained Autoencoder (VAE) from Diffusers.
    We generally freeze this model during training.
    """
    def __init__(self, model_key="stabilityai/sd-vae-ft-mse", use_fp16=True):
        super().__init__()
        # Load VAE
        # Usually we download from HF hub. 
        # Using a standard SD VAE.
        self.vae = AutoencoderKL.from_pretrained(model_key)
        
        if use_fp16:
            self.vae.half()
        
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
