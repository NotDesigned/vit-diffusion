import torch
import torch.nn.functional as F
import numpy as np
import wandb
from torchvision.utils import make_grid
from PIL import Image

def sample_images(model, scheduler, vae, num_samples, device, input_size, in_channels, mode='vit', strategy='sample'):
    """
    Samples latent images and decodes them.
    strategy: 'sample' (Pick one head) or 'mean' (Weighted average)
    """
    model.eval()
    
    # Start with Gaussian Noise
    latents = torch.randn(num_samples, in_channels, input_size, input_size, device=device)
    
    scheduler.set_timesteps(1000)
    
    for t in scheduler.timesteps:
        # Expand scalar time to batch
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            output = model(latents, timesteps, None)
            
            if mode == 'vit':
                w, mu, U, lam = output
                
                if strategy == 'sample':
                    # Probabilistic Sampling
                    # w: (B, K)
                    k_indices = torch.multinomial(w, 1) # (B, 1)
                    
                    # Gather chosen mu
                    # mu is (B, K, C, H, W)
                    B_sz, K_sz, C, H, W = mu.shape
                    
                    k_expanded = k_indices.view(B_sz, 1, 1, 1, 1).expand(-1, 1, C, H, W)
                    predicted_noise = torch.gather(mu, 1, k_expanded).squeeze(1)
                    
                elif strategy == 'mean':
                    # Weighted Mean
                    w_expanded = w.view(num_samples, -1, 1, 1, 1) # (B, K, 1, 1, 1)
                    predicted_noise = torch.sum(w_expanded * mu, dim=1)
                
            else:
                # Standard DDPM
                predicted_noise = output
                
        # Step
        step_output = scheduler.step(predicted_noise, t, latents)
        latents = step_output.prev_sample

    # Decode latents
    # VAE requires float32 or compatible type. ensure latents match vae dtype if needed.
    # The AutoencoderWrapper handles casting usually? Let's check train.py: 
    # latents = vae.encode(images.to(device, dtype=torch.float16))
    # So VAE is likely fp16.
    
    # Ensure latents are same dtype as vae
    dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=dtype)
    
    # decode expects (B, 4, H, W)
    images = vae.decode(latents)
    
    # Images are in [-1, 1], convert to [0, 1] for display
    images = (images / 2 + 0.5).clamp(0, 1)
    
    return images

def log_image_eval(model, scheduler, vae, device, step, args, wandb_on=False):
    if not wandb_on:
        return
        
    # Generate 8 samples (grid of 4x2)
    num_samples = 8
    
    log_dict = {}
    
    if args.mode == 'vit':
        # 1. Sampled Strategy (Correct)
        imgs_sample = sample_images(model, scheduler, vae, num_samples, device, 
                                    input_size=args.input_size, in_channels=4, 
                                    mode='vit', strategy='sample')
        grid_sample = make_grid(imgs_sample, nrow=4)
        ndarr_sample = grid_sample.permute(1, 2, 0).cpu().numpy()
        img_sample_pil = Image.fromarray((ndarr_sample * 255).astype(np.uint8))
        
        log_dict[f"eval/generated_images_sample"] = wandb.Image(img_sample_pil, caption=f"Step {step} - Sample Strategy")
        
        # 2. Mean Strategy (Comparison)
        imgs_mean = sample_images(model, scheduler, vae, num_samples, device, 
                                  input_size=args.input_size, in_channels=4, 
                                  mode='vit', strategy='mean')
        grid_mean = make_grid(imgs_mean, nrow=4)
        ndarr_mean = grid_mean.permute(1, 2, 0).cpu().numpy()
        img_mean_pil = Image.fromarray((ndarr_mean * 255).astype(np.uint8))
        
        log_dict[f"eval/generated_images_mean"] = wandb.Image(img_mean_pil, caption=f"Step {step} - Mean Strategy")
        
    else:
        # Standard Mode
        imgs = sample_images(model, scheduler, vae, num_samples, device, 
                             input_size=args.input_size, in_channels=4, 
                             mode='standard')
        grid = make_grid(imgs, nrow=4)
        ndarr = grid.permute(1, 2, 0).cpu().numpy()
        img_pil = Image.fromarray((ndarr * 255).astype(np.uint8))
        log_dict[f"eval/generated_images"] = wandb.Image(img_pil, caption=f"Step {step}")

    wandb.log(log_dict, step=step)
