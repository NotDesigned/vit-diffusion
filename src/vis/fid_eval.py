import torch
import numpy as np
import os
from torchvision.utils import save_image
from tqdm import tqdm

try:
    from torch_fidelity import calculate_metrics
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: torch-fidelity not installed. FID calculation disabled.")


def compute_fid(model, scheduler, vae, num_samples, device, real_dataset_path, temp_dir='./temp_samples', mode='vit', strategy='sample', num_inference_steps=50, accelerator=None):
    """
    Compute FID between generated samples and real dataset.
    
    Args:
        model: Trained diffusion model
        scheduler: Noise scheduler
        vae: VAE for decoding latents
        num_samples: Number of samples to generate for FID calculation
        device: torch device
        real_dataset_path: Path to real images (used for FID reference)
        temp_dir: Temporary directory to save generated images
        mode: 'vit' or 'standard'
        strategy: Sampling strategy for VIT mode
        num_inference_steps: Number of denoising steps (default: 50, much faster than 1000)
        accelerator: Optional Accelerator instance for distributed generation
    
    Returns:
        dict: FID metrics
    """
    if not FID_AVAILABLE:
        return {"fid": None, "error": "torch-fidelity not installed"}
    
    # Determine distributed settings
    if accelerator is not None:
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index
        is_main_process = accelerator.is_main_process
    else:
        num_processes = 1
        process_index = 0
        is_main_process = True
    
    # Split generation across processes for faster FID computation
    samples_per_process = num_samples // num_processes
    start_idx = process_index * samples_per_process
    
    # Main process generates remaining samples if num_samples not divisible
    if is_main_process:
        samples_per_process += num_samples % num_processes
    
    # Each process saves to its own subdirectory to avoid conflicts
    process_temp_dir = os.path.join(temp_dir, f"rank_{process_index}")
    
    # Generate samples
    print(f"[Rank {process_index}] Generating {samples_per_process} samples for FID calculation...")
    os.makedirs(process_temp_dir, exist_ok=True)
    
    model.eval()
    latent_size = 32  # Assuming standard VAE latent size
    in_channels = 4
    batch_size = 32  # Small batch to avoid OOM
    
    generated_count = 0
    pbar = tqdm(total=samples_per_process, desc=f"Generating FID samples (Rank {process_index})", disable=not is_main_process)
    
    while generated_count < samples_per_process:
        current_bs = min(batch_size, samples_per_process - generated_count)
        latents = torch.randn(current_bs, in_channels, latent_size, latent_size, device=device)
        
        # Use fewer steps for faster FID computation (50 is standard, 1000 is overkill)
        scheduler.set_timesteps(num_inference_steps)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                timesteps = torch.full((current_bs,), t, device=device, dtype=torch.long)
                output = model(latents, timesteps, None)
                
                if mode == 'vit':
                    w, mu, U = output
                    
                    if strategy == 'sample':
                        k_indices = torch.multinomial(w, 1)
                        B_sz, K_sz, C, H, W = mu.shape
                        k_expanded = k_indices.view(B_sz, 1, 1, 1, 1).expand(-1, 1, C, H, W)
                        predicted_sample = torch.gather(mu, 1, k_expanded).squeeze(1)
                    elif strategy == 'mean':
                        w_expanded = w.view(current_bs, -1, 1, 1, 1)
                        predicted_sample = torch.sum(w_expanded * mu, dim=1)
                else:
                    predicted_sample = output
                
                step_output = scheduler.step(predicted_sample, t, latents)
                latents = step_output.prev_sample
        
        # Decode (dtype conversion handled in vae.decode())
        images = vae.decode(latents)
        
        # Ensure images are in CPU and correct format for saving
        # VAE output is typically in [-1, 1] range
        images = images.cpu().float()
        
        # Convert to [0, 1] for saving
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        # Save images
        for i in range(current_bs):
            idx = start_idx + generated_count + i
            save_image(images[i], os.path.join(process_temp_dir, f"{idx:05d}.png"))
        
        generated_count += current_bs
        pbar.update(current_bs)
    
    pbar.close()
    
    # Wait for all processes to finish generation
    if accelerator is not None:
        accelerator.wait_for_everyone()
    
    # Only main process calculates FID using all generated samples
    if not is_main_process:
        return {"fid": None, "skipped": "non-main process"}
    
    # Calculate FID
    print("Calculating FID...")
    try:
        metrics = calculate_metrics(
            input1=temp_dir,
            input2=real_dataset_path,
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False,
            samples_find_deep=True,  # Allow searching in subdirectories (rank_0, rank_1, etc.)
            samples_find_ext='png,jpg,jpeg'  # Specify image extensions
        )
        
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return metrics
    
    except Exception as e:
        print(f"FID calculation failed: {e}")
        return {"fid": None, "error": str(e)}


def log_fid_to_wandb(fid_metrics, step, wandb_on=True):
    """Log FID metrics to WandB"""
    if wandb_on and fid_metrics.get('frechet_inception_distance') is not None:
        import wandb
        wandb.log({
            "eval/fid": fid_metrics['frechet_inception_distance'],
        }, step=step)
        print(f"FID: {fid_metrics['frechet_inception_distance']:.2f}")
