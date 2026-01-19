import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import wandb
from PIL import Image

def sample_synthetic_2d(model, scheduler, num_samples, device, input_size=1, mode='vit', strategy='sample'):
    """
    Samples points from the diffusion model (2D case).
    strategy: 'sample' (Probabilistic) or 'mean' (Weighted Average)
    """
    model.eval()
    
    # Start with Gaussian Noise
    latents = torch.randn(num_samples, 2, input_size, input_size, device=device)
    
    scheduler.set_timesteps(1000)
    
    for t in scheduler.timesteps:
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        with torch.inference_mode():
            output = model(latents, timesteps, None)
            
            if mode == 'vit':
                w, mu, U, lam = output
                
                if strategy == 'sample':
                    # Probabilistic Sampling: Pick one head per sample based on w
                    k_indices = torch.multinomial(w, 1) # (B, 1)
                    C, H, W = mu.shape[2:]
                    k_expanded = k_indices.view(num_samples, 1, 1, 1, 1).expand(-1, 1, C, H, W)
                    predicted_sample = torch.gather(mu, 1, k_expanded).squeeze(1)
                elif strategy == 'mean':
                    # Weighted Mean: Collapse to mode center
                    # x0 = sum(w * mu)
                    w_expanded = w.view(num_samples, -1, 1, 1, 1)
                    predicted_sample = torch.sum(w_expanded * mu, dim=1)
                else:
                    raise ValueError(f"Unknown strategy {strategy}")

            else:
                predicted_sample = output
                
        step_output = scheduler.step(predicted_sample, t, latents)
        latents = step_output.prev_sample
        
    return latents.view(num_samples, 2).cpu().numpy()

def visualize_vector_field(model, device, t=500, grid_size=20, bounds=(-2, 2)):
    """
    Visualizes the multi-branched vector field at a specific timestep.
    """
    model.eval()
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    xv, yv = np.meshgrid(x, y)
    
    # Prepare grid batch
    grid_points = torch.tensor(np.stack([xv, yv], axis=2).reshape(-1, 2), dtype=torch.float32).to(device)
    B = grid_points.shape[0]
    latents = grid_points.view(B, 2, 1, 1) 
    timesteps = torch.full((B,), t, device=device, dtype=torch.long)
    
    with torch.inference_mode():
        w, mu, U, lam = model(latents, timesteps, None)
        # w: (B, K)
        # mu: (B, K, 2, 1, 1)
    
    # Process for plotting
    w = w.cpu().numpy() # (B, K)
    mu_vectors = mu.view(B, -1, 2).cpu().numpy() # (B, K, 2)
    origin = grid_points.cpu().numpy() # (B, 2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Hydra Vector Field Split at t={t}")
    
    # For each point in grid (i), plot K arrows
    # Alpha based on w
    K = w.shape[1]
    
    # Subsample for clarity if grid is large
    step = 1
    
    for i in range(0, B, step):
        start = origin[i]
        for k in range(K):
            weight = w[i, k]
            if weight < 0.05: continue # Skip weak branches
            
            pred_x0 = mu_vectors[i, k]
            
            # Plot arrow
            # x_t is at 'start'. Predicted x_0 is 'pred_x0'.
            # Vector is (pred_x0 - start)
            
            # We scale the arrow visually to avoid clutter, though physics says it should point to x0.
            # At high noise, x0 is far, so we might want to scale it down.
            direction = pred_x0 - start
            
            ax.arrow(start[0], start[1], direction[0]*0.2, direction[1]*0.2, 
                     head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=float(weight))
                     
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def create_vector_field_gif(model, device, save_path, grid_size=20, bounds=(-2, 2), num_frames=20):
    """
    Creates a GIF of the vector field evolution over time.
    """
    frames = []
    # Reverse time: 999 -> 0
    timesteps = np.linspace(999, 0, num_frames, dtype=int)
    
    for t in timesteps:
        img = visualize_vector_field(model, device, t=t, grid_size=grid_size, bounds=bounds)
        frames.append(img)
        
    # Save frames as GIF
    # duration is per frame in ms
    frames[0].save(save_path, format='GIF', save_all=True, append_images=frames[1:], duration=250, loop=0)


def plot_synthetic_data(samples, title="Generated Data"):
    """
    Plots a scatter plot of 2D data.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot with some transparency
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=2)
    ax.set_title(title)
    ax.set_xlim(-2.5, 2.5) # Assuming normalized roughly to this range
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def log_synthetic_eval(model, scheduler, device, step, type, mode='vit', wandb_on=False):
    """
    Runs sampling and logs to WandB (if enabled).

    Args:
        mode: 'vit' or 'standard' - must match model's mode
    """
    if not wandb_on:
        return

    # 1. Probabilistic Sampling (for vit mode) or standard sampling
    samples_sample = sample_synthetic_2d(model, scheduler, num_samples=2000, device=device, mode=mode, strategy='sample')
    image_sample = plot_synthetic_data(samples_sample, title=f"Step {step} - {type} (Sampled)")

    # 2. Weighted Mean (Naive) - Should fail on multimodal (only for vit mode)
    if mode == 'vit':
        samples_mean = sample_synthetic_2d(model, scheduler, num_samples=2000, device=device, mode=mode, strategy='mean')
        image_mean = plot_synthetic_data(samples_mean, title=f"Step {step} - {type} (Mean)")
    
    # 3. Vector Field Split (Structure)
    # Only meaningful if K > 1
    # image_vf = visualize_vector_field(model, device, t=500) # Replaced by GIF
    
    # Build log dict
    log_dict = {
        f"eval/{type}_scatter_sample": wandb.Image(image_sample),
    }

    # Add mean comparison only for vit mode
    if mode == 'vit':
        log_dict[f"eval/{type}_scatter_mean"] = wandb.Image(image_mean)

        # Vector field GIF only meaningful for multi-head vit mode
        if hasattr(model, 'module'):
            vis_heads = (model.module.head.K > 1) if hasattr(model.module.head, 'K') else False
        else:
            vis_heads = (model.head.K > 1) if hasattr(model.head, 'K') else False

        if vis_heads:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
                gif_path = tmp.name

            create_vector_field_gif(model, device, save_path=gif_path, num_frames=20)
            log_dict[f"eval/{type}_diff_process"] = wandb.Video(gif_path, format="gif")

    wandb.log(log_dict, step=step)
