import torch
import argparse
import os
from torchvision.utils import save_image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from torch_fidelity import calculate_metrics
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: torch-fidelity not installed. FID calculation disabled.")

from src.models import DiT
from src.utils.autoencoder import AutoencoderWrapper


def generate_samples(model, scheduler, vae, num_samples, batch_size, device, output_dir, 
                     mode='vit', strategy='sample', input_size=32, num_inference_steps=50):
    """
    Generate samples using DDPMScheduler (consistent with training).
    
    Args:
        model: Trained diffusion model
        scheduler: DDPMScheduler instance
        vae: VAE for decoding latents
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: torch device
        output_dir: Directory to save generated images
        mode: 'vit' or 'standard'
        strategy: 'sample' (probabilistic) or 'mean' (weighted average) for vit mode
        input_size: Latent spatial size (default 32 for 256px images)
        num_inference_steps: Number of denoising steps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    in_channels = 4
    generated_count = 0
    pbar = tqdm(total=num_samples, desc="Generating samples")
    
    scheduler.set_timesteps(num_inference_steps)
    
    while generated_count < num_samples:
        current_bs = min(batch_size, num_samples - generated_count)
        
        # Start from Gaussian noise
        latents = torch.randn(current_bs, in_channels, input_size, input_size, device=device)
        
        # Reverse diffusion
        with torch.no_grad():
            for t in scheduler.timesteps:
                timesteps = torch.full((current_bs,), t, device=device, dtype=torch.long)
                output = model(latents, timesteps, None)
                
                if mode == 'vit':
                    w, mu, U = output
                    
                    if strategy == 'sample':
                        # Probabilistic: sample one head per batch item
                        k_indices = torch.multinomial(w, 1)  # (B, 1)
                        B_sz, K_sz, C, H, W = mu.shape
                        k_expanded = k_indices.view(B_sz, 1, 1, 1, 1).expand(-1, 1, C, H, W)
                        predicted_sample = torch.gather(mu, 1, k_expanded).squeeze(1)
                    elif strategy == 'mean':
                        # Weighted mean across heads
                        w_expanded = w.view(current_bs, -1, 1, 1, 1)
                        predicted_sample = torch.sum(w_expanded * mu, dim=1)
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                else:
                    # Standard mode: output is already x0 prediction
                    predicted_sample = output
                
                step_output = scheduler.step(predicted_sample, t, latents)
                latents = step_output.prev_sample
        
        # Decode latents to images
        images = vae.decode(latents)
        
        # Convert to [0, 1] for saving (VAE outputs [-1, 1])
        images = (images / 2 + 0.5).clamp(0, 1)
        
        # Save images
        for i in range(current_bs):
            idx = generated_count + i
            save_image(images[i], os.path.join(output_dir, f"{idx:05d}.png"))
            
        generated_count += current_bs
        pbar.update(current_bs)
        
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Generate samples and compute FID")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Output directory for samples")
    parser.add_argument("--ref_dir", type=str, help="Path to real images for FID calculation")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--strategy", type=str, default='sample', choices=['sample', 'mean'], 
                        help="Sampling strategy for VIT mode")
    
    # Model Configs (must match training)
    parser.add_argument("--mode", type=str, default='vit', choices=['standard', 'vit'])
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--vit_num_heads", type=int, default=4)
    parser.add_argument("--vit_rank", type=int, default=16)
    
    args = parser.parse_args()
    
    # Setup accelerator (handles mixed precision)
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    
    # Determine VAE dtype from accelerator
    mixed_precision_to_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": None,
    }
    vae_dtype = mixed_precision_to_dtype.get(accelerator.mixed_precision)
    
    # Load VAE
    vae = AutoencoderWrapper(dtype=vae_dtype).to(device)
    vae.eval()
    
    # Build model
    vit_conf = {'num_heads': args.vit_num_heads, 'rank': args.vit_rank} if args.mode == 'vit' else None
    
    model = DiT(
        input_size=args.input_size, 
        patch_size=args.patch_size, 
        in_channels=4,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mode=args.mode,
        vit_conf=vit_conf
    )
    
    # Load checkpoint
    # Try multiple possible checkpoint formats
    ckpt_paths = [
        os.path.join(args.ckpt, 'model.safetensors'),
        os.path.join(args.ckpt, 'pytorch_model.bin'),
    ]
    
    loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            if ckpt_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict)
            loaded = True
            break
    
    if not loaded:
        print(f"Warning: No checkpoint found in {args.ckpt}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Setup scheduler (must match training)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        beta_start=0.0001, 
        beta_end=0.02, 
        beta_schedule="linear", 
        prediction_type="sample",
        clip_sample=False 
    )
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    generate_samples(
        model=model,
        scheduler=scheduler,
        vae=vae,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
        mode=args.mode,
        strategy=args.strategy,
        input_size=args.input_size,
        num_inference_steps=args.num_inference_steps
    )
    print(f"Samples saved to {args.output_dir}")
    
    # Calculate FID if reference directory provided
    if args.ref_dir and FID_AVAILABLE:
        print("Calculating FID...")
        try:
            metrics = calculate_metrics(
                input1=args.output_dir, 
                input2=args.ref_dir, 
                cuda=torch.cuda.is_available(), 
                fid=True, 
                verbose=False
            )
            print(f"FID: {metrics.get('frechet_inception_distance', 'N/A')}")
        except Exception as e:
            print(f"FID calculation failed: {e}")
    elif args.ref_dir and not FID_AVAILABLE:
        print("Skipping FID calculation (torch-fidelity not installed)")


if __name__ == "__main__":
    main()
