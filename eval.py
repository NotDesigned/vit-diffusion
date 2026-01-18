import torch
import argparse
import os
from torchvision.utils import save_image
from tqdm import tqdm
from accelerate import Accelerator

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    print("torch-fidelity not installed.")

from src.models import DiT
from src.utils.autoencoder import AutoencoderWrapper
from src.diffusion.sampler import SwitchingSDESampler

def generate_samples(model, vae, num_samples, batch_size, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Init sampler
    sampler = SwitchingSDESampler(num_inference_steps=50)
    
    latent_size = 32 # Hardcoded based on train config usually
    channels = 4
    
    generated_count = 0
    pbar = tqdm(total=num_samples)
    
    while generated_count < num_samples:
        current_bs = min(batch_size, num_samples - generated_count)
        
        # Sample Latents
        # shape: (B, C, H, W)
        shape = (current_bs, channels, latent_size, latent_size)
        
        # Run Reverse Diffusion
        final_latents = sampler.sample(model, shape, device=device)
        
        # Decode
        images = vae.decode(final_latents)
        
        # Save
        for i in range(current_bs):
            idx = generated_count + i
            save_image(images[i], os.path.join(output_dir, f"{idx:05d}.png"))
            
        generated_count += current_bs
        pbar.update(current_bs)
        
    pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="generated_samples")
    parser.add_argument("--ref_dir", type=str, help="Path to training images for FID")
    parser.add_argument("--num_samples", type=int, default=100)
    
    # Model Configs (Must match training)
    parser.add_argument("--mode", type=str, default='vit', choices=['standard', 'vit'])
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--vit_num_heads", type=int, default=4)
    parser.add_argument("--vit_rank", type=int, default=4)
    
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    
    # Load Models
    vae = AutoencoderWrapper().to(device)
    vae.eval()
    
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
    
    # Load Weights (Simplified)
    # Checkpoint loading depends on how accelerate saved it.
    # Here assuming safe_load or torch.load of model state dict
    # accelerator.load_state(args.ckpt) # This loads everything
    # Or manual:
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, args.ckpt, 'pytorch_model.bin') # Example path
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        # Fallback to loading safetensors from accelerate directory
        model_path = os.path.join(args.ckpt, 'model.safetensors')
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(model_path))
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path} or {model_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    print("Generating Samples...")
    generate_samples(model, vae, args.num_samples, 8, device, args.output_dir)
    
    if args.ref_dir:
        print("Calculating FID...")
        metrics = calculate_metrics(
            input1=args.output_dir, 
            input2=args.ref_dir, 
            cuda=True, 
            isc=True, 
            fid=True, 
            verbose=False
        )
        print(metrics)

if __name__ == "__main__":
    main()
