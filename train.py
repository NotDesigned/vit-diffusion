import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from accelerate import Accelerator
from accelerate.logging import get_logger
import os
import argparse
from contextlib import contextmanager
from tqdm import tqdm
from torchvision import transforms, datasets
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import wandb


@contextmanager
def ema_scope(ema_model, use_ema):
    """Context manager for temporarily switching to EMA weights during evaluation."""
    if use_ema and ema_model is not None:
        ema_model.store()
        ema_model.copy_to()
    try:
        yield
    finally:
        if use_ema and ema_model is not None:
            ema_model.restore()

from src.models import DiT
from src.diffusion.loss import TrinityLoss
from src.utils.autoencoder import AutoencoderWrapper
from src.vis.synthetic_plot import log_synthetic_eval
from src.vis.image_eval import log_image_eval
from src.vis.fid_eval import compute_fid, log_fid_to_wandb
from src.data.latent_dataset import get_latent_dataloader

def get_dataloader(data_path, batch_size, image_size):
    """
    Creates a dataloader for a directory of images.
    Structure should be:
    primary_dir/
      class_A/
        img1.jpg
        ...
      class_B/
        ...
    """
    # If path doesn't exist or is empty, fallback to Dummy for testing
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found. Using Dummy Dataset.")
        dataset = DummyDataset(size=image_size, length=100)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Map to [-1, 1]
        ])
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

class DummyDataset(Dataset):
    def __init__(self, size=32, length=100):
        self.data = torch.randn(length, 3, 256, 256) # Returns Dummy Images
        
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config file") # Removed config support

    
    # Training Configs
    parser.add_argument("--mode", type=str, default='vit', choices=['standard', 'vit'], help="Training mode")
    parser.add_argument("--dataset_type", type=str, default='image_folder', choices=['image_folder', 'synthetic'], help="Dataset type")
    parser.add_argument("--synthetic_type", type=str, default='swiss_roll', help="Type of synthetic dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    # Model Configs
    parser.add_argument("--hidden_size", type=int, default=384, help="Transformer hidden size")
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=6, help="Transformer attention heads")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size")
    parser.add_argument("--input_size", type=int, default=32, help="Latent input size (e.g. 32 for 256px)")
    
    # VIT Specific
    parser.add_argument("--vit_num_heads", type=int, default=4, help="Number of Hydra heads")
    parser.add_argument("--vit_rank", type=int, default=16, help="Rank of local subspace")
    parser.add_argument("--sigma_gmm", type=float, default=1.0, help="Sigma for GMM loss, not useful if using annealing")

    # Loss Configs
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Steps to warmup aux losses")
    parser.add_argument("--lambda_gmm", type=float, default=10.0, help="Weight for GMM loss")
    parser.add_argument("--lambda_align", type=float, default=5.0, help="Max weight for alignment loss")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Max weight for reg loss (dim + ortho)")
    parser.add_argument("--lambda_div", type=float, default=1.0, help="Weight for diversity loss")
    parser.add_argument("--lambda_repul", type=float, default=1.0, help="Weight for repulsion loss")
    parser.add_argument("--only_winner_align", action='store_true', help="Only align winner head")
    
    parser.add_argument("--temp_anneal_steps", type=int, default=20000, help="Steps to heal Sigma GMM")
    parser.add_argument("--sigma_start", type=float, default=2.0, help="Start Temp")
    parser.add_argument("--sigma_end", type=float, default=0.1, help="End Temp")
    
    parser.add_argument("--no_schedule", action='store_true', help="Disable loss schedule")
    
    # Logging & WandB Configs
    parser.add_argument("--use_wandb", action='store_true', help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="vit-diffusion", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB specific run name")
    parser.add_argument("--log_every_epoch", type=int, default=10, help="Log and checkpoint every N epochs")
    parser.add_argument("--ckpt_every_epoch", type=int, default=10, help="Checkpoint every N epochs")
    
    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    
    # EMA (Exponential Moving Average)
    parser.add_argument("--use_ema", action='store_true', help="Use EMA for model weights")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    
    # FID Evaluation (only for image datasets, runs at log_every_epoch)
    parser.add_argument("--compute_fid", action='store_true', help="Compute FID during training (image datasets only)")
    parser.add_argument("--fid_num_samples", type=int, default=5000, help="Number of samples for FID calculation")
    parser.add_argument("--fid_inference_steps", type=int, default=50, help="Number of denoising steps for FID (default: 50, faster than 1000)")

    # Latent caching (speeds up training by pre-encoding images)
    parser.add_argument("--use_latent_cache", action='store_true', help="Pre-encode images to latent space and cache to disk")
    parser.add_argument("--latent_cache_dir", type=str, default=None, help="Directory for latent cache (default: data_path/.latent_cache)")
    parser.add_argument("--force_recompute_latents", action='store_true', help="Force recompute latents even if cache exists")

    args = parser.parse_args()
    
    print("Loaded config:", args) 
    
    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
    
    if args.use_wandb:
        # Convert args to dict for logging config
        config_dict = vars(args)
        accelerator.init_trackers(
            project_name=args.wandb_project, 
            config=config_dict,
            init_kwargs={"wandb": {"entity": args.wandb_entity, "name": args.wandb_run_name}}
        )
    
    device = accelerator.device

    # 1. Setup Data & Config
    if args.dataset_type == 'synthetic':
        from src.data.synthetic import SyntheticDataset
        dataset = SyntheticDataset(size=65536*2, type=args.synthetic_type)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        in_channels = 2
        input_size = 1
        patch_size = 1
        use_vae = False
        vae = None
    else:
        # Calculate real image size based on latent input_size * 8 (VAE downsample factor)
        real_image_size = args.input_size * 8

        # Scale input_size based on VAE compression (usually /8)
        vae = AutoencoderWrapper().to(device)
        vae.eval()
        # Freeze VAE
        for p in vae.parameters():
            p.requires_grad = False

        if args.use_latent_cache:
            # Use pre-encoded latent dataset (faster training)
            print("Using pre-encoded latent cache for training")
            dataloader = get_latent_dataloader(
                data_path=args.data_path,
                vae=vae,
                batch_size=args.batch_size,
                image_size=real_image_size,
                cache_dir=args.latent_cache_dir,
                device=device,
                force_recompute=args.force_recompute_latents
            )
            use_vae = False  # Latents are already encoded
        else:
            dataloader = get_dataloader(args.data_path, args.batch_size, real_image_size)
            use_vae = True

        in_channels = 4
        input_size = args.input_size
        patch_size = args.patch_size

    # Vit Params dict
    vit_conf = {'num_heads': args.vit_num_heads, 'rank': args.vit_rank} if args.mode == 'vit' else None

    model = DiT(
        input_size=input_size, 
        patch_size=patch_size, 
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mode=args.mode,
        vit_conf=vit_conf
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Scheduler
    # prediction_type="sample" means the model predicts x_0 (original sample)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        beta_start=0.0001, 
        beta_end=0.02, 
        beta_schedule="linear", 
        prediction_type="sample",
        clip_sample=False 
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    logger = get_logger(__name__)
    
    # EMA Model (for stable inference)
    ema_model = None
    if args.use_ema:
        from torch_ema import ExponentialMovingAverage
        ema_model = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}")

    # Resume Checkpoint Logic
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "" and os.path.exists(args.resume_from_checkpoint):
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            
            # Load EMA state if it exists
            if args.use_ema:
                ema_path = os.path.join(args.resume_from_checkpoint, "ema_model.pt")
                if os.path.exists(ema_path):
                    ema_model.load_state_dict(torch.load(ema_path, map_location=device))
                    print("EMA state loaded successfully")
                else:
                    print("Warning: EMA checkpoint not found, starting EMA from scratch")
            
            # Infer epoch from path if matches 'epoch_\d+'
            try:
                base_name = os.path.basename(os.path.normpath(args.resume_from_checkpoint))
                if base_name.startswith("epoch_"):
                    starting_epoch = int(base_name.split("_")[1]) + 1
                    # Ensure checkpoint_dir is the parent directory, not nested
                    if not args.checkpoint_dir.endswith(base_name):
                        # Already correct
                        pass
                    else:
                        # Fix: remove the epoch part from checkpoint_dir
                        args.checkpoint_dir = os.path.dirname(args.resume_from_checkpoint)
                        print(f"Adjusted checkpoint_dir to: {args.checkpoint_dir}")
            except Exception as e:
                print(f"Could not infer epoch from path, starting from epoch 0. Error: {e}")
        else:
             print(f"Checkpoint path {args.resume_from_checkpoint} not found. Starting from scratch.")
    
    # 3. Loss
    trinity_loss = TrinityLoss(
        sigma_gmm=args.sigma_gmm,
        lambda_gmm=args.lambda_gmm, 
        lambda_align=args.lambda_align, 
        lambda_reg=args.lambda_reg, 
        lambda_div=args.lambda_div,
        lambda_repul=args.lambda_repul,
        only_winner_align=args.only_winner_align
    )
    mse_loss = torch.nn.MSELoss()
    
    print(f"Starting training in mode: {args.mode}")
    
    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        current_step = epoch * len(dataloader)
        
        for step, batch in progress_bar:
            if use_vae:
                # Batch is images (B, 3, H, W) - need VAE encoding
                if isinstance(batch, (list, tuple)):
                    images, _ = batch
                else:
                    images = batch

                with torch.no_grad():
                    # Encode to Latents
                    # Note: VAE expects normalized inputs
                    latents = vae.encode(images.to(device, dtype=torch.float16))
            else:
                # Batch is pre-encoded latents or synthetic data
                if isinstance(batch, (list, tuple)):
                    latents, _ = batch
                else:
                    latents = batch
                latents = latents.to(device)
                
            # Sample Noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            
            # Sample Timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            
            # Add Noise using Scheduler
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            optimizer.zero_grad()
            
            # Predict
            # Model takes (x, t, y=None)
            pred = model(noisy_latents, timesteps, None)
            
            loss = 0
            current_step = epoch * len(dataloader) + step
            
            if args.mode == 'vit':
                # Warmup Schedule for Auxiliary Losses
                gmm_weight = args.lambda_gmm
                align_weight = args.lambda_align
                reg_weight = args.lambda_reg
                repul_weight = args.lambda_repul
                sigma_gmm = args.sigma_gmm
                
                if not args.no_schedule:
                    warmup_steps = args.warmup_steps
                    
                    # Ramping up from 0 to target
                    progress = min(1.0, current_step / warmup_steps)
                    align_weight *= progress
                    reg_weight *= progress
                    repul_weight *= progress # Repulsion also warms up to avoid early instability
                    
                # Temperature Annealing
                # Sigma GMM: Start High -> End Low
                if args.temp_anneal_steps > 0:
                    temp_progress = min(1.0, current_step / args.temp_anneal_steps)
                    sigma_gmm = args.sigma_start + (args.sigma_end - args.sigma_start) * temp_progress
                
                # pred is (w, mu, U, lam)
                # target is 'latents' (x_0)
                loss, loss_dict = trinity_loss(
                    latents, pred,
                    lambda_gmm=gmm_weight, 
                    lambda_align=align_weight, 
                    lambda_reg=reg_weight,
                    lambda_repul=repul_weight,
                    sigma_gmm=sigma_gmm
                )
            else:
                # pred is noise
                loss = mse_loss(pred, latents)
                loss_dict = {'mse': loss.item()}
                
            accelerator.backward(loss)
            optimizer.step()
            
            # Update EMA
            if args.use_ema:
                ema_model.update()
            
            if step % 10 == 0:
                progress_bar.set_postfix(**loss_dict)
                if args.use_wandb:
                    full_log = {"train_loss": loss.item(), "epoch": epoch, "step": current_step}
                    full_log.update(loss_dict)
                    if args.mode == 'vit':
                        full_log.update({
                            "align_weight": align_weight, 
                            "reg_weight": reg_weight, 
                            "sigma_gmm": sigma_gmm,
                            "repul_weight": repul_weight
                        })
                    
                    accelerator.log(full_log, step=current_step)
                
        if epoch % args.log_every_epoch == 0:
            if accelerator.is_local_main_process:
                # Validation / Plotting
                with ema_scope(ema_model, args.use_ema):
                    if args.dataset_type == 'synthetic':
                        print(f"Generating synthetic samples for epoch {epoch}...")
                        log_synthetic_eval(model, noise_scheduler, device, current_step, args.synthetic_type, wandb_on=args.use_wandb)
                    else:
                        print(f"Generating image samples for epoch {epoch}...")
                        log_image_eval(model, noise_scheduler, vae, device, current_step, args, wandb_on=args.use_wandb)

                        # Compute FID if enabled (only for image datasets)
                        if args.compute_fid and args.dataset_type == 'image_folder':
                            if accelerator.is_local_main_process:
                                print(f"Computing FID at epoch {epoch} (using {args.fid_inference_steps} inference steps, {accelerator.num_processes} GPUs)...")
                            fid_metrics = compute_fid(
                                model=model,
                                scheduler=noise_scheduler,
                                vae=vae,
                                num_samples=args.fid_num_samples,
                                device=device,
                                real_dataset_path=args.data_path,
                                temp_dir=f"{args.checkpoint_dir}/fid_temp",
                                mode=args.mode,
                                strategy='sample',
                                num_inference_steps=args.fid_inference_steps,
                                accelerator=accelerator
                            )
                            if accelerator.is_local_main_process:
                                log_fid_to_wandb(fid_metrics, current_step, wandb_on=args.use_wandb)
        
        if epoch % args.ckpt_every_epoch == 0 and accelerator.is_local_main_process:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = f"{args.checkpoint_dir}/epoch_{epoch}"
            accelerator.save_state(ckpt_path)
            
            # Save EMA state separately
            if args.use_ema:
                ema_path = os.path.join(ckpt_path, "ema_model.pt")
                torch.save(ema_model.state_dict(), ema_path)
                print(f"EMA state saved to {ema_path}")
    
    if args.use_wandb:
        accelerator.end_training()

if __name__ == "__main__":
    main()
