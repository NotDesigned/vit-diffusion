# VIT-Diffusion: Design & Architecture Plan

## 1. Project Structure

```
vit-diffusion/
├── configs/                # Configuration files (YAML/JSON)
│   ├── train.yaml          # Main training config
│   ├── model.yaml          # Model architecture config (VIT vs Standard)
│   └── eval.yaml           # Evaluation config
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py     # Standard Transformer Backbone (DiT-like)
│   │   └── heads.py        # Hydra-Head implementation & Standard Linear Head
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── loss.py         # Trinity Loss (GMM, Alignment, Spectral)
│   │   └── sampler.py      # Switching SDE & Standard IDDPM/DDIM
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py       # Image dataset loading
│   └── utils/
│       ├── __init__.py
│       ├── autoencoder.py  # VAE Wrapper (using diffusers/pretrained)
│       └── metrics.py      # FID/IS calculation wrappers
├── train.py                # Main training entry point
├── eval.py                 # Evaluation entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 2. Component Details

### 2.1 Base Model & Hydra Head (`src/models/`)
*   **Backbone:** A DiT (Diffusion Transformer) implementation. It takes latent $z_t$, time $t$, and context $c$ and outputs a hidden state $h$.
*   **Heads:**
    *   `StandardHead`: Projects $h \to \epsilon$ (Standard Diffusion).
    *   `HydraHead`: Projects $h \to \{w, \mu, U, \Lambda\}$.
        *   Implement the specific activations: Softmax for $w$, Identity for $\mu$, Sigmoid for $\Lambda$.
        *   $U$ is output as unconstrained, orthogonality handles by loss.

### 2.2 Autoencoder (`src/utils/autoencoder.py`)
*   We will use the `diffusers` library to load a pre-trained VAE (e.g., `stabilityai/sd-vae-ft-mse`).
*   This avoids training an AE from scratch and ensures a high-quality latent space.
*   Functions: `encode(image) -> latent`, `decode(latent) -> image`.

### 2.3 Loss Functions (`src/diffusion/loss.py`)
*   **Standard Mode:** MSE Loss $\|\epsilon - \epsilon_\theta\|^2$.
*   **VIT Mode:** `TrinityLoss`.
    *   `WinnerTakesAllLoss`: GMM log-likelihood.
    *   `SpectralAlignmentLoss`: Tangent subspace projection error.
    *   `Regularization`: Sparsity on $\Lambda$, Orthogonality on $U$, Entropy on $w$.

### 2.4 Sampling (`src/diffusion/sampler.py`)
*   **Standard:** Includes wrappers for `k-diffusion` samplers or simple DDIM/Euler.
*   **VIT:** `SwitchingSDESampler`.
    *   Implements the "Quantum Measurement" step ($k^* \sim w$).
    *   Implements the "Eigen-Dynamics" evolution step.

## 3. Workflow

1.  **Setup:**
    *   User installs requirements (`torch`, `diffusers`, `timm`, `einops`, etc.).
    *   Configures `configs/train.yaml` to select `mode: 'standard'` or `mode: 'vit'`.

2.  **Training:**
    *   `python train.py` loads the VAE (frozen).
    *   Data is encoded to latents on the fly or pre-cached.
    *   Backbone predicts geometry/noise.
    *   Loss is computed based on selected mode.

3.  **Evaluation:**
    *   `python eval.py` generates samples using the appropriate sampler.
    *   Calculates FID against the training set using `torch-fidelity`.
