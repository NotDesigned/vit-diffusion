# VIT-Diffusion: Varifold-Informed Topological Diffusion
**Technical Design Document v2.0**

## 1. Theoretical Foundation & Motivation

### 1.1 The Continuity Paradox
Standard generative models (GANs, VAEs, standard Diffusion) rely on Deep Neural Networks (DNNs) which are continuous function approximators. However, Optimal Transport (OT) theory, specifically Figalli's regularity theory, dictates that transport maps between multimodal distributions or non-convex supports contain **singularity sets** where the map is discontinuous.
* **Conflict:** Approximating discontinuous maps with continuous DNNs leads to **Mode Collapse** (ignoring modes) or **Mode Mixture** (generating spurious samples in between modes).
* **AE-OT Insight:** The AE-OT model explicitly separates **Manifold Embedding** (via AE) from **Optimal Transport** (via convex optimization) to handle these singularities.

### 1.2 The Varifold Perspective
VIT-Diffusion adopts the AE-OT philosophy of separation but replaces the explicit discrete OT solver with a learned geometric evolution. We treat the latent data manifold not as a set of points, but as a **Varifold** (a measure on the Grassmannian bundle).
* **Quantum-Geometric Representation:** The local geometry at any point $x_t$ is described by a density matrix $\rho$, representing a mixture of possible tangent spaces (logic branches):
    $$
    \rho(x_t) = \sum_{k=1}^K w_k(x_t) \cdot | \psi_k \rangle \langle \psi_k | \approx \sum_{k=1}^K w_k (U_k \Lambda_k U_k^T)
    $$
    where $w_k$ is the probability of branch $k$, and $U_k \Lambda_k U_k^T$ describes the local tangent space of that branch.

---

## 2. Model Architecture

The model consists of a standard Transformer Backbone (representing the Manifold Embedding) and a specialized Hydra-Head (representing the Transport/Geometry).

**Notation:**
* $B$: Batch Size
* $D$: Latent Dimension
* $K$: Number of Geometric Heads (Branches)
* $R$: Maximum Local Rank (Subspace Dimension)

### 2.1 The Hydra-Head (Output Layer)
Instead of a single score vector $\epsilon$, the model outputs a tuple $\mathcal{G} = \{w, \mu, U, \Lambda\}$.

| Output Symbol | Dimensions | Activation | Physical/Geometric Meaning |
| :--- | :--- | :--- | :--- |
| $w$ | $(B, K)$ | $\text{Softmax}$ | **Branch Probability:** The probability that the particle belongs to macroscopic mode $k$. Detects singularities via entropy $H(w)$. |
| $\mu$ | $(B, K, D)$ | $\text{Identity}$ | **Eigen-Drift:** The score function (denoising direction) specific to branch $k$. |
| $U$ | $(B, K, D, R)$ | $\text{Identity}^*$ | **Eigen-Basis:** The orthonormal basis vectors spanning the local tangent space of branch $k$. |
| $\Lambda$ | $(B, K, R)$ | $\text{Sigmoid}$ | **Eigen-Strength:** The effective dimension coefficients (soft rank) of the local manifold. |

*\*Note: Orthogonality of $U$ is enforced via Loss, not activation.*

---

## 3. Training Methodology (Trinity Loss) (Updated)

We utilize a competitive **Denoising Score Matching** strategy. We do not average errors; we reward the "Winner" branch to force symmetry breaking at singularities.

**Target:** Gaussian Noise $\epsilon \sim \mathcal{N}(0, I)$.

### 3.1 Loss Components
1. **Mixture of Geometries (GMM) - The "Competition"**
   Forces only the most accurate head to learn from each sample.
   $$ L_{GMM} = -\log \left( \sum_{k=1}^K w_k \cdot \exp\left( -\frac{\| \epsilon - \mu_k \|^2}{2\sigma_{gmm}^2} \right) \right) $$
   *Refinement:* We use Mean Squared Error (MSE) instead of Sum (SSE) for distance calculation to ensure stability across different resolutions. $\sigma_{gmm}=0.1$ effectively acts as a high weight multiplier ($\approx 50\times$).

2. **Spectral Alignment - The "Geometry"**
   Forces the winner's predicted local basis $U_k$ to align with the actual error direction.
   $$ L_{Align} = \sum_{k=1}^K w_k \| \epsilon - \text{Proj}_{U_k}(\epsilon) \|^2 $$
   *Refinement:* Weighted by $\lambda_{align} \approx 10.0$ with a warmup schedule (5k steps) to balance against the reconstruction loss.

3. **Geometric Regularization**
   * **Sparsity:** Minimize $\sum |\lambda|$ to encourage low-rank structures.
   * **Orthogonality:** Enforce $U^T U = I$.
   * **Diversity:** Maximize Entropy of *batch-average* weights to prevent mode collapse.

---

## 4. Usage

### 4.1 Training
The training script is fully command-line driven. 

**Quick Start (ViT Mode):**
```bash
./run_train.sh --data_path /path/to/images
```

**Full Parameter Reference (`train.py`):**

| Category | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| **General** | `--mode` | `'vit'` | Training mode: `vit` (Hydra Head) or `standard` (DiT). |
| | `--data_path` | `'./data'` | Path to image dataset folder. |
| | `--batch_size` | `8` | Batch size per GPU. |
| | `--learning_rate` | `1e-4` | Learning rate (AdamW). |
| | `--num_epochs` | `100` | Total training epochs. |
| | `--checkpoint_dir` | `'checkpoints'` | Directory to save model checkpoints. |
| **Model** | `--hidden_size` | `384` | Transformer embedding dimension. |
| | `--depth` | `12` | Number of Transformer blocks. |
| | `--num_heads` | `6` | Number of Attention heads. |
| | `--patch_size` | `2` | Patch size for Latent tokenization (2 for 32x32 latent). |
| | `--input_size` | `32` | Input Latent size (e.g., 32 for 256px images via VAE). |
| **ViT-Specific** | `--vit_num_heads` | `4` | Number of Geometric Hydra Heads ($K$). |
| | `--vit_rank` | `16` | Rank of local subspace ($R$). |
| **Loss** | `--warmup_steps` | `5000` | Steps to warmup auxiliary losses (Align/Reg). |
| | `--lambda_align` | `10.0` | Weight for Spectral Alignment Loss. |
| | `--lambda_reg` | `0.05` | Weight for Regularization (Sparsity/Ortho/Div). |
| | `--no_schedule` | `False` | Presence disables the warmup schedule. |
| **Logging** | `--use_wandb` | `False` | Enable Weights & Biases logging. |
| | `--wandb_project` | `'vit-diffusion'` | WandB Project Name. |
| | `--wandb_entity` | `None` | WandB Entity/Username. |
| | `--wandb_run_name` | `None` | WandB Run Name. |

### 4.2 Evaluation / Sampling

To generate samples from a trained checkpoint using the correct model config:

```bash
python eval.py --ckpt checkpoints/epoch_90 --num_samples 16 --mode vit
```

**Parameter Reference (`eval.py`):**

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--ckpt` | **Required** | Path to the checkpoint directory (containing `model.safetensors` or `pytorch_model.bin`). |
| `--output_dir` | `'generated_samples'` | Folder to save generated images. |
| `--num_samples` | `100` | Number of images to generate. |
| `--ref_dir` | `None` | Path to reference images for FID calculation (Requires `torch-fidelity`). |
| `--mode` | `'vit'` | Must match training mode. |
| `--hidden_size` | `384` | Must match training. |
| `--depth` | `12` | Must match training. |
| `--num_heads` | `6` | Must match training. |
| `--patch_size` | `2` | Must match training. |
| `--input_size` | `32` | Must match training. |
| `--vit_num_heads` | `4` | Must match training. |
| `--vit_rank` | `4` | Must match training. |

*(Note: `eval.py` currently defaults `vit_rank` to 4, please override if training used 16)*



### 3.1 Winner-Takes-All Loss (GMM)
Handles the discontinuity by allowing multiple valid "next steps" but penalizing the model if *none* are correct. Only the best head is strongly updated.

$$
L_{\text{GMM}} = - \log \left( \sum_{k=1}^K w_k \cdot \exp \left( - \frac{\| \epsilon - \mu_k \|^2}{2\sigma^2} \right) \right)
$$

### 3.2 Spectral Alignment Loss
Decouples direction ($U$) from dimension ($\Lambda$). Forces the true score $\epsilon$ to lie within the predicted tangent subspace of the winning branch.

$$
L_{\text{Align}} = \sum_{k=1}^K w_k \cdot \| \epsilon - (U_k \text{diag}(\Lambda_k) U_k^T) \epsilon \|^2
$$
*Efficient computation:* Compute as $U_k \cdot (\Lambda_k \cdot (U_k^T \cdot \epsilon))$ to avoid $O(D^2)$ matrix construction.

### 3.3 Regularization
* **Sparsity ($L_{\text{dim}}$):** $\sum |\Lambda_k|$. Compress $R$ to the minimum necessary intrinsic dimension.
* **Orthogonality ($L_{\text{ortho}}$):** $\| U_k^T U_k - I \|_F^2$. Ensure $U$ forms a valid frame.
* **Diversity ($L_{\text{div}}$):** $-H(\bar{w})$. Prevent mode collapse where only one head is utilized across the batch.

---

## 4. Inference Dynamics: Switching SDE

The inference process models a **Quantum Trajectory with Continuous Measurement**. At singularities, the system does not average drifts (which leads to the "void"); instead, it collapses to a single eigenstate.

### 4.1 The Equation
$$
\begin{aligned}
\text{Measurement:} & \quad k^* \sim \text{Categorical}(w(x_t)) \\
\text{Evolution:} & \quad dx_t = \mu_{k^*}(x_t) dt + \sigma \left( U_{k^*}(x_t) \sqrt{\Lambda_{k^*}(x_t)} \right) dW_t
\end{aligned}
$$

### 4.2 Behavior at Singularity
* **AE-OT Approach:** Detect singularity via dihedral angles and **abandon** the sample.
* **VIT-Diffusion Approach:** Detect singularity via high entropy in $w$ (e.g., $w \approx [0.5, 0.5]$). The sampling of $k^*$ forces a **Hard Selection**, causing the particle to physically move towards one mode, breaking the symmetry and resolving the ambiguity.

---

## 5. Usage Guide

### 5.1 Installation
```bash
pip install -r requirements.txt
```

### 5.2 Dataset Preparation
Structure your data as an ImageFolder:
```text
/path/to/data/
    category1/
        img1.jpg
    category2/
        img2.jpg
```

### 5.3 Training Command
The training script `train.py` supports standard argparse arguments which override `configs/train.yaml`.

#### Basic Usage
```bash
python train.py --config configs/train.yaml --data_path /path/to/data/afhq/train
```

#### Full Argument List

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Training** | | | |
| `--mode` | str | `vit` | Training mode: `vit` or `standard`. |
| `--data_path` | str | - | Path to dataset root. |
| `--batch_size` | int | 4 | Training batch size. |
| `--learning_rate`| float| 1e-4 | Learning rate. |
| `--num_epochs` | int | 100 | Total epochs. |
| `--checkpoint_dir`| str | `checkpoints`| Directory to save weights. |
| **Model** | | | |
| `--hidden_size` | int | 1024 | Transformer embedding dim. |
| `--depth` | int | 28 | Number of DiT blocks. |
| `--num_heads` | int | 16 | Number of attention heads. |
| `--input_size` | int | 32 | Latent spatial size (img_size/8). |
| **VIT-Diffusion**| | | |
| `--vit_num_heads`| int | 4 | Number of geometric branches ($K$). |
| `--vit_rank` | int | 4 | Local tangent subspace rank ($R$). |
| **Loss Schedule**| | | |
| `--warmup_steps` | int | 1000 | Steps to ramp up aux losses. |
| `--lambda_align` | float| 1.0 | Max weight for Alignment Loss. |
| `--lambda_reg` | float| 0.01 | Max weight for Regularization. |
| `--no_schedule` | flag | False | If set, disable warmup schedule. |
| **Logging**| | | |
| `--use_wandb` | flag | False | Enable WandB logging. |
| `--wandb_project`| str | `vit-diffusion`| WandB project name. |
| `--wandb_entity` | str | - | WandB username/entity. |
| `--wandb_run_name`| str | - | Optional run name. |

