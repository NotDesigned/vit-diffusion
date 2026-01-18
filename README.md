# VIT-Diffusion: Varifold-Informed Topological Diffusion
**Technical Implemenation v2.1**

## 1. Theoretical Foundation & Motivation

### 1.1 The Continuity Paradox
Standard generative models (GANs, VAEs, standard Diffusion) rely on Deep Neural Networks (DNNs) which are continuous function approximators. However, Optimal Transport (OT) theory dictates that transport maps between multimodal distributions contain **singularity sets** (ridges) where the map must be discontinuous (i.e. splitting symmetry).
* **Conflict:** Approximating discontinuous maps with continuous DNNs leads to **Mode Collapse** (ignoring modes) or **Mode Mixture** (generating spurious average samples).
* **Solution:** VIT-Diffusion treats the latent space not as a single manifold, but as a probabilistic mixture of local geometries ("Hydra Head").

### 1.2 The Hydra-Head Architecture
Instead of predicting a single noise vector $\epsilon$, we predict a tuple $\mathcal{G} = \{w, \mu, U, \Lambda\}$:

| Symbol | Dimensions | Meaning |
| :--- | :--- | :--- |
| $w$ | $(B, K)$ | **Branch Probabilities**: Which macroscopic mode does this particle belong to? |
| $\mu$ | $(B, K, D)$ | **Eigen-Drift**: $K$ distinct proposed directions (vectors). |
| $U$ | $(B, K, D, R)$ | **Eigen-Basis**: Local tangent space directions (for manifold dimension reduction). |
| $\Lambda$ | $(B, K, R)$ | **Eigen-Strength**: Importance of each basis vector. |

---

## 2. Trinity Loss: Advanced Physics-Informed Training

The loss function is designed to handle singular regions via competition and symmetry breaking.

$$ L_{total} = L_{GMM} + \lambda_{align} L_{align} + \lambda_{repul} L_{repul} + L_{reg} $$

### 2.1 Winner-Takes-All (GMM Loss)
$$ L_{GMM} = -\log \sum_k w_k \exp\left(-\frac{\|\epsilon - \mu_k\|^2}{2\sigma^2}\right) $$
*   **Mechanism**: A soft competition. Only the head closest to the ground truth gets the gradient signal.
*   **Temperature Annealing**: We accept `sigma_gmm` to decay from high (melting phase) to low (freezing phase) during training to assist convergence.

### 2.2 Repulsion Loss (Symmetry Breaking)
$$ L_{repul} = \sum_{i \neq j} \sqrt{w_i w_j} \cdot \max(0, \cos(\mu_i, \mu_j)) $$
*   **Mechanism**: If two heads have high probability ($w_i, w_j$ both high) for the same point, their directions $\mu_i, \mu_j$ are forced to be **different** (cosine similarity penalized).
*   **Effect**: This actively forces bifurcation at saddle points.

### 2.3 Spectral Alignment
Forces the geometric basis $U$ to align with the drift direction, effectively learning the local dimensionality of the manifold.

---

## 3. Usage

### 3.1 Synthetic Experiments (Recommended First Step)
Verify the "Bifurcation" mechanism on 2D toy datasets (Swiss Roll, 8-Gaussians).

**Dataset Options**: `swiss_roll`, `8gaussians`, `checkerboard`, `olympic`.

```bash
# Train on 8-Gaussians (Multimodal test)
# Will generate Bifurcation GIFs in WandB
./train_synthetic.sh --synthetic_type 8gaussians --wandb_run_name test_bifurcation
```

**Key Arguments for Synthetic**:
*   `--mixed_precision "no"`: Crucial for 2D coordinate precision.
*   `--vit_rank 2`: Full rank for 2D space.
*   `--batch_size 1024`: Large batch for stable gradients.

### 3.2 Image Training (AFHQ / ImageNet)
Train on real image datasets using VAE Latent Space.

```bash
# Standard Training
./run_train.sh --data_path /path/to/images --batch_size 32
```

**Resume Training**:
The script supports resuming from the last checkpoint automatically if provided.
```bash
./run_train.sh --resume_from_checkpoint checkpoints/run_001/epoch_50
```

---

## 4. Visualization & Evaluation

The training script automatically performs advanced visualization (if WandB is enabled).

### 4.1 Scatter Plots (`eval/scatter_sample` vs `mean`)
*   **Sampled**: Uses $k \sim w$ sampling. Should strictly follow the data distribution multimodally.
*   **Mean**: Uses $\sum w \mu$ averaging. Should collapse to the center (failure mode) if the model works correctly.

### 4.2 Vector Field GIF (`eval/diff_process`)
A generic GIF showing the reverse diffusion process $t: 1000 \to 0$.
*   **High Noise**: Unimodal, chaotic field.
*   **Bifurcation Point**: Arrows split into multiple directions (transparent ghosts).
*   **Low Noise**: Clean flow to separate modes.

---

## 5. Folder Structure
*   `src/models`: DiT Backbone and Hydra/Standard Heads.
*   `src/diffusion`: Trinity Loss implementation.
*   `src/data`: Synthetic dataset generators.
*   `src/vis`: Plotting and GIF generation logic.
*   `train.py`: Main trainer.

