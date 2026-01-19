import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Individual Loss Functions (Modular)
# =============================================================================

def compute_gmm_loss(x0_flat, mu_flat, w, sigma_gmm):
    """
    Winner-Takes-All GMM Loss.
    
    L_GMM = -log( sum_k w_k * exp( -||x0 - mu_k||^2 / 2σ² ) )
    
    Args:
        x0_flat: (B, D) - Flattened ground truth
        mu_flat: (B, K, D) - Flattened predictions for each head
        w: (B, K) - Branch probabilities
        sigma_gmm: float - Temperature parameter
        
    Returns:
        l_gmm: scalar loss
        d_squared: (B, K) - Squared distances (for winner selection)
    """
    D = x0_flat.shape[1]
    
    # Distance squared: (B, K), normalized by D for size-invariance
    d_squared = torch.sum((x0_flat.unsqueeze(1) - mu_flat) ** 2, dim=2) / D
    
    # Log-Sum-Exp trick for numerical stability
    logits = torch.log(w + 1e-8) - d_squared / (2 * sigma_gmm**2)
    l_gmm = -torch.logsumexp(logits, dim=1).mean() * sigma_gmm**2
    
    return l_gmm, d_squared


def compute_repulsion_loss(mu_flat, w):
    """
    Repulsion Loss for Symmetry Breaking.
    
    L_repul = sum_{i≠j} sqrt(w_i * w_j) * max(0, cos(μ_i, μ_j))
    
    Forces heads with high probability to predict different directions.
    
    Args:
        mu_flat: (B, K, D) - Flattened predictions
        w: (B, K) - Branch probabilities
        
    Returns:
        l_repul: scalar loss
    """
    B, K, D = mu_flat.shape
    
    if K <= 1:
        return torch.tensor(0.0, device=mu_flat.device)
    
    # Normalize mu for cosine similarity: (B, K, D)
    mu_norm = F.normalize(mu_flat, p=2, dim=2)
    
    # Cosine Similarity Matrix: (B, K, K)
    sim_matrix = torch.matmul(mu_norm, mu_norm.transpose(1, 2))
    
    # Gating Matrix: sqrt(w_i * w_j) -> (B, K, K)
    w_sqrt = torch.sqrt(w + 1e-8)
    gate_matrix = torch.matmul(w_sqrt.unsqueeze(2), w_sqrt.unsqueeze(1))
    
    # Mask diagonal (no self-penalty)
    mask = torch.eye(K, device=mu_flat.device).unsqueeze(0)
    
    # Only penalize positive cosine similarity
    sim_penalty = torch.relu(sim_matrix)
    
    l_repul = torch.sum(gate_matrix * sim_penalty * (1 - mask)) / (B * K * (K - 1))
    
    return l_repul


def compute_alignment_loss_winner_only(mu_flat, U, lam, winner_idx):
    """
    Spectral Alignment Loss (Winner Only).
    
    L_align = || μ_win - U_win Λ_win U_win^T μ_win ||²
    
    Forces U to align with the displacement direction for the winner head only.
    
    Args:
        mu_flat: (B, K, D) - Flattened predictions
        U: (B, K, D, R) - Eigen-basis
        lam: (B, K, R) - Eigen-strength
        winner_idx: (B,) - Index of winning head per sample
        
    Returns:
        l_align: scalar loss
        mu_win: (B, D) - Winner's mu (for logging)
    """
    B = mu_flat.shape[0]
    D = mu_flat.shape[2]
    R = U.shape[-1]
    
    # Gather winner's mu, U, lam
    idx_mu = winner_idx.view(B, 1, 1).expand(-1, 1, D)
    mu_win = mu_flat.detach().gather(1, idx_mu).squeeze(1)  # (B, D)
    
    idx_U = winner_idx.view(B, 1, 1, 1).expand(-1, 1, D, R)
    U_win = U.gather(1, idx_U).squeeze(1)  # (B, D, R)
    
    idx_lam = winner_idx.view(B, 1, 1).expand(-1, 1, R)
    lam_win = lam.gather(1, idx_lam).squeeze(1)  # (B, R)
    
    # Projection: U Λ U^T μ
    target = mu_win.view(B, D, 1)
    Ut_t = torch.matmul(U_win.transpose(1, 2), target)  # (B, R, 1)
    lam_weighted = Ut_t * lam_win.unsqueeze(2)  # (B, R, 1)
    recon = torch.matmul(U_win, lam_weighted).squeeze(2)  # (B, D)
    
    l_align = F.mse_loss(recon, mu_win)
    
    return l_align, mu_win


def compute_alignment_loss_all_heads(mu_flat, U, lam, w):
    """
    Spectral Alignment Loss (All Heads, Weighted).
    
    L_align = sum_k w_k * || μ_k - U_k Λ_k U_k^T μ_k ||²
    
    Args:
        mu_flat: (B, K, D) - Flattened predictions
        U: (B, K, D, R) - Eigen-basis
        lam: (B, K, R) - Eigen-strength
        w: (B, K) - Branch probabilities (detached for weighting)
        
    Returns:
        l_align: scalar loss
    """
    B, K, D = mu_flat.shape
    
    # Projection for all heads: U Λ U^T μ
    mu_in = mu_flat.view(B, K, D, 1)
    Ut_mu = torch.matmul(U.transpose(2, 3), mu_in).squeeze(-1)  # (B, K, R)
    lam_weighted = lam * Ut_mu  # (B, K, R)
    projected_mu = torch.matmul(U, lam_weighted.unsqueeze(-1)).squeeze(-1)  # (B, K, D)
    
    # Error per head
    align_err = torch.sum((mu_flat - projected_mu) ** 2, dim=2) / D  # (B, K)
    
    # Weighted sum (detach w to only update U, not w)
    l_align = torch.sum(w.detach() * align_err, dim=1).mean()
    
    return l_align


def compute_orthogonality_loss(U):
    """
    Orthogonality Regularization.
    
    L_ortho = || U^T U - I ||²
    
    Forces basis vectors to be orthonormal.
    
    Args:
        U: (B, K, D, R) - Eigen-basis
        
    Returns:
        l_ortho: scalar loss
    """
    R = U.shape[-1]
    
    # Gram matrix: U^T U -> (B, K, R, R)
    gram = torch.matmul(U.transpose(2, 3), U)
    eye = torch.eye(R, device=U.device).view(1, 1, R, R)
    
    l_ortho = torch.mean((gram - eye) ** 2) / (R ** 2)
    
    return l_ortho


def compute_dimension_loss(lam, threshold=0.1):
    """
    Dimension Regularization (Encourage Effective Rank).
    
    L_dim = mean(relu(threshold - λ))
    
    Penalizes eigenvalues below threshold to encourage full rank usage.
    
    Args:
        lam: (B, K, R) - Eigen-strength
        threshold: float - Minimum eigenvalue threshold
        
    Returns:
        l_dim: scalar loss
    """
    l_dim = torch.mean(torch.relu(threshold - lam))
    return l_dim


def compute_diversity_loss(w):
    """
    Diversity Loss (Batch-level Entropy).
    
    L_div = -H(mean(w)) = sum(avg_w * log(avg_w))
    
    Maximizes entropy of average branch probability across batch.
    
    WARNING: This loss uses batch statistics and may cause issues with:
    - Multi-GPU training (different batches have different avg_w)
    - Gradient accumulation (partial batches)
    
    Consider setting lambda_div=0 and relying on L_repul for diversity.
    
    Args:
        w: (B, K) - Branch probabilities
        
    Returns:
        l_div: scalar loss (negative entropy, minimize to maximize entropy)
    """
    avg_w = w.mean(dim=0)  # (K,)
    l_div = torch.sum(avg_w * torch.log(avg_w + 1e-8))
    return l_div


# =============================================================================
# Main Loss Class
# =============================================================================

class TrinityLoss(nn.Module):
    """
    Trinity Loss for VIT-Diffusion.
    
    L_total = λ_gmm * L_GMM 
            + λ_align * (L_align + L_ortho) 
            + λ_reg * L_dim 
            + λ_div * L_div 
            + λ_repul * L_repul
    """
    
    def __init__(
        self, 
        sigma_gmm, 
        lambda_gmm=10.0, 
        lambda_align=1.0, 
        lambda_reg=1.0, 
        lambda_div=1.0,  # Consider setting to 0, see README
        lambda_repul=1.0, 
        only_winner_align=False
    ):
        super().__init__()
        self.sigma_gmm = sigma_gmm
        self.lambda_gmm = lambda_gmm
        self.lambda_align = lambda_align
        self.lambda_reg = lambda_reg
        self.lambda_div = lambda_div
        self.lambda_repul = lambda_repul
        self.only_winner_align = only_winner_align

    def forward(
        self, 
        x0, 
        pred, 
        lambda_gmm=None, 
        lambda_align=None, 
        lambda_reg=None, 
        lambda_div=None, 
        lambda_repul=None, 
        sigma_gmm=None
    ):
        """
        Compute Trinity Loss.
        
        Args:
            x0: (B, C, H, W) - Ground Truth Sample
            pred: tuple(w, mu, U, lam) - Model predictions
            lambda_*: Optional weight overrides
            sigma_gmm: Optional temperature override
            
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss values
        """
        # Apply defaults
        if lambda_gmm is None: lambda_gmm = self.lambda_gmm
        if lambda_align is None: lambda_align = self.lambda_align
        if lambda_reg is None: lambda_reg = self.lambda_reg
        if lambda_div is None: lambda_div = self.lambda_div
        if lambda_repul is None: lambda_repul = self.lambda_repul
        if sigma_gmm is None: sigma_gmm = self.sigma_gmm

        w, mu, U, lam = pred

        # Disable autocast for numerical stability
        with torch.autocast(device_type=x0.device.type, enabled=False):
            # Cast to float32
            x0 = x0.float()
            w = w.float()
            mu = mu.float()
            U = U.float()
            lam = lam.float()

            B, K = w.shape
            x0_flat = x0.flatten(1)      # (B, D)
            mu_flat = mu.flatten(2)      # (B, K, D)

            # 1. GMM Loss (Winner-Takes-All)
            l_gmm, d_squared = compute_gmm_loss(x0_flat, mu_flat, w, sigma_gmm)
            
            # Identify winner
            with torch.no_grad():
                winner_idx = torch.argmin(d_squared, dim=1)  # (B,)

            # 2. Repulsion Loss (Symmetry Breaking)
            l_repul = compute_repulsion_loss(mu_flat, w)

            # 3. Spectral Alignment Loss
            if self.only_winner_align:
                l_align, mu_win = compute_alignment_loss_winner_only(
                    mu_flat, U, lam, winner_idx
                )
            else:
                l_align = compute_alignment_loss_all_heads(mu_flat, U, lam, w)
                # Get mu_win for logging
                D = x0_flat.shape[1]
                idx_mu = winner_idx.view(B, 1, 1).expand(-1, 1, D)
                mu_win = mu_flat.detach().gather(1, idx_mu).squeeze(1)

            # 4. Orthogonality Regularization
            l_ortho = compute_orthogonality_loss(U)

            # 5. Dimension Regularization
            l_dim = compute_dimension_loss(lam)

            # 6. Diversity Loss (Batch Entropy) - Consider disabling (lambda_div=0)
            l_div = compute_diversity_loss(w)

            # Total Loss
            total_loss = (
                lambda_gmm * l_gmm 
                + lambda_align * (l_align + l_ortho) 
                + lambda_reg * l_dim 
                + lambda_div * l_div 
                + lambda_repul * l_repul
            )

            return total_loss, {
                "gmm": l_gmm.item(),
                "align": l_align.item(),
                "ortho": l_ortho.item(),
                "dim": l_dim.item(),
                "div": l_div.item(),
                "repul": l_repul.item(),
                "win_mu_loss": F.mse_loss(mu_win, x0_flat).item(),
            }
