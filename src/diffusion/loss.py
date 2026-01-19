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
    l_gmm = -torch.logsumexp(logits, dim=1).mean() * (sigma_gmm**2)
    
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
            + λ_ortho * L_ortho 
            + λ_div * L_div 
            + λ_repul * L_repul
    
    Note: λ (Eigen-Strength) has been removed. mu = U @ c ensures mu is in span(U),
    and the coefficients c implicitly encode importance.
    """
    
    def __init__(
        self, 
        sigma_gmm, 
        lambda_gmm=10.0, 
        lambda_ortho=1.0,
        lambda_div=1.0,  # Consider setting to 0, see README
        lambda_repul=1.0, 
    ):
        super().__init__()
        self.sigma_gmm = sigma_gmm
        self.lambda_gmm = lambda_gmm
        self.lambda_ortho = lambda_ortho
        self.lambda_div = lambda_div
        self.lambda_repul = lambda_repul

    def forward(
        self, 
        x0, 
        pred, 
        lambda_gmm=None, 
        lambda_ortho=None, 
        lambda_div=None, 
        lambda_repul=None, 
        sigma_gmm=None
    ):
        """
        Compute Trinity Loss.
        
        Args:
            x0: (B, C, H, W) - Ground Truth Sample
            pred: tuple(w, mu, U) - Model predictions
            lambda_*: Optional weight overrides
            sigma_gmm: Optional temperature override
            
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss values
        """
        # Apply defaults
        if lambda_gmm is None: lambda_gmm = self.lambda_gmm
        if lambda_ortho is None: lambda_ortho = self.lambda_ortho
        if lambda_div is None: lambda_div = self.lambda_div
        if lambda_repul is None: lambda_repul = self.lambda_repul
        if sigma_gmm is None: sigma_gmm = self.sigma_gmm

        w, mu, U = pred

        # Disable autocast for numerical stability
        with torch.autocast(device_type=x0.device.type, enabled=False):
            # Cast to float32
            x0 = x0.float()
            w = w.float()
            mu = mu.float()
            U = U.float()

            B, K = w.shape
            D = x0.flatten(1).shape[1]
            x0_flat = x0.flatten(1)      # (B, D)
            mu_flat = mu.flatten(2)      # (B, K, D)

            # 1. GMM Loss (Winner-Takes-All)
            l_gmm, d_squared = compute_gmm_loss(x0_flat, mu_flat, w, sigma_gmm)
            
            # Identify winner for logging
            with torch.no_grad():
                winner_idx = torch.argmin(d_squared, dim=1)  # (B,)
                idx_mu = winner_idx.view(B, 1, 1).expand(-1, 1, D)
                mu_win = mu_flat.detach().gather(1, idx_mu).squeeze(1)

            # 2. Repulsion Loss (Symmetry Breaking)
            l_repul = compute_repulsion_loss(mu_flat, w)

            # 3. Orthogonality Regularization
            l_ortho = compute_orthogonality_loss(U)

            # 4. Diversity Loss (Batch Entropy) - Consider disabling (lambda_div=0)
            l_div = compute_diversity_loss(w)

            # Total Loss
            total_loss = (
                lambda_gmm * l_gmm 
                + lambda_ortho * l_ortho 
                + lambda_div * l_div 
                + lambda_repul * l_repul
            )

            return total_loss, {
                "gmm": l_gmm.item(),
                "ortho": l_ortho.item(),
                "div": l_div.item(),
                "repul": l_repul.item(),
                "win_mu_loss": F.mse_loss(mu_win, x0_flat).item(),
            }
