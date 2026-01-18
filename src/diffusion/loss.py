import torch
import torch.nn as nn
import torch.nn.functional as F

class TrinityLoss(nn.Module):
    def __init__(self, sigma_gmm=1, lambda_align=1.0, lambda_reg=0.01):
        super().__init__()
        self.sigma_gmm = sigma_gmm
        self.lambda_align = lambda_align
        self.lambda_reg = lambda_reg

    def forward(self, noise, pred, lambda_align=None, lambda_reg=None):
        """
        noise: (B, C, H, W) - True Gaussian Noise
        pred: tuple(w, mu, U, lam)
        lambda_align, lambda_reg: Optional overrides for weights
        """
        if lambda_align is None: lambda_align = self.lambda_align
        if lambda_reg is None: lambda_reg = self.lambda_reg

        w, mu, U, lam = pred
        
        # Cast to float32 for numerical stability in Loss
        noise = noise.float()
        w = w.float()
        mu = mu.float()
        U = U.float()
        lam = lam.float()

        B, K = w.shape
        # Flatten noise/mu to (B, D) or (B, K, D)
        noise_flat = noise.flatten(1) # (B, D)
        mu_flat = mu.flatten(2) # (B, K, D)
        
        D = noise_flat.shape[1]

        # 1. Winner-Takes-All (GMM Loss)
        # L_GMM = - log( sum( w_k * exp( - ||eps - mu_k||^2 / 2sigma^2 ) ) )
        
        # Distance squared: (B, K)
        # Normalize by D to make sigma_gmm size-invariant (MSE instead of SSE)
        d_squared = torch.sum((noise_flat.unsqueeze(1) - mu_flat) ** 2, dim=2) / D
        
        # Log-Sum-Exp trick for numerical stability
        # log( sum( exp( log_wk - d^2/2s^2 ) ) )
        logits = torch.log(w + 1e-8) - d_squared / (2 * self.sigma_gmm**2)
        l_gmm = -torch.logsumexp(logits, dim=1).mean()
        
        # Identify winner for Alignment
        with torch.no_grad():
            winner_idx = torch.argmin(d_squared, dim=1) # (B,)

        # 2. Spectral Alignment Loss
        # L_Align = || eps - (U D U^T) eps ||^2
        # Only on winner branch to stabilize? Or weighted by w?
        # Design doc says: "sum w_k * || ... ||^2".
        # But efficiently: "U_k . (Lam_k . (U_k^T . eps))"
        
        # Since U is (B, K, D, R), doing this for full D is expensive.
        # But we must do it.
        # Let's parallelize over K or select winner? 
        # Weighted sum is better for differentiability of w.
        
        # Compute projection P(eps) = U * Lam * U^T * eps
        # x1 = U^T * eps -> (B, K, R, D) * (B, 1, D, 1) -> (B, K, R, 1)
        # x2 = Lam * x1 -> (B, K, R) * (B, K, R) -> (B, K, R)
        # x3 = U * x2 -> (B, K, D, R) * (B, K, R, 1) -> (B, K, D)
        
        eps_in = noise_flat.view(B, 1, -1, 1) # (B, 1, D, 1)
        Ut_eps = torch.matmul(U.transpose(2, 3), eps_in).squeeze(-1) # (B, K, R)
        
        lam_weighted = lam * Ut_eps # (B, K, R) element-wise
        
        projected_noise = torch.matmul(U, lam_weighted.unsqueeze(-1)).squeeze(-1) # (B, K, D)
        
        # Error
        align_err = torch.sum((noise_flat.unsqueeze(1) - projected_noise) ** 2, dim=2) / D # (B, K)
        
        # Weighted by w, only fix the alignment, so we detach w
        l_align = torch.sum(w.detach() * align_err, dim=1).mean() 
        
        # 3. Regularization
        # Sparsity: sum |Lambda|    
        l_dim = torch.mean(torch.abs(lam))
        
        # Orthogonality: || U^T U - I ||^2
        # Ut U: (B, K, R, R)
        gram = torch.matmul(U.transpose(2, 3), U)
        eye = torch.eye(gram.shape[-1], device=gram.device).view(1, 1, gram.shape[-1], gram.shape[-1])
        l_ortho = torch.mean((gram - eye) ** 2) * 1.0 / (gram.shape[-1] ** 2)
        
        # Diversity: -H(mean(w))
        # Maximize entropy of the *average* distribution across batch to prevent mode collapse.
        avg_w = w.mean(dim=0) # (K,)
        l_div = torch.sum(avg_w * torch.log(avg_w + 1e-8)) # Negative Entropy (we want to minimize this -> maximize entropy)
        
        total_loss = l_gmm + lambda_align * l_align + lambda_reg * (l_dim + l_ortho + l_div)
        
        return total_loss, {
            "gmm": l_gmm.item(),
            "align": l_align.item(),
            "dim": l_dim.item(),
            "ortho": l_ortho.item(),
            "div": l_div.item()
        }
