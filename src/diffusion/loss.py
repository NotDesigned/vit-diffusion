import torch
import torch.nn as nn
import torch.nn.functional as F

class TrinityLoss(nn.Module):
    def __init__(self, sigma_gmm=1, lambda_align=1.0, lambda_reg=0.01, lambda_div=1.0):
        super().__init__()
        self.sigma_gmm = sigma_gmm
        self.lambda_align = lambda_align
        self.lambda_reg = lambda_reg
        self.lambda_div = lambda_div

    def forward(self, noise, pred, lambda_align=None, lambda_reg=None, lambda_div=None):
        """
        noise: (B, C, H, W) - True Gaussian Noise
        pred: tuple(w, mu, U, lam)
        lambda_align, lambda_reg: Optional overrides for weights
        """
        if lambda_align is None: lambda_align = self.lambda_align
        if lambda_reg is None: lambda_reg = self.lambda_reg
        if lambda_div is None: lambda_div = self.lambda_div

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
        # L_Align = || \mu - U Lambda U^T \mu ||^2 for winner
        idx_mu = winner_idx.view(B, 1, 1).expand(-1, 1, D) # (B, 1, D)
        mu_win = mu_flat.detach().gather(1, idx_mu).squeeze(1)
        target = mu_win.view(B, D, 1) # (B, D, 1)
        U_win = U.gather(1, winner_idx.view(B, 1, 1, 1).expand(-1, 1, D, U.shape[-1])).squeeze(1) # (B, D, R)
        lam_win = lam.gather(1, winner_idx.view(B, 1, 1).expand(-1, 1, lam.shape[-1])).squeeze(1) # (B, R)

        Ut_t = torch.matmul(U_win.transpose(2, 1), target) # Product: first. U_win.T (B, R, D) x target (B, D, 1) -> (B, R, 1)
        lam_weighted = Ut_t * lam_win.unsqueeze(2) # (B, R, 1) * (B, R, 1) -> (B, R, 1)
        
        recon = torch.matmul(U_win, lam_weighted).squeeze(2) # (B, D, 1) -> (B, D)

        l_align = F.mse_loss(recon, mu_win)
        
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
        
        total_loss = l_gmm + lambda_align * l_align + lambda_reg * (l_dim + l_ortho) + lambda_div * l_div
        
        return total_loss, {
            "gmm": l_gmm.item(),
            "align": l_align.item(),
            "dim": l_dim.item(),
            "ortho": l_ortho.item(),
            "div": l_div.item()
        }
