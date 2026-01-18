import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandardHead(nn.Module):
    """
    Standard Diffusion Head: Predicts the noise/score directly.
    Output: epsilon (B, C, H, W)
    """
    def __init__(self, hidden_size, out_channels, patch_size=2):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        # The backbone (DiT) usually outputs (B, L, hidden_size).
        # We project back to (B, L, patch_size*patch_size*out_channels)
        # Then unpatchify.
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, H, W):
        # x: (B, L, hidden_size)
        x = self.proj(x)
        return self.unpatchify(x, H, W)

    def unpatchify(self, x, H, W):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = H // p
        w = W // p
        c = self.out_channels
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

class HydraHead(nn.Module):
    """
    VIT-Diffusion Hydra Head.
    Outputs the geometric tuple G = {w, mu, U, Lambda}
    
    Notation:
    - K: Number of heads (modes)
    - R: Rank of local subspace
    - D: Total data dimension (C * H * W)
    """
    def __init__(self, hidden_size, out_channels, img_size, patch_size=2, num_heads=4, rank=4):
        super().__init__()
        self.K = num_heads
        self.R = rank
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        # self.D = out_channels * img_size * img_size # Total dimension
        
        # 1. Branch Probability w: (B, K)
        # We pool the sequence to get a global descriptor
        self.w_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_heads)
        )

        # 2. Eigen-Drift mu: (B, K, D) which is reshaped to (B, K, C, H, W)
        # We predict K distinct scores. 
        # Output shape from linear: (B, L, K * p*p*C)
        self.mu_proj = nn.Linear(hidden_size, num_heads * patch_size * patch_size * out_channels)

        # 3. Eigen-Basis U: (B, K, D, R)
        # We predict U from the spatial sequence to preserve local noise direction info.
        # Format: Each patch predicts its contribution to the global basis vectors.
        # Output: (B, L, K * p*p*C * R)
        self.u_proj = nn.Linear(hidden_size, num_heads * patch_size * patch_size * out_channels * rank)
        
        # 4. Eigen-Strength Lambda: (B, K, R)
        self.lambda_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_heads * rank)
        )

    def forward(self, x, H, W):
        """
        x: (B, L, hidden_size)
        """
        B, L, _ = x.shape
        
        # Global pooling for global params (w, U, Lambda)
        x_pool = x.mean(dim=1) # (B, hidden_size)
        
        # --- 1. w (Probabilities) ---
        w_logits = self.w_proj(x_pool) # (B, K)
        w = torch.softmax(w_logits, dim=-1)
        
        # --- 2. mu (Drift / Score) ---
        # mu is dense, so we predict it from the sequence x
        mu_flat = self.mu_proj(x) # (B, L, K * ppm * C)
        
        # Reshape to (B, K, C, H, W)
        # First split K
        mu_flat = mu_flat.view(B, L, self.K, -1) # (B, L, K, ppm*C)
        # We need to unpatchify for each K. 
        # Permute to (B*K, L, ...) to use unpatchify logic
        mu_flat = mu_flat.permute(0, 2, 1, 3).reshape(B * self.K, L, -1)
        mu = self.unpatchify(mu_flat, H, W) # (B*K, C, H, W)
        mu = mu.view(B, self.K, self.out_channels, H, W)
        
        # --- 3. U (Eigen-Basis) ---
        # Predict U spatially from x (B, L, H)
        U_flat = self.u_proj(x) # (B, L, K * ppm * C * R)
        ppm_C = self.patch_size * self.patch_size * self.out_channels
        
        # Reshape for unpatchify: (B*K*R, L, ppm_C)
        # Sequence of reshape/permutes needs to match mu's logic but extended for R
        # U_flat: (B, L, K, R, ppm*C)
        U_flat = U_flat.view(B, L, self.K, self.R, ppm_C)
        # Move Batch, K, R to front -> (B*K*R, L, ppm*C)
        U_flat = U_flat.permute(0, 2, 3, 1, 4).reshape(B * self.K * self.R, L, ppm_C)
        
        U_imgs = self.unpatchify(U_flat, H, W) # (B*K*R, C, H, W)
        U_vecs = U_imgs.flatten(1) # (B*K*R, D)
        U = U_vecs.view(B, self.K, -1, self.R) # (B, K, D, R)
        # Note: Orthogonality is enforced by loss, not here.
        
        # --- 4. Lambda (Eigen-Strength) ---
        lambda_logits = self.lambda_proj(x_pool) # (B, K * R)
        lam = torch.sigmoid(lambda_logits).view(B, self.K, self.R)
        
        return w, mu, U, lam

    def unpatchify(self, x, H, W):
        p = self.patch_size
        h = H // p
        w_dim = W // p
        c = self.out_channels
        x = x.reshape(shape=(x.shape[0], h, w_dim, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w_dim * p))
        return imgs
