import torch
import torch.nn as nn
import torch.nn.functional as F


def unpatchify(x, H, W, patch_size, out_channels):
    """
    Converts patch tokens back to image format.
    x: (N, L, patch_size**2 * C)
    Returns: (N, C, H, W)
    """
    p = patch_size
    h = H // p
    w = W // p
    x = x.reshape(shape=(x.shape[0], h, w, p, p, out_channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(shape=(x.shape[0], out_channels, h * p, w * p))


class HydraHead(nn.Module):
    """
    VIT-Diffusion Hydra Head.
    Outputs the geometric tuple G = {w, mu, U}
    
    Notation:
    - K: Number of heads (modes)
    - R: Rank of local subspace
    - D: Total data dimension (C * H * W)
    
    Key Design: mu is computed as a linear combination of U basis vectors:
        μ_k = sum_r c_{k,r} * U_{k,:,r}
    
    This ensures mu lies in the subspace spanned by U.
    The coefficients c implicitly encode importance (no separate λ needed).
    
    Total # of parameters:
        - w: D -> K
        - c: D -> K * R (coefficients, then mu = U @ c)
        - U: D -> K * D * R
    """
    def __init__(self, hidden_size, out_channels, img_size, patch_size=2, num_heads=4, rank=4):
        super().__init__()
        self.K = num_heads
        self.R = rank
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        # self.D = out_channels * img_size * img_size # Total dimension
        
        # adaLN modulation (consistent with DiTFinalLayer)
        # Applies time/class conditioning to the pooled representation
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        
        # 1. Branch Probability w: (B, K)
        # We pool the sequence to get a global descriptor
        self.w_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_heads)
        )

        # 2. Coefficients c: (B, K, R) - mu is computed as c @ U^T (linear combo of basis)
        # This replaces direct mu prediction, ensuring mu ∈ span(U)
        self.c_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_heads * rank)
        )

        # 3. Eigen-Basis U: (B, K, D, R)
        # We predict U from the spatial sequence to preserve local noise direction info.
        # Format: Each patch predicts its contribution to the global basis vectors.
        # Output: (B, L, K * p*p*C * R)
        self.u_proj = nn.Linear(hidden_size, num_heads * patch_size * patch_size * out_channels * rank)

    def forward(self, x, c, H, W):
        """
        x: (B, L, hidden_size)
        c: (B, hidden_size) - conditioning (time + class embedding)
        
        Returns:
            w: (B, K) - Branch probabilities
            mu: (B, K, C, H, W) - Predicted states (computed as U @ c)
            U: (B, K, D, R) - Eigen-basis
        """
        B, L, _ = x.shape
        D = self.out_channels * H * W  # Total dimension
        
        # Global pooling with adaLN conditioning
        x_pool = x.mean(dim=1)  # (B, hidden_size)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_pool = (1 + scale) * self.norm(x_pool) + shift
        
        # --- 1. w (Probabilities) ---
        w_logits = self.w_proj(x_pool)  # (B, K)
        w = torch.softmax(w_logits, dim=-1)
        
        # --- 2. U (Eigen-Basis) --- (computed BEFORE mu, since mu depends on U)
        # Predict U spatially from x (B, L, H)
        U_flat = self.u_proj(x)  # (B, L, K * ppm * C * R)
        ppm_C = self.patch_size * self.patch_size * self.out_channels
        
        # Reshape for unpatchify: (B*K*R, L, ppm_C)
        U_flat = U_flat.view(B, L, self.K, self.R, ppm_C)
        U_flat = U_flat.permute(0, 2, 3, 1, 4).reshape(B * self.K * self.R, L, ppm_C)

        U_imgs = unpatchify(U_flat, H, W, self.patch_size, self.out_channels)  # (B*K*R, C, H, W)
        U_vecs = U_imgs.flatten(1)  # (B*K*R, D)
        
        # Reshape to (B, K, R, D) then permute to (B, K, D, R)
        U = U_vecs.view(B, self.K, self.R, -1)  # (B, K, R, D)
        U = U.permute(0, 1, 3, 2)               # (B, K, D, R)
        
        # --- 3. c (Coefficients) and mu (Drift) ---
        # c: (B, K, R) - coefficients for linear combination of U basis vectors
        c = self.c_proj(x_pool).view(B, self.K, self.R)  # (B, K, R)
        
        # mu = sum_r c_r * U[:, r] = U @ c  (linear combination)
        # U: (B, K, D, R), c: (B, K, R) -> mu_flat: (B, K, D)
        mu_flat = torch.einsum('bkdr,bkr->bkd', U, c)  # (B, K, D)
        
        # Reshape to image format: (B, K, C, H, W)
        mu = mu_flat.view(B, self.K, self.out_channels, H, W)
        
        return w, mu, U
