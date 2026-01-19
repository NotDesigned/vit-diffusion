import torch
import torch.nn as nn
import torch.nn.functional as F

class TrinityLoss(nn.Module):
    def __init__(self, sigma_gmm, lambda_gmm=10.0, lambda_align=1.0, lambda_reg=1.0, lambda_div=1.0, lambda_repul=1.0, only_winner_align=False):
        super().__init__()
        self.sigma_gmm = sigma_gmm
        self.lambda_gmm = lambda_gmm
        self.lambda_align = lambda_align
        self.lambda_reg = lambda_reg
        self.lambda_div = lambda_div
        self.lambda_repul = lambda_repul
        self.only_winner_align = only_winner_align

    def forward(self, x0, pred, lambda_gmm=None, lambda_align=None, lambda_reg=None, lambda_div=None, lambda_repul=None, sigma_gmm=None):
        """
        x0: (B, C, H, W) - Ground Truth Sample
        pred: tuple(w, mu, U, lam)
        lambda_* : Optional overrides for weights
        sigma_gmm: Optional override for temperature annealing
        """
        if lambda_gmm is None: lambda_gmm = self.lambda_gmm
        if lambda_align is None: lambda_align = self.lambda_align
        if lambda_reg is None: lambda_reg = self.lambda_reg
        if lambda_div is None: lambda_div = self.lambda_div
        if lambda_repul is None: lambda_repul = self.lambda_repul
        if sigma_gmm is None: sigma_gmm = self.sigma_gmm

        w, mu, U, lam = pred

        # Disable autocast for numerical stability (logsumexp, matmul can overflow in fp16/bf16)
        with torch.autocast(device_type=x0.device.type, enabled=False):
            x0 = x0.float()
            w = w.float()
            mu = mu.float()
            U = U.float()
            lam = lam.float()

            B, K = w.shape
            # Flatten x0/mu to (B, D) or (B, K, D)
            x0_flat = x0.flatten(1)  # (B, D)
            mu_flat = mu.flatten(2)  # (B, K, D)

            D = x0_flat.shape[1]

            # 1. Winner-Takes-All (GMM Loss)
            # L_GMM = - log( sum( w_k * exp( - ||x0 - mu_k||^2 / 2sigma^2 ) ) )

            # Distance squared: (B, K)
            # Normalize by D to make sigma_gmm size-invariant (MSE instead of SSE)
            d_squared = torch.sum((x0_flat.unsqueeze(1) - mu_flat) ** 2, dim=2) / D

            # Log-Sum-Exp trick for numerical stability
            # log( sum( exp( log_wk - d^2/2s^2 ) ) )
            logits = torch.log(w + 1e-8) - d_squared / (2 * sigma_gmm**2)
            l_gmm = -torch.logsumexp(logits, dim=1).mean() * sigma_gmm**2

            # Identify winner for Alignment
            with torch.no_grad():
                winner_idx = torch.argmin(d_squared, dim=1)  # (B,)

            # 4. Repulsion Loss (Symmetry Breaking)
            # L_repul = sum_{i!=j} sqrt(w_i w_j) * max(0, cos(mu_i, mu_j))
            # Only compute if K > 1
            l_repul = torch.tensor(0.0, device=x0.device)
            if K > 1:
                # Flatten mu to (B, K, D) which is already mu_flat
                # Normalize mu for cosine similarity: (B, K, D)
                mu_norm = F.normalize(mu_flat, p=2, dim=2)

                # Compute Cosine Similarity Matrix: (B, K, K)
                # (B, K, D) @ (B, D, K) -> (B, K, K)
                sim_matrix = torch.matmul(mu_norm, mu_norm.transpose(1, 2))

                # Gating Matrix: sqrt(w_i * w_j) -> (B, K, K)
                # w: (B, K) -> w.sqrt(): (B, K)
                w_sqrt = torch.sqrt(w + 1e-8)
                gate_matrix = torch.matmul(w_sqrt.unsqueeze(2), w_sqrt.unsqueeze(1))

                # Mask diagonal (self-similarity should not be penalized)
                mask = torch.eye(K, device=x0.device).unsqueeze(0)  # (1, K, K)

                # We want to penalized high similarity (cos close to 1)
                # only when gate is high (ambiguous region)
                # Only penalize positive cosine similarity
                sim_penalty = torch.relu(sim_matrix)

                l_repul = torch.sum(gate_matrix * sim_penalty * (1 - mask)) / (B * K * (K - 1))

            # 2. Spectral Alignment Loss
            # L_Align = || \mu - U Lambda U^T \mu ||^2 for winner
            idx_mu = winner_idx.view(B, 1, 1).expand(-1, 1, D)  # (B, 1, D)
            mu_win = mu_flat.detach().gather(1, idx_mu).squeeze(1)
            if self.only_winner_align:
                target = mu_win.view(B, D, 1)  # (B, D, 1)
                U_win = U.gather(1, winner_idx.view(B, 1, 1, 1).expand(-1, 1, D, U.shape[-1])).squeeze(1)  # (B, D, R)
                lam_win = lam.gather(1, winner_idx.view(B, 1, 1).expand(-1, 1, lam.shape[-1])).squeeze(1)  # (B, R)

                Ut_t = torch.matmul(U_win.transpose(2, 1), target)  # U_win.T (B, R, D) x target (B, D, 1) -> (B, R, 1)
                lam_weighted = Ut_t * lam_win.unsqueeze(2)  # (B, R, 1) * (B, R, 1) -> (B, R, 1)

                recon = torch.matmul(U_win, lam_weighted).squeeze(2)  # (B, D, 1) -> (B, D)

                l_align = F.mse_loss(recon, mu_win)
            else:
                mu_in = mu_flat.view(B, K, D, 1)  # (B, K, D, 1)
                Ut_mu = torch.matmul(U.transpose(2, 3), mu_in).squeeze(-1)  # (B, K, R)

                lam_weighted = lam * Ut_mu  # (B, K, R) element-wise

                projected_mu = torch.matmul(U, lam_weighted.unsqueeze(-1)).squeeze(-1)  # (B, K, D)

                # Error
                align_err = torch.sum((mu_flat - projected_mu) ** 2, dim=2) / D  # (B, K)

                # Weighted by w, only fix the alignment, so we detach w
                l_align = torch.sum(w.detach() * align_err, dim=1).mean()

            # 3. Regularization
            # Sparsity: sum |Lambda|
            # l_dim = torch.mean(torch.abs(lam))
            # only penalize small eigenvalues to encourage effective rank
            l_dim = torch.mean(torch.relu(0.1 - lam))

            # Orthogonality: || U^T U - I ||^2
            # Ut U: (B, K, R, R)
            gram = torch.matmul(U.transpose(2, 3), U)
            eye = torch.eye(gram.shape[-1], device=gram.device).view(1, 1, gram.shape[-1], gram.shape[-1])
            l_ortho = torch.mean((gram - eye) ** 2) * 1.0 / (gram.shape[-1] ** 2)

            # Diversity: -H(mean(w))
            # Maximize entropy of the *average* distribution across batch to prevent mode collapse.
            avg_w = w.mean(dim=0)  # (K,)
            l_div = torch.sum(avg_w * torch.log(avg_w + 1e-8))  # Negative Entropy (minimize -> maximize entropy)

            total_loss = lambda_gmm * l_gmm + lambda_align * (l_align + l_ortho) + lambda_reg * l_dim + lambda_div * l_div + lambda_repul * l_repul

            return total_loss, {
                "gmm": l_gmm.item(),
                "align": l_align.item(),
                "dim": l_dim.item(),
                "ortho": l_ortho.item(),
                "div": l_div.item(),
                "repul": l_repul.item(),
                # actual \mu predict loss for the winner
                "win_mu_loss": F.mse_loss(mu_win, x0_flat).item()
            }
