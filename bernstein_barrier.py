import torch
import torch.nn as nn
import math

class BernsteinBarrier(nn.Module):
    """
    Module PyTorch agissant comme une Control Barrier Function (CBF).
    Il encapsule le moteur BernsteinCore pour fournir la distance ET le gradient exact.
    """
    def __init__(self, bernstein_core, d_safe=0.05, alpha=0.01):
        super().__init__()
        self.core = bernstein_core
        self.d_safe = d_safe  # Marge de sécurité
        self.alpha = alpha

    def forward(self, q, x_obs, pose=None):
        """
        Per-link barrier formulation.

        Instead of: softmin_N(min_K(sdf))  — blends gradient across all links
        We compute:  min_K(softmin_N(sdf_k)) — gradient flows to the most dangerous link

        This ensures that when the fork tip is near obstacle A and panda_hand is near
        obstacle B, the correction targets whichever link is truly most endangered,
        not a weighted compromise of both directions.
        """
        B = q.size(0)

        if pose is None:
            pose = torch.eye(4, device=q.device).unsqueeze(0).expand(B, 4, 4)

        if not q.requires_grad:
            q.requires_grad_(True)

        # 1. Per-link SDF: [B, K, N]
        _, sdf_per_link = self.core.get_whole_body_sdf_batch(x_obs, pose, q, return_per_link=True)

        # 2. Per-link softmin over N obstacle points → h_k for each link [B, K]
        #    We combine Log-Average-Exp with the Min-Trick over the negative exponents
        #    for numerical stability and scale/dimensional invariance.
        sdf_min, _ = sdf_per_link.min(dim=-1, keepdim=True)  # [B, K, 1]
        N = sdf_per_link.shape[-1]
        shifted_sdf = sdf_per_link - sdf_min  # [B, K, N]
        exp_terms = torch.exp(-shifted_sdf / self.alpha)  # [B, K, N]
        h_per_link = sdf_min.squeeze(-1) - self.alpha * torch.log(exp_terms.sum(dim=-1)) - self.d_safe

        # 3. Global h: hard min over links — autograd flows to the most dangerous link only
        h, _ = h_per_link.min(dim=1)  # [B]

        # Closest obstacle point globally — flatten K×N, argmin stays on GPU (no CPU sync)
        with torch.no_grad():
            N = sdf_per_link.shape[2]
            flat_min = sdf_per_link[0].reshape(-1).argmin()  # index in [0, K*N)
            min_idx  = (flat_min % N).view(1)                # index in [0, N)

        # 4. Gradient — autograd differentiates through min_K then through the winning
        #    link's softmin_N, giving the correct per-link direction.
        grad_h = torch.autograd.grad(
            outputs=h,
            inputs=q,
            grad_outputs=torch.ones_like(h),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

        return h, grad_h, min_idx
