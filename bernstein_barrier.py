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
        #    The standard logsumexp underestimates the true minimum by alpha * ln(N).
        #    We normalize it by subtracting ln(N) to center the soft-minimum on the true distance.
        N = sdf_per_link.shape[-1]
        h_per_link = -self.alpha * (torch.logsumexp(-sdf_per_link / self.alpha, dim=-1) - math.log(N)) - self.d_safe

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
