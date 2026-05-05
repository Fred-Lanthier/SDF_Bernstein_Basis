import torch
import torch.nn as nn

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
        B = q.size(0)
        
        if pose is None:
            pose = torch.eye(4, device=q.device).unsqueeze(0).expand(B, 4, 4)

        if not q.requires_grad:
            q.requires_grad_(True)

        # 1. Distances pour tous les points : shape [Batch, N_points]
        sdf_distances = self.core.get_whole_body_sdf_batch(x_obs, pose, q)

        # 2. Smooth min using logsumexp (plus doux que torch.min)
        # min(x) \approx -alpha * log(sum(exp(-x/alpha)))
        h_smooth = -self.alpha * torch.logsumexp(-sdf_distances / self.alpha, dim=1)
        
        # On garde une trace de l'index min (même si non utilisé pour h) pour la signature
        # Si besoin de l'index réel pour la viz, on peut le calculer sans gradient
        with torch.no_grad():
            _, min_idx = torch.min(sdf_distances, dim=1)

        # 3. Calcul de la Barrière h(q)
        h = h_smooth - self.d_safe

        # 4. Le Gradient pyTorch (sait exactement quel "link" dériver via URDFLayer)
        grad_h = torch.autograd.grad(
            outputs=h,
            inputs=q,
            grad_outputs=torch.ones_like(h),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

        return h, grad_h, min_idx
