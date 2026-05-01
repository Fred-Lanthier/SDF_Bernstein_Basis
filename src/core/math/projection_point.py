import torch


def spherical_projection(points: torch.Tensor, r1: float = 1.0, center: torch.Tensor = None):
    """
    Proietta i punti fuori da una sfera di raggio r1.
    I punti interni restano invariati.
    """
    if center is None:
        center = torch.zeros((1, 3), device=points.device, dtype=points.dtype)
    else:
        center = center.reshape(1, 3).to(device=points.device, dtype=points.dtype)

    rel = points - center
    r = rel.norm(dim=1).clamp_min(1e-12)
    outside = r > r1

    scale = torch.where(outside, (r1 / r), torch.ones_like(r))
    proj = center + rel * scale.unsqueeze(1)
    dist = torch.where(outside, r - r1, torch.zeros_like(r))
    return proj, dist, outside


def correct_sphere_gradient(points_scaled: torch.Tensor, g_in: torch.Tensor, outside: torch.Tensor, r1: float):
    r = points_scaled.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u = points_scaled / r

    ug = (u * g_in).sum(dim=1, keepdim=True)
    Jtg = (r1 / r) * (g_in - u * ug)

    mask = outside.reshape(-1, 1)
    g_bernstein_z = torch.where(mask, Jtg, g_in)
    g_sphere_z = torch.where(mask, u, 0.0)
    return g_bernstein_z + g_sphere_z
