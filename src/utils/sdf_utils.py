import numpy as np
import skimage.measure
import torch
import trimesh
from scipy.ndimage import gaussian_filter

from src.utils.MeshUtils import denormalize_mesh, normalize_mesh


def sdf_to_mesh(
    weights: torch.Tensor,
    nbData: int,
    domain_min: float,
    domain_max: float,
    scaling_factor: torch.Tensor,
    centroid_offset: torch.Tensor,
    basis_function_from_3Dpoints=None,
):
    """
    Genera la mesh isosuperficie (livello 0) da un campo SDF espresso tramite pesi e funzioni base.
    Tutto rimane sullo stesso device di `weights`.
    """
    device = weights.device
    dtype = weights.dtype

    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain, indexing="ij")
    p = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    d_list = []
    for p_s in torch.split(p, 10_000, dim=0):
        phi_p, _ = basis_function_from_3Dpoints(p_s, use_derivative=False)
        d_s = torch.matmul(phi_p, weights)
        d_list.append(d_s)

    d = torch.cat(d_list, dim=0)
    d_np = d.view(nbData, nbData, nbData).detach().cpu().numpy()
    d_smooth = gaussian_filter(d_np, sigma=1.0)

    spacing = (domain_max - domain_min) / nbData
    verts, faces, normals, values = skimage.measure.marching_cubes(
        d_smooth, level=0.0, spacing=(spacing, spacing, spacing)
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = mesh.subdivide_to_size(max_edge=1)

    scaling_factor_np = scaling_factor.detach().cpu().numpy() if torch.is_tensor(scaling_factor) else np.asarray(scaling_factor)
    centroid_offset_np = centroid_offset.detach().cpu().numpy() if torch.is_tensor(centroid_offset) else np.asarray(centroid_offset)

    mesh = normalize_mesh(mesh.copy())
    mesh = denormalize_mesh(mesh.copy(), scaling_factor_np, centroid_offset_np)
    return mesh
