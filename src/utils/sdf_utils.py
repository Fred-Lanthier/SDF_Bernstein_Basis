import torch
import skimage.measure
import trimesh
import numpy as np
from src.core.math.transformations import matrix_to_pos_quat
from src.utils.MeshUtils import normalize_mesh, denormalize_mesh
from src.core.assets.entities.MeshModel import Mesh
from typing import Union
from scipy.ndimage import gaussian_filter



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
    # --- device coerente ---
    device = weights.device
    dtype = weights.dtype

    # --- griglia regolare nello stesso device/dtype ---
    domain = torch.linspace(domain_min, domain_max, nbData, device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain, indexing="ij")
    p = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    # --- calcolo SDF in batch per evitare OOM ---
    d_list = []
    for p_s in torch.split(p, 10_000, dim=0):
        phi_p, _ = basis_function_from_3Dpoints(p_s, use_derivative=False)
        d_s = torch.matmul(phi_p, weights)
        d_list.append(d_s)

    d = torch.cat(d_list, dim=0)
    d_np = d.view(nbData, nbData, nbData).detach().cpu().numpy()

    # --- marching cubes su CPU ---
    d_smooth = gaussian_filter(d_np, sigma=1.0)
    spacing = (domain_max - domain_min) / nbData
    verts, faces, normals, values = skimage.measure.marching_cubes(    d_smooth, level=0.0, spacing=(spacing, spacing, spacing))

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = mesh.subdivide_to_size(max_edge=1)

    # --- conversione sicura CPU per i parametri ---
    scaling_factor_np = scaling_factor.detach().cpu().numpy() if torch.is_tensor(scaling_factor) else np.asarray(scaling_factor)
    centroid_offset_np = centroid_offset.detach().cpu().numpy() if torch.is_tensor(centroid_offset) else np.asarray(centroid_offset)

    # --- normalizzazione e denormalizzazione ---
    mesh = normalize_mesh(mesh.copy())
    mesh = denormalize_mesh(mesh.copy(), scaling_factor_np, centroid_offset_np)

    return mesh





def rdf_as_mesh(sdf_mesh, tf):
    mesh = mesh_transfrom(sdf_mesh, tf)
    return mesh

def mesh_transfrom(mesh : Union[trimesh.Trimesh, Mesh], tf) -> trimesh.Trimesh:
    if isinstance(mesh, Mesh):
        mesh = mesh.mesh

    translation, quaternion = matrix_to_pos_quat(tf)
    mesh.apply_transform(trimesh.transformations.quaternion_matrix(quaternion.cpu().numpy()))
    mesh.apply_translation(translation)
    return mesh

def point_cloud_transform(points:torch.Tensor, tf:np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        if points.is_cuda:
            points = points.cpu()
        points = points.numpy()
    
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = points_homogeneous @ tf.T
    return transformed_points[:, :3]


def sample_nearby_points(points, sdf, central_point, radius, n_samples, device):
    def get_nearby_points(points, query_point, radius):
        distances = np.linalg.norm(points - query_point, axis=1)
        nearby_idx = np.where(distances < radius)[0]
        return nearby_idx

    nearby_indices = get_nearby_points(points, central_point, radius)
    
    # Assicurati di avere sempre n_samples punti campionati
    
    choice_indices = np.random.choice(nearby_indices, n_samples, replace=True)  # Campiona con replacement


    # Estrai i punti vicini e i loro sdf
    sampled_points = torch.from_numpy(points[choice_indices]).float().to(device)
    sampled_sdf = torch.from_numpy(sdf[choice_indices]).float().to(device)
    
    return sampled_points, sampled_sdf

def check_ellipsoid_fit(points, center, axes, scales):
    print("Verifica ortogonalità matrice assi (axes):")
    ortonorm = np.allclose(axes.T @ axes, np.eye(3), atol=1e-7)
    print(f"axes.T @ axes ≈ I? {ortonorm}")

    # Calcolo punti locali (coordinate nel sistema degli assi)
    points_centered = points - center
    points_local = points_centered @ axes
    print(f"Shape points_local: {points_local.shape}")

    # Controlla max valori assoluti nei punti locali e semiassi
    max_local = np.max(np.abs(points_local), axis=0)
    print(f"Max valori assoluti nei punti locali: {max_local}")
    print(f"Semiassi forniti (scales): {scales}")

    # Check che i semiassi siano almeno i max valori assoluti (dovrebbero essere uguali o più grandi)
    for i, (m, s) in enumerate(zip(max_local, scales)):
        if s < m:
            print(f"Attenzione: semi-asse {i} ({s}) è minore del massimo valore locale {m}!")

    # Verifica formula ellissoide
    val = (points_local[:, 0] / scales[0])**2 + (points_local[:, 1] / scales[1])**2 + (points_local[:, 2] / scales[2])**2
    all_inside = np.all(val <= 1 + 1e-12)
    n_outside = np.sum(val > 1 + 1e-12)

    print(f"Tutti i punti sono dentro l'ellissoide? {all_inside}")
    if not all_inside:
        print(f"Punti fuori: {n_outside} su {len(points)}")

    # Mostra alcuni valori anomali
    if n_outside > 0:
        idx_outside = np.where(val > 1 + 1e-12)[0]
        print("Valori ellissoide dei primi 5 punti fuori:", val[idx_outside[:5]])
        print("Coordinate locali dei primi 5 punti fuori:", points_local[idx_outside[:5]])

    return all_inside, val
