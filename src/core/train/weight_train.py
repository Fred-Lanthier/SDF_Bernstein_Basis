from src.core.math.Bernstain_P import BersteinPoly
import math
import torch
import numpy as np
import trimesh


def elevate_bernstein_weights(weights, old_n_func, new_n_func):
    """Degree-elevate tensor-product Bernstein weights without changing the SDF."""
    old_n_func = int(old_n_func)
    new_n_func = int(new_n_func)
    if new_n_func < old_n_func:
        raise ValueError("Bernstein degree elevation cannot reduce n_func.")
    if new_n_func == old_n_func:
        return torch.as_tensor(weights).reshape(-1).clone()

    old_degree = old_n_func - 1
    new_degree = new_n_func - 1
    degree_delta = new_degree - old_degree
    elevation = torch.zeros((new_n_func, old_n_func), dtype=torch.float64)
    for new_index in range(new_n_func):
        old_start = max(0, new_index - degree_delta)
        old_stop = min(old_degree, new_index)
        denominator = math.comb(new_degree, new_index)
        for old_index in range(old_start, old_stop + 1):
            elevation[new_index, old_index] = (
                math.comb(old_degree, old_index)
                * math.comb(degree_delta, new_index - old_index)
                / denominator
            )

    elevated = torch.as_tensor(weights, dtype=torch.float64).reshape(
        old_n_func,
        old_n_func,
        old_n_func,
    )
    elevated = torch.einsum("ia,abc->ibc", elevation, elevated)
    elevated = torch.einsum("jb,ibc->ijc", elevation, elevated)
    elevated = torch.einsum("kc,ijc->ijk", elevation, elevated)
    return elevated.reshape(-1).to(dtype=torch.float32)


def build_normal_ray_training_data(
    mesh,
    centroid_offset,
    scale_factor,
    number_of_points=50000,
    min_distance_m=0.001,
    focus_distance_m=0.01,
    focus_fraction=0.75,
    max_distance_m=0.05,
    normal_tolerance_degrees=5.0,
    seed=7,
):
    """Generate verified outward samples in the normalized model frame."""
    mesh_normalized = mesh.copy()
    mesh_normalized.vertices = (
        np.asarray(mesh_normalized.vertices) - np.asarray(centroid_offset)
    ) / float(scale_factor)

    rng = np.random.default_rng(seed)
    min_distance_normalized = float(min_distance_m) / float(scale_factor)
    focus_distance_normalized = float(focus_distance_m) / float(scale_factor)
    max_distance_normalized = float(max_distance_m) / float(scale_factor)
    accepted_points = []
    accepted_sdf = []
    accepted_directions = []

    for round_index in range(12):
        accepted_count = sum(len(points) for points in accepted_points)
        remaining = number_of_points - accepted_count
        if remaining <= 0:
            break

        candidate_count = max(remaining * 2, 2048)
        surface_points, face_indices = trimesh.sample.sample_surface(
            mesh_normalized,
            candidate_count,
            seed=seed + round_index,
        )
        surface_normals = mesh_normalized.face_normals[face_indices]

        # Concentrate samples in the safety-critical shell while retaining
        # enough farther samples to preserve the global distance profile.
        near_count = int(round(candidate_count * focus_fraction))
        offsets = np.empty(candidate_count, dtype=np.float64)
        offsets[:near_count] = rng.uniform(
            min_distance_normalized,
            focus_distance_normalized,
            size=near_count,
        )
        offsets[near_count:] = rng.uniform(
            focus_distance_normalized,
            max_distance_normalized,
            size=candidate_count - near_count,
        )
        rng.shuffle(offsets)

        candidates = surface_points + surface_normals * offsets[:, None]
        in_domain = np.all((candidates >= -1.0) & (candidates <= 1.0), axis=1)
        candidates = candidates[in_domain]
        candidate_normals = surface_normals[in_domain]
        candidate_offsets = offsets[in_domain]
        if len(candidates) == 0:
            continue

        closest, distances, triangle_ids = trimesh.proximity.closest_point(
            mesh_normalized,
            candidates,
        )
        displacement = candidates - closest
        nearest_normals = mesh_normalized.face_normals[triangle_ids]
        external = np.einsum("ij,ij->i", displacement, nearest_normals) >= 0.0
        valid_distance = distances > 1e-6
        directions = displacement / np.maximum(distances[:, None], 1e-12)
        normal_alignment = np.einsum("ij,ij->i", directions, candidate_normals)
        stable_normal = normal_alignment >= np.cos(
            np.deg2rad(normal_tolerance_degrees)
        )
        distance_matches_offset = np.abs(distances - candidate_offsets) <= (
            0.00025 / float(scale_factor)
        )
        valid = external & valid_distance & stable_normal & distance_matches_offset

        accepted_points.append(candidates[valid].astype(np.float32))
        accepted_sdf.append(distances[valid].astype(np.float32))
        accepted_directions.append(directions[valid].astype(np.float32))

    if not accepted_points:
        raise RuntimeError("Could not generate valid normal-ray training points.")

    points = np.concatenate(accepted_points, axis=0)
    sdf = np.concatenate(accepted_sdf, axis=0)
    directions = np.concatenate(accepted_directions, axis=0)
    if len(points) < number_of_points:
        raise RuntimeError(
            f"Generated only {len(points)} normal-ray points; requested {number_of_points}."
        )

    selection = rng.permutation(len(points))[:number_of_points]
    return points[selection], sdf[selection], directions[selection]


class BernsteinWeightsTrain(BersteinPoly):
    def __init__(self, n_func=8, domain_min = -1, domain_max=1, device='cpu', dtype=torch.float32):
        super().__init__(n_func, domain_min, domain_max, device, dtype)
    
    def set_weights(self, wb:torch.Tensor):
        self.weights = wb.to(self.device).to(self.dtype)
        # print(f'\033[95mWeights set with shape {self.wb.shape}\033[0m')

    def _to_device_dtype(self, x):
        if isinstance(x, torch.Tensor):
            # detach to avoid gradients leaking in; clone if you need your own storage
            return x.detach().to(device=self.device, dtype=self.dtype)
        else:
            # numpy array, list, scalar...
            return torch.tensor(x, device=self.device, dtype=self.dtype)

    def set_scale_and_offset(self, scale, translate):
        self.scale_factor   = self._to_device_dtype(scale)
        self.centroid_offset = self._to_device_dtype(translate)

        # print(f'\033[95mScale and translate set to scale {self.scale} and translate {self.translate}\033[0m')
    
    def train(self, point_near:np.ndarray, near_sdf:np.ndarray, query_points:np.ndarray, query_sdf:np.ndarray, epoches=400,
              sample_near=1024, sample_rand=64):
        # print(f'\033[95m Training Bernstain function using domain: [{self.domain_min},{self.domain_max}] for {epoches} epoches\033[0m')
        # print(f'Point near shape: {point_near.shape}, Near sdf shape: {near_sdf.shape}')
        # print(f'Query points shape: {query_points.shape}, Query sdf shape: {query_sdf.shape}')
        # query_sdf[query_sdf <-1] = -query_sdf[query_sdf <-1]

        wb = torch.zeros(self.n_func**3).float().to(self.device)
        B = (torch.eye(self.n_func**3)/1e-2).float().to(self.device)

        for iter in range(epoches):
            choice_near = np.random.choice(len(point_near), sample_near,replace=False) 
            p_near,sdf_near = torch.from_numpy(point_near[choice_near]).float().to(self.device),torch.from_numpy(near_sdf[choice_near]).float().to(self.device)
            choice_random = np.random.choice(len(query_points), sample_rand,replace=False)
            p_random,sdf_random = torch.from_numpy(query_points[choice_random]).float().to(self.device),torch.from_numpy(query_sdf[choice_random]).float().to(self.device)
            p = torch.cat([p_near,p_random],dim=0)
            sdf = torch.cat([sdf_near,sdf_random],dim=0)
            
            phi_xyz, _ = self.basis_function_from_3Dpoints(p.float().to(self.device),use_derivative=False)


            K = torch.matmul(B,phi_xyz.T).matmul(torch.linalg.inv((torch.eye(len(p)).float().to(self.device)+torch.matmul(torch.matmul(phi_xyz,B),phi_xyz.T))))
            B -= torch.matmul(K,phi_xyz).matmul(B)
            delta_wb = torch.matmul(K,(sdf - torch.matmul(phi_xyz,wb)).squeeze())
            loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz,wb).squeeze(), sdf, reduction='mean').item()

            print(f'\033[95m Iteration {iter} loss {loss}\033[0m')
            # loss_list.append(loss)
            wb += delta_wb   
            # print(f'iter {iter} wb {wb.shape}')

        return wb



    def train_eikonal(self, point_near: np.ndarray, near_sdf: np.ndarray,
            query_points: np.ndarray, query_sdf: np.ndarray,
            epoches=400, lr=1e-3, lambda_eikonal=0.05, lambda_bound=0.5, sample_near=8192, sample_rand=2048, batch_size=512):
        
        print(f'\033[95m Training Bernstein SDF with Eikonal + Boundary loss [{self.domain_min}, {self.domain_max}] for {epoches} epochs\033[0m')
        print(f' [95m Using N_FUNC={self.n_func} ({self.n_func**3} parameters) and sample sizes: near={sample_near}, rand={sample_rand} [0m')

        # Inizializza i pesi come parametro ottimizzabile
        wb = torch.zeros(self.n_func**3, requires_grad=True, device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam([wb], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)

        for iter in range(epoches):
            optimizer.zero_grad()

            # Campionamento
            idx_near = np.random.choice(len(point_near), sample_near, replace=False)
            idx_rand = np.random.choice(len(query_points), sample_rand, replace=False)

            p_near = torch.from_numpy(point_near[idx_near]).float().to(self.device)
            sdf_near = torch.from_numpy(near_sdf[idx_near]).float().to(self.device)

            p_rand = torch.from_numpy(query_points[idx_rand]).float().to(self.device)
            sdf_rand = torch.from_numpy(query_sdf[idx_rand]).float().to(self.device)

            p_all = torch.cat([p_near, p_rand], dim=0)
            sdf_gt_all = torch.cat([sdf_near, sdf_rand], dim=0)
            
            # Batch loop to avoid OOM
            total_loss_sdf = 0
            total_loss_eik = 0
            
            # We must accumulate gradients across batches
            for b_idx in range(0, p_all.shape[0], batch_size):
                p = p_all[b_idx : b_idx + batch_size]
                sdf_gt = sdf_gt_all[b_idx : b_idx + batch_size]

                # Base di Bernstein + derivate
                phi_xyz, d_phi_xyz = self.basis_function_from_3Dpoints(p, use_derivative=True)

                sdf_pred = torch.matmul(phi_xyz, wb)  # Predizione SDF

                # Gradiente analitico ∇SDF
                grad = torch.einsum('ijk,j->ik', d_phi_xyz, wb)
                grad_norm = grad.norm(dim=1)

                # Loss componenti with moderate surface weighting
                surface_weight = torch.exp(-4.0 * sdf_gt.abs()) + 0.05
                loss_sdf = (surface_weight * (sdf_pred.squeeze() - sdf_gt) ** 2).mean()

                # Enforce Eikonal loss everywhere to prevent swiss cheese oscillations
                loss_eikonal = ((grad_norm - 1) ** 2).mean()
                
                # Scaled loss for accumulation
                loss_part = (loss_sdf + lambda_eikonal * loss_eikonal) * (p.shape[0] / p_all.shape[0])
                loss_part.backward()
                
                total_loss_sdf += loss_sdf.item() * (p.shape[0] / p_all.shape[0])
                total_loss_eik += loss_eikonal.item() * (p.shape[0] / p_all.shape[0])
            
            # Boundary constraint (small batch, separate)
            p_bound = (torch.rand((128, 3), device=self.device) * 2 - 1)
            face_idx = torch.randint(0, 3, (128,), device=self.device)
            side = torch.randint(0, 2, (128,), device=self.device) * 2 - 1
            p_bound[torch.arange(128), face_idx] = side.float()
            
            phi_bound, _ = self.basis_function_from_3Dpoints(p_bound, use_derivative=False)
            sdf_bound = torch.matmul(phi_bound, wb)
            loss_boundary = torch.relu(0.1 - sdf_bound).mean() 

            (lambda_bound * loss_boundary).backward()

            # Update e scheduler
            optimizer.step()
            scheduler.step()

            if iter % 100 == 0 or iter == epoches - 1:
                print(f"\033[95mEpoch {iter:04d} | SDF: {total_loss_sdf:.5f} | Eik: {total_loss_eik:.5f} | Bnd: {loss_boundary.item():.5f} | Total: {total_loss_sdf + total_loss_eik + loss_boundary.item():.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}\033[0m")

        return wb.detach()

    def train_geometric(
        self,
        point_near,
        near_sdf,
        query_points,
        query_sdf,
        ray_points,
        ray_sdf,
        ray_directions,
        initial_weights=None,
        epochs=400,
        learning_rate=1e-3,
        sample_near=2048,
        sample_rand=1024,
        sample_ray=1024,
        lambda_ray=2.0,
        lambda_eikonal=0.05,
        lambda_direction=0.2,
        lambda_gradient_vector=0.25,
        ray_focus_distance=None,
        ray_focus_fraction=0.75,
        near_surface_weight=4.0,
        seed=7,
    ):
        """Fine-tune a low-order Bernstein SDF using value and direction data."""
        rng = np.random.default_rng(seed)
        if initial_weights is None:
            initial_weights = torch.zeros(self.n_func ** 3, dtype=torch.float32)
        weights = torch.nn.Parameter(
            torch.as_tensor(initial_weights, dtype=torch.float32, device=self.device)
            .reshape(-1)
            .clone()
        )
        optimizer = torch.optim.Adam([weights], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
        )

        arrays = [
            point_near,
            near_sdf,
            query_points,
            query_sdf,
            ray_points,
            ray_sdf,
            ray_directions,
        ]
        (
            point_near,
            near_sdf,
            query_points,
            query_sdf,
            ray_points,
            ray_sdf,
            ray_directions,
        ) = [np.asarray(value) for value in arrays]

        print(
            f"Training geometric Bernstein SDF: n_func={self.n_func}, epochs={epochs}, "
            f"near={sample_near}, random={sample_rand}, rays={sample_ray}"
        )
        if ray_focus_distance is None:
            ray_focus_distance = float(np.max(ray_sdf))
        focus_ray_indices = np.flatnonzero(ray_sdf <= ray_focus_distance)
        far_ray_indices = np.flatnonzero(ray_sdf > ray_focus_distance)
        if len(focus_ray_indices) == 0:
            raise ValueError("No ray samples fall inside ray_focus_distance.")

        for epoch in range(epochs):
            optimizer.zero_grad()
            near_index = rng.choice(
                len(point_near),
                size=min(sample_near, len(point_near)),
                replace=False,
            )
            rand_index = rng.choice(
                len(query_points),
                size=min(sample_rand, len(query_points)),
                replace=False,
            )
            ray_batch_size = min(sample_ray, len(ray_points))
            focus_count = min(
                int(round(ray_batch_size * ray_focus_fraction)),
                len(focus_ray_indices),
            )
            far_count = min(ray_batch_size - focus_count, len(far_ray_indices))
            focus_count = min(
                ray_batch_size - far_count,
                len(focus_ray_indices),
            )
            ray_index_parts = [
                rng.choice(focus_ray_indices, size=focus_count, replace=False)
            ]
            if far_count:
                ray_index_parts.append(
                    rng.choice(far_ray_indices, size=far_count, replace=False)
                )
            ray_index = np.concatenate(ray_index_parts)
            rng.shuffle(ray_index)

            p_near = torch.as_tensor(
                point_near[near_index], dtype=torch.float32, device=self.device
            )
            sdf_near = torch.as_tensor(
                near_sdf[near_index], dtype=torch.float32, device=self.device
            )
            p_rand = torch.as_tensor(
                query_points[rand_index], dtype=torch.float32, device=self.device
            )
            sdf_rand = torch.as_tensor(
                query_sdf[rand_index], dtype=torch.float32, device=self.device
            )
            p_ray = torch.as_tensor(
                ray_points[ray_index], dtype=torch.float32, device=self.device
            )
            sdf_ray = torch.as_tensor(
                ray_sdf[ray_index], dtype=torch.float32, device=self.device
            )
            direction_ray = torch.as_tensor(
                ray_directions[ray_index],
                dtype=torch.float32,
                device=self.device,
            )

            phi_near, grad_phi_near = self.basis_function_from_3Dpoints(
                p_near, use_derivative=True
            )
            phi_rand, grad_phi_rand = self.basis_function_from_3Dpoints(
                p_rand, use_derivative=True
            )
            phi_ray, grad_phi_ray = self.basis_function_from_3Dpoints(
                p_ray, use_derivative=True
            )

            pred_near = phi_near @ weights
            pred_rand = phi_rand @ weights
            pred_ray = phi_ray @ weights
            grad_near = torch.einsum("nwc,w->nc", grad_phi_near, weights)
            grad_rand = torch.einsum("nwc,w->nc", grad_phi_rand, weights)
            grad_ray = torch.einsum("nwc,w->nc", grad_phi_ray, weights)

            loss_near = torch.nn.functional.mse_loss(pred_near, sdf_near)
            loss_rand = torch.nn.functional.mse_loss(pred_rand, sdf_rand)
            loss_ray = torch.nn.functional.mse_loss(pred_ray, sdf_ray)

            all_gradient = torch.cat([grad_near, grad_rand, grad_ray], dim=0)
            loss_eikonal = ((all_gradient.norm(dim=1) - 1.0) ** 2).mean()
            ray_unit = torch.nn.functional.normalize(grad_ray, dim=1, eps=1e-8)
            target_unit = torch.nn.functional.normalize(direction_ray, dim=1, eps=1e-8)
            ray_focus_scale = max(float(ray_focus_distance), 1e-8)
            ray_weights = 1.0 + near_surface_weight * torch.exp(
                -sdf_ray / ray_focus_scale
            )
            ray_weights = ray_weights / ray_weights.mean()
            direction_error = 1.0 - (ray_unit * target_unit).sum(dim=1)
            loss_direction = (ray_weights * direction_error).mean()
            gradient_vector_error = ((grad_ray - target_unit) ** 2).sum(dim=1)
            loss_gradient_vector = (
                ray_weights * gradient_vector_error
            ).mean()

            loss = (
                loss_near
                + loss_rand
                + lambda_ray * loss_ray
                + lambda_eikonal * loss_eikonal
                + lambda_direction * loss_direction
                + lambda_gradient_vector * loss_gradient_vector
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 25 == 0 or epoch == epochs - 1:
                print(
                    "Epoch {:04d} | total={:.6f} near={:.6f} random={:.6f} "
                    "ray={:.6f} eikonal={:.6f} direction={:.6f} "
                    "gradient_vector={:.6f}".format(
                        epoch,
                        loss.item(),
                        loss_near.item(),
                        loss_rand.item(),
                        loss_ray.item(),
                        loss_eikonal.item(),
                        loss_direction.item(),
                        loss_gradient_vector.item(),
                    )
                )

        return weights.detach()

    # def train_recursive_with_eikonal(self, point_near: np.ndarray, near_sdf: np.ndarray,
    #                                 query_points: np.ndarray, query_sdf: np.ndarray,
    #                                 epoches=400, lambda_eikonal=0.01):

    #     print(f"\033[95mRecursive Non-Linear SDF Training with Eikonal | Epochs: {epoches} | λ={lambda_eikonal}\033[0m")

    #     wb = torch.zeros(self.n_func**3).float().to(self.device)
    #     B = (torch.eye(self.n_func**3) / 1e-4).float().to(self.device)  # inizializza B come matrice regolarizzata

    #     for epoch in range(epoches):
    #         idx_near = np.random.choice(len(point_near), 1024, replace=False)
    #         idx_rand = np.random.choice(len(query_points), 64, replace=False)

    #         p_near = torch.from_numpy(point_near[idx_near]).float().to(self.device)
    #         sdf_near = torch.from_numpy(near_sdf[idx_near]).float().to(self.device)

    #         p_rand = torch.from_numpy(query_points[idx_rand]).float().to(self.device)
    #         sdf_rand = torch.from_numpy(query_sdf[idx_rand]).float().to(self.device)

    #         p = torch.cat([p_near, p_rand], dim=0)
    #         sdf_gt = torch.cat([sdf_near, sdf_rand], dim=0)

    #         phi_xyz, d_phi_xyz = self.basis_function_from_3Dpoints(p, use_derivative=True)

    #         # Parte SDF classica
    #         residual_sdf = sdf_gt - torch.matmul(phi_xyz, wb)
    #         K_sdf = B @ phi_xyz.T @ torch.linalg.inv(torch.eye(len(p)).to(self.device) + phi_xyz @ B @ phi_xyz.T)
    #         B = B - K_sdf @ phi_xyz @ B
    #         delta_wb_sdf = K_sdf @ residual_sdf

    #         # Parte Eikonal: ||∇SDF|| ≈ 1
    #         grad = torch.einsum('ijk,j->ik', d_phi_xyz, wb)  # (N, 3)
    #         grad_norm = grad.norm(dim=1)
    #         residual_grad = (1.0 - grad_norm).detach()  # solo direzione di correzione, no grad

    #         # Costruiamo "pseudo-osservazioni" sulla norma del gradiente
    #         # Vettori direzionali: ∇SDF / ∥∇SDF∥
    #         unit_grad = grad / (grad_norm.unsqueeze(1) + 1e-8)  # (N, 3)
    #         phi_grad = torch.einsum('ijk,ik->ij', d_phi_xyz, unit_grad)

    #         K_eik = B @ phi_grad.T @ torch.linalg.inv(torch.eye(len(p)).to(self.device) + phi_grad @ B @ phi_grad.T)
    #         B = B - lambda_eikonal * K_eik @ phi_grad @ B
    #         delta_wb_eik = lambda_eikonal * K_eik @ residual_grad

    #         # Aggiorna pesi
    #         wb += delta_wb_sdf + delta_wb_eik

    #         # Log
    #         loss_sdf = residual_sdf.pow(2).mean().item()
    #         loss_eikonal = residual_grad.pow(2).mean().item()
    #         print(f"\033[95mEpoch {epoch:03d} | SDF Loss: {loss_sdf:.5f} | Eikonal: {loss_eikonal:.5f}\033[0m")

    #     return wb
