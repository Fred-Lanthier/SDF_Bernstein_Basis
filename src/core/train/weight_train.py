from src.core.math.Bernstain_P import BersteinPoly
import torch
import numpy as np


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
