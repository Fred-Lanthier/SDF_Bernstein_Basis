import numpy as np
import torch

from src.core.math.Bernstain_P import BersteinPoly


class BernsteinWeightsTrain(BersteinPoly):
    def __init__(self, n_func=8, domain_min=-1, domain_max=1, device="cpu", dtype=torch.float32):
        super().__init__(n_func, domain_min, domain_max, device, dtype)

    def set_weights(self, wb: torch.Tensor):
        self.weights = wb.to(self.device).to(self.dtype)

    def _to_device_dtype(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=self.device, dtype=self.dtype)
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    def set_scale_and_offset(self, scale, translate):
        self.scale_factor = self._to_device_dtype(scale)
        self.centroid_offset = self._to_device_dtype(translate)

    def train(
        self,
        point_near: np.ndarray,
        near_sdf: np.ndarray,
        query_points: np.ndarray,
        query_sdf: np.ndarray,
        epoches: int = 400,
        sample_near: int = 1024,
        sample_rand: int = 64,
    ):
        point_near = np.asarray(point_near)
        near_sdf = np.asarray(near_sdf).reshape(-1)
        query_points = np.asarray(query_points)
        query_sdf = np.asarray(query_sdf).reshape(-1)

        wb = torch.zeros(self.n_func ** 3, dtype=self.dtype, device=self.device)
        B = (torch.eye(self.n_func ** 3, device=self.device, dtype=self.dtype) / 1e-2).clone()

        for iter_idx in range(int(epoches)):
            replace_near = len(point_near) < int(sample_near)
            replace_rand = len(query_points) < int(sample_rand)
            choice_near = np.random.choice(len(point_near), int(sample_near), replace=replace_near)
            choice_random = np.random.choice(len(query_points), int(sample_rand), replace=replace_rand)

            p_near = torch.from_numpy(point_near[choice_near]).to(self.device, dtype=self.dtype)
            sdf_near = torch.from_numpy(near_sdf[choice_near]).to(self.device, dtype=self.dtype)
            p_random = torch.from_numpy(query_points[choice_random]).to(self.device, dtype=self.dtype)
            sdf_random = torch.from_numpy(query_sdf[choice_random]).to(self.device, dtype=self.dtype)

            p = torch.cat([p_near, p_random], dim=0)
            sdf = torch.cat([sdf_near, sdf_random], dim=0)

            phi_xyz, _ = self.basis_function_from_3Dpoints(p.to(self.device, dtype=self.dtype), use_derivative=False)

            eye = torch.eye(len(p), device=self.device, dtype=self.dtype)
            K = torch.matmul(B, phi_xyz.T).matmul(torch.linalg.inv(eye + torch.matmul(torch.matmul(phi_xyz, B), phi_xyz.T)))
            B -= torch.matmul(K, phi_xyz).matmul(B)
            delta_wb = torch.matmul(K, (sdf - torch.matmul(phi_xyz, wb)).squeeze())
            loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz, wb).squeeze(), sdf, reduction="mean").item()

            print(f"\033[95m Iteration {iter_idx} loss {loss}\033[0m")
            wb += delta_wb

        self.weights = wb
        return wb
