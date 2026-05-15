import torch

class BernsteinCore():
    def __init__(self, rdf_weights_instance, robot, device, link_names):
        self.device = device    
        self.robot = robot
        
        self.used_links = link_names
        self.K = len(self.used_links)
        
        # Setup batched parameters in RDF_Weights
        rdf_weights_instance.set_ordered_batch_params(link_names)
        
        # Extrait les paramètres du modèle généré par SDF_Bernstein_Basis
        self.offsets = rdf_weights_instance.centroids_batch.squeeze((1, 2)).to(device) # Shape [K, 3]
        self.scales = rdf_weights_instance.scale_factors_batch.squeeze().to(device) # Shape [K]
        self.weights = rdf_weights_instance.weights_batch.to(device) # Shape [K, N_func^3]
        self.n_func = rdf_weights_instance.max_n_func

        # Pré-calcul des coefficients binomiaux (ils ne changent jamais)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        self.comb = torch.exp(torch.lgamma(torch.tensor(n + 1.0, device=device)) - 
                              torch.lgamma(i + 1.0) - 
                              torch.lgamma(torch.tensor(n, device=device) - i + 1.0)).contiguous()

        self.used_links_tensor = torch.arange(self.K, dtype=torch.long, device=device)
        self.i_tensor = torch.arange(self.n_func, device=device)

    def build_bernstein_t(self, t):
        t = torch.clamp(t, min=1e-4, max=1-1e-4) # Limite dynamique
        n = self.n_func - 1
        
        phi = self.comb * (1 - t).unsqueeze(-1) ** (n - self.i_tensor) * t.unsqueeze(-1) ** self.i_tensor
        return phi.float()

    def build_basis_function_from_points(self, p):
        N = len(p)
        # Shift domain from [-1, 1] to [0, 1] 
        p = ((p - (-1.0)) / (1.0 - (-1.0))).reshape(-1)
        phi = self.build_bernstein_t(p) 
        phi = phi.reshape(N, 3, self.n_func)
        
        phi_x = phi[:,0,:]
        phi_y = phi[:,1,:]
        phi_z = phi[:,2,:]
        
        # Optimisation des multiplications de base (Même structure que l'ancien code)
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, self.n_func**3)
        
        return phi_xyz

    def get_whole_body_sdf_batch(self, x, pose, theta, return_per_link=False):
        """
        Calcule la distance SDF ultra-rapidement en batch vectorisé.

        Args:
            return_per_link: if True, also returns sdf [B, K, N] before the min over links.
                             Enables per-link barrier computation in BernsteinBarrier.
        """
        B = theta.size(0)
        N = x.size(0)
        K = self.K

        # 1. Forward Kinematics (FK) via URDFLayer
        trans_list = self.robot.get_transformations_each_link(pose, theta)

        # Select exactly the transforms for the requested links
        matched_trans = []
        for target_link in self.used_links:
            idx = None
            t_name = target_link.replace('panda_', '').replace('_w', '').replace('.pt', '')
            for i, info in enumerate(self.robot.meshes_info):
                i_name = info['link_name'].replace('panda_', '')
                if t_name in i_name or i_name in t_name:
                    idx = i
                    break

            if idx is None:
                raise ValueError(f"Link {target_link} not found in URDF visuals")
            matched_trans.append(trans_list[idx])

        trans_stacked = torch.stack(matched_trans, dim=1)

        fk_trans = trans_stacked.reshape(B*K, 4, 4)

        # 2. Transform world points to each link's local frame
        # torch.bmm(diff, R) computes R^T @ diff in column-vector notation,
        # which is the correct world→local transform for an orthogonal R.
        R = fk_trans[:, :3, :3].contiguous()
        t_vec = fk_trans[:, :3, 3].contiguous()

        diff = x.unsqueeze(0) - t_vec.unsqueeze(1)
        x_robot_frame_batch = torch.bmm(diff, R)

        # 3. Mise à l'échelle pour SDF_Bernstein_Basis
        offsets_expanded = self.offsets.unsqueeze(0).expand(B, K, 3).reshape(B*K, 1, 3)
        scales_expanded = self.scales.unsqueeze(0).expand(B, K).reshape(B*K, 1, 1)

        x_scaled = (x_robot_frame_batch - offsets_expanded) / scales_expanded

        # 4. Bornage au volume [-1, 1]
        # (Dans le nouveau système, c'est une approximation sphérique à la frontière,
        # mais le clamping est le plus rapide pour la collision de près)
        x_bounded = torch.clamp(x_scaled, min=-1.0+1e-2, max=1.0-1e-2)
        res_x = x_scaled - x_bounded

        # 5. Évaluation du polynôme de Bernstein
        phi = self.build_basis_function_from_points(x_bounded.reshape(B*K*N, 3))
        phi = phi.reshape(B, K, N, -1).transpose(0, 1).reshape(K, B*N, -1)

        # 6. Produit Scalaire des Poids
        sdf = torch.einsum('kni,ki->kn', phi, self.weights).reshape(K, B, N).transpose(0, 1).reshape(B*K, N)
        sdf = sdf + res_x.norm(dim=-1)
        sdf = sdf.reshape(B, K, N)

        # Rescale selon les dimensions d'origine
        sdf = sdf * self.scales.unsqueeze(0).unsqueeze(2)  # [B, K, N]

        sdf_value, _ = sdf.min(dim=1)  # [B, N] — min over links

        if return_per_link:
            return sdf_value, sdf  # [B, N], [B, K, N]

        return sdf_value
