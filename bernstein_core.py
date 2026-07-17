import torch

class BernsteinCore():
    def __init__(self, rdf_weights_instance, robot, device, link_names):
        self.device = device    
        self.robot = robot
        
        self.used_links = link_names
        self.K = len(self.used_links)
        self._link_mesh_infos = [
            self._resolve_link_mesh_info(link_name) for link_name in self.used_links
        ]
        
        # Setup batched parameters in RDF_Weights. These legacy tensors are kept for
        # compatibility, but runtime SDF evaluation uses per-order groups below so
        # low-order robot links are not padded to the fork_tip order.
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

        # ── Grasped object (rigid point cloud attached to a link frame) ────────
        # When grasp_attach_link names a link present in the FK dict, the grasped
        # object's point cloud (snapshot in that link's frame) is appended as an
        # extra protected "link" in get_whole_body_sdf_batch: its SDF is the
        # distance to the nearest grasped point minus grasp_radius (a union of
        # spheres conforming to the real target shape — no box/orientation fit).
        # This makes a grasped object part of the protected body AND the
        # self-filter. grasp_active in {0,1} gates it without changing the
        # (CUDA-graph) tensor topology; grasp_points is a fixed [M,3] buffer.
        self.grasp_attach_link = None
        self.grasp_npts = 0
        self.grasp_points = None                          # [M, 3] attach-link frame
        self.grasp_active = torch.zeros((), device=device)
        self.grasp_radius = torch.zeros((), device=device)
        # Soft-min temperature over grasped points. >0 blends the nearest points
        # so the SDF gradient is smooth (no jump when the nearest point switches);
        # 0 = hard min (C0, jumpy gradient). Baked at CUDA-graph capture.
        self.grasp_softmin_beta = 0.0
        # Per-link d_safe adjustment for the grasped object. The barrier subtracts
        # one global d_safe from every link; this offset (= grasp_d_safe - d_safe)
        # is folded into the grasp SDF so the grasped object's effective standoff
        # is grasp_d_safe while the arm/fork keep the global d_safe.
        self.grasp_dsafe_offset = 0.0

        self.groups = {}
        for idx, link in enumerate(link_names):
            model = getattr(rdf_weights_instance, link + rdf_weights_instance.model_extension)
            n_func = int(model.n_func)
            if n_func not in self.groups:
                self.groups[n_func] = {"indices": [], "link_names": []}
            self.groups[n_func]["indices"].append(idx)
            self.groups[n_func]["link_names"].append(link)

        for n_func, group in self.groups.items():
            group["n_func"] = n_func
            group["indices_tensor"] = torch.tensor(
                group["indices"], dtype=torch.long, device=device)
            group["offsets"] = torch.cat([
                getattr(rdf_weights_instance, link + rdf_weights_instance.model_extension)
                .centroid_offset.unsqueeze(0)
                for link in group["link_names"]
            ], dim=0).to(device)
            group["scales"] = torch.cat([
                getattr(rdf_weights_instance, link + rdf_weights_instance.model_extension)
                .scale_factor.unsqueeze(0)
                for link in group["link_names"]
            ], dim=0).reshape(-1).to(device)
            group["weights"] = torch.cat([
                getattr(rdf_weights_instance, link + rdf_weights_instance.model_extension)
                .weights.unsqueeze(0).to(device)
                for link in group["link_names"]
            ], dim=0)

            n = n_func - 1
            i = torch.arange(n_func, device=device)
            group["comb"] = torch.exp(
                torch.lgamma(torch.tensor(n + 1.0, device=device))
                - torch.lgamma(i + 1.0)
                - torch.lgamma(torch.tensor(n, device=device) - i + 1.0)
            ).contiguous()
            group["i_tensor"] = i

    def configure_grasp_cloud(self, attach_link, max_points, radius, softmin_beta=0.0,
                              d_safe_global=0.0, grasp_d_safe=None):
        """Allocate the fixed-size grasped-cloud buffer (call BEFORE CUDA-graph
        capture so the grasp term is baked into the graph). grasp_d_safe (None =
        use the global d_safe) sets the grasped object's own clearance margin."""
        self.grasp_attach_link = attach_link
        self.grasp_npts = int(max_points)
        self.grasp_points = torch.zeros((self.grasp_npts, 3), device=self.device)
        self.grasp_radius.fill_(float(radius))
        self.grasp_softmin_beta = float(softmin_beta)
        if grasp_d_safe is None:
            grasp_d_safe = d_safe_global
        self.grasp_dsafe_offset = float(grasp_d_safe) - float(d_safe_global)
        self.grasp_active.zero_()

    @staticmethod
    def _link_match_key(name):
        return name.replace('panda_', '').replace('_w', '').replace('.pt', '')

    def _resolve_link_mesh_info(self, target_link):
        target_key = self._link_match_key(target_link)
        for info in self.robot.meshes_info:
            mesh_key = info['link_name'].replace('panda_', '')
            if target_key in mesh_key or mesh_key in target_key:
                return info
        raise ValueError(f"Link {target_link} not found in URDF visuals")

    def _stack_used_link_transforms(self, pose, theta, link_poses=None):
        if theta.shape[-1] < self.robot.dof:
            batch_shape = theta.shape[:-1]
            padding = theta.new_zeros((*batch_shape, self.robot.dof - theta.shape[-1]))
            theta = torch.cat([theta, padding], dim=-1)

        if link_poses is None:
            link_poses = self.robot._native_forward_kinematics(theta)

        transforms = []
        for info in self._link_mesh_infos:
            link_name = info['link_name']
            if link_name not in link_poses:
                raise ValueError(f"Link {link_name} not mapped.")
            T_joint = link_poses[link_name]
            T_world_joint = pose @ T_joint
            batch_shape = T_world_joint.shape[:-2]
            T_visual = info['visual_offset'].to(dtype=T_world_joint.dtype).expand(*batch_shape, 4, 4)
            transforms.append(T_world_joint @ T_visual)

        return torch.stack(transforms, dim=1)

    def build_bernstein_t(self, t, n_func=None, comb=None, i_tensor=None):
        if n_func is None:
            n_func = self.n_func
            comb = self.comb
            i_tensor = self.i_tensor
        t = torch.clamp(t, min=1e-4, max=1-1e-4) # Limite dynamique
        n = n_func - 1
        
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i_tensor) * t.unsqueeze(-1) ** i_tensor
        return phi.float()

    def build_basis_function_from_points(self, p, n_func=None, comb=None, i_tensor=None):
        if n_func is None:
            n_func = self.n_func
            comb = self.comb
            i_tensor = self.i_tensor
        N = len(p)
        # Shift domain from [-1, 1] to [0, 1] 
        p = ((p - (-1.0)) / (1.0 - (-1.0))).reshape(-1)
        phi = self.build_bernstein_t(p, n_func, comb, i_tensor)
        phi = phi.reshape(N, 3, n_func)
        
        phi_x = phi[:,0,:]
        phi_y = phi[:,1,:]
        phi_z = phi[:,2,:]
        
        # Optimisation des multiplications de base (Même structure que l'ancien code)
        phi_xy = torch.einsum("ij,ik->ijk", phi_x, phi_y).view(-1, n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk", phi_xy, phi_z).view(-1, n_func**3)
        
        return phi_xyz

    def _grasp_cloud_sdf(self, x, T_wf):
        """Signed distance from world points x [N,3] to the grasped point cloud
        (a union of spheres of radius grasp_radius), batched over T_wf [B,4,4].
        Returns [B, N]. Differentiable w.r.t. T_wf (hence theta).

        grasp_points are in the attach-link frame; world = R @ p_local + t, the
        inverse of the world->local convention (bmm(diff, R)) used elsewhere.
        """
        R = T_wf[:, :3, :3]                                   # [B, 3, 3]
        t = T_wf[:, :3, 3]                                    # [B, 3]
        pts_w = torch.matmul(self.grasp_points.unsqueeze(0),  # [B, M, 3]
                             R.transpose(1, 2)) + t.unsqueeze(1)
        diff = x.unsqueeze(0).unsqueeze(0) - pts_w.unsqueeze(2)   # [B, M, N, 3]
        d = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)        # [B, M, N]
        beta = self.grasp_softmin_beta
        if beta and beta > 0.0:
            # Soft-min over the M grasped points (numerically stable LogSumExp
            # with the min-shift trick) → smooth gradient, no switch-point jumps.
            d_min, _ = d.min(dim=1, keepdim=True)                # [B, 1, N]
            dist = d_min.squeeze(1) - beta * torch.log(
                torch.exp(-(d - d_min) / beta).sum(dim=1))       # [B, N]
        else:
            dist = d.min(dim=1).values                           # [B, N] hard min
        # Subtract grasp_dsafe_offset (= grasp_d_safe - global d_safe) so that,
        # after the barrier subtracts the global d_safe, the grasped object's
        # net margin equals grasp_d_safe.
        return dist - self.grasp_radius - self.grasp_dsafe_offset  # [B, N]

    def get_whole_body_sdf_batch(self, x, pose, theta, return_per_link=False, link_poses=None):
        """
        Calcule la distance SDF ultra-rapidement en batch vectorisé.

        Args:
            return_per_link: if True, also returns sdf [B, K, N] before the min over links.
                             Enables per-link barrier computation in BernsteinBarrier.
            link_poses: optional FK dict from URDFLayer._native_forward_kinematics(theta).
                        Passing it avoids recomputing FK when the caller already
                        needed link poses for candidate filtering.
        """
        B = theta.size(0)
        N = x.size(0)
        K = self.K

        # 1. Forward Kinematics (FK) for exactly the protected SDF links.
        #    Compute link_poses once here (if not provided) so the grasped-box
        #    term below can attach to its link frame with grad flowing through FK.
        if link_poses is None:
            theta_fk = theta
            if theta_fk.shape[-1] < self.robot.dof:
                pad = theta_fk.new_zeros(
                    (*theta_fk.shape[:-1], self.robot.dof - theta_fk.shape[-1]))
                theta_fk = torch.cat([theta_fk, pad], dim=-1)
            link_poses = self.robot._native_forward_kinematics(theta_fk)
        trans_stacked = self._stack_used_link_transforms(pose, theta, link_poses=link_poses)

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

        x_bounded_4d = x_bounded.reshape(B, K, N, 3)
        res_x_4d = res_x.reshape(B, K, N, 3)
        sdf_parts = []

        for _, group in self.groups.items():
            indices_tensor = group["indices_tensor"]
            group_indices = group["indices"]
            K_g = len(group_indices)
            n_func = group["n_func"]

            x_bounded_g = torch.index_select(x_bounded_4d, 1, indices_tensor)
            x_bounded_flat = x_bounded_g.transpose(0, 1).reshape(K_g, B * N, 3)
            phi = self.build_basis_function_from_points(
                x_bounded_flat.reshape(K_g * B * N, 3),
                n_func=n_func,
                comb=group["comb"],
                i_tensor=group["i_tensor"],
            )
            phi = phi.reshape(K_g, B * N, -1)

            sdf_g = torch.einsum('kni,ki->kn', phi, group["weights"])
            sdf_g = sdf_g.reshape(K_g, B, N).transpose(0, 1)
            # Out-of-domain extension: guaranteed LOWER bound on the true
            # distance, sqrt(P^2 + ||res||^2) for P >= 0. Proof: every link
            # surface point s lies in the clamp cube, and on each clamped
            # axis |p_j - s_j| = |c_j - s_j| + |res_j| (same side), hence
            # ||p - s||^2 >= ||c - s||^2 + ||res||^2 for all s, i.e.
            # d_true(p)^2 >= P^2 + ||res||^2. Unlike the Lipschitz bound
            # P - ||res||, this bound is MONOTONE INCREASING in the
            # excursion, so its gradient keeps the physical outward
            # direction (required by the CBF / null-space clearance). The
            # legacy P + ||res|| was an UPPER bound (could over-report
            # clearance). P < 0 at the clamped point (link crossing the
            # clamp-margin shell) falls back to P + ||res|| to preserve the
            # penetration sign; interior points keep the raw polynomial.
            res_norm = torch.index_select(res_x_4d, 1, indices_tensor).norm(dim=-1)
            outside_val = torch.where(
                sdf_g >= 0.0,
                torch.sqrt(sdf_g.clamp_min(0.0).square() + res_norm.square()),
                sdf_g + res_norm,
            )
            sdf_g = torch.where(res_norm > 0.0, outside_val, sdf_g)
            sdf_g = sdf_g * group["scales"].unsqueeze(0).unsqueeze(2)

            for group_i, original_i in enumerate(group_indices):
                sdf_parts.append((original_i, sdf_g[:, group_i, :]))

        sdf_parts.sort(key=lambda item: item[0])
        sdf = torch.stack([item[1] for item in sdf_parts], dim=1)

        # 5. Optional grasped-object point cloud, appended as an extra protected
        #    "link". Attached rigidly to grasp_attach_link; gated by grasp_active
        #    so it has zero effect (huge SDF, never the min) when not grasped.
        if (self.grasp_attach_link is not None
                and self.grasp_points is not None
                and self.grasp_attach_link in link_poses):
            T_wf = pose @ link_poses[self.grasp_attach_link]      # [B, 4, 4]
            grasp_sdf = self._grasp_cloud_sdf(x, T_wf)            # [B, N]
            grasp_sdf = grasp_sdf + (1.0 - self.grasp_active) * 1.0e3
            sdf = torch.cat([sdf, grasp_sdf.unsqueeze(1)], dim=1)  # [B, K+1, N]

        sdf_value, _ = sdf.min(dim=1)  # [B, N] — min over links

        if return_per_link:
            return sdf_value, sdf  # [B, N], [B, K, N]

        return sdf_value
