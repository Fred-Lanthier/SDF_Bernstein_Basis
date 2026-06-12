#!/usr/bin/env python3
"""Per-link gradient-quality diagnostic: how close is ||grad|| to 1 and how
accurate is the gradient direction on clean external points, for a single link
(no min-over-links effects)."""

import argparse, os, sys
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.abspath('../..')); sys.path.insert(0, os.path.abspath('.'))
import visualize_robot_sdf_layers as V


def diag(link, override, device, n=6000, seed=3):
    rl, w, core = V.build_sdf_stack(device, [link], model_override_dir=override)
    pose = torch.eye(4, device=device).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, device)
    meshes = V.build_ground_truth_meshes(w, rl, pose, q9, [link])
    m = meshes[0]

    pts, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fi]
    off = np.random.default_rng(seed).uniform(0.005, 0.03, size=(len(pts), 1))
    P = (pts + nrm * off).astype(np.float32)

    # exact gradient direction = outward normal of nearest face
    dist, gexact = V.ground_truth_distance_and_gradient(meshes, P)
    sdf, g = V.evaluate_sdf_and_grad(core, P, pose, q9, 32768, device)
    gn = np.linalg.norm(g, axis=1)
    gu = g / np.clip(gn[:, None], 1e-9, None)
    geu = gexact / np.clip(np.linalg.norm(gexact, axis=1, keepdims=True), 1e-9, None)
    ang = np.degrees(np.arccos(np.clip(np.sum(gu * geu, axis=1), -1, 1)))

    tag = "PROD " if override is None else "EIKON"
    print(f"{tag} {link}: ||grad|| mean={gn.mean():.3f} std={gn.std():.3f} "
          f"[{gn.min():.3f},{gn.max():.3f}] %<0.8={np.mean(gn<0.8)*100:.1f} %>1.2={np.mean(gn>1.2)*100:.1f} "
          f"| dir err mean={ang.mean():.1f} med={np.median(ang):.1f} p95={np.percentile(ang,95):.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", default="panda_link6")
    ap.add_argument("--override", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag(args.link, None, dev)               # production
    if args.override:
        diag(args.link, args.override, dev)  # experiment


if __name__ == "__main__":
    main()
