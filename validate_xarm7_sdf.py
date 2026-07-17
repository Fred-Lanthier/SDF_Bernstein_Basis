#!/usr/bin/env python3
"""Validate xArm7 Bernstein link SDFs against the true mesh.

For each link, evaluate the model SDF at the dataset query points, convert to
meters (sdf_norm * scale_factor), and compare to the ground-truth unsigned
distance (KD-tree on densely sampled mesh surface) for clearly-exterior points.
A correct model has mean/|err| in the millimetre range; the converted models
show a large constant over-estimate (robot looks bigger than it is).

Usage:
  python3 validate_xarm7_sdf.py --model-dir panda_test/Models              # production
  python3 validate_xarm7_sdf.py --model-dir panda_test/Models/nfunc12_candidate
"""
import argparse, os
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree
from src.core.train.weight_train import BernsteinWeightsTrain

ROOT = os.path.dirname(os.path.abspath(__file__))
PT = os.path.join(ROOT, "panda_test")
LINKS = [f"xarm7_link{i}" for i in [ "_base",1,2,3,4,5,6,7]]
LINKS = ["xarm7_link_base"] + [f"xarm7_link{i}" for i in range(1, 8)]


def eval_model_sdf(model, pts_norm, device):
    tr = BernsteinWeightsTrain(n_func=int(model["n_func"]),
                               domain_min=float(model["domain_min"]),
                               domain_max=float(model["domain_max"]),
                               device=device, dtype=torch.float32)
    tr.set_weights(model["weights"].to(device).float())
    with torch.no_grad():
        phi, _ = tr.basis_function_from_3Dpoints(
            torch.from_numpy(pts_norm).float().to(device), use_derivative=False)
        sdf = (phi @ tr.weights).squeeze().cpu().numpy()
    return sdf  # normalized units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(PT, "Models"))
    ap.add_argument("--n-samples", type=int, default=6000)
    ap.add_argument("--surf-samples", type=int, default=200000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    print(f"Validating models in: {args.model_dir}\n")
    print(f"{'link':16s} {'mean_err(m)':>12s} {'|err|(m)':>10s} {'corr':>6s} {'p95|err|(m)':>12s}")
    worst = 0.0
    for link in LINKS:
        mp = os.path.join(args.model_dir, f"{link}_w.pt")
        if not os.path.isfile(mp):
            print(f"{link:16s}   (missing)"); continue
        d = np.load(os.path.join(PT, "Dataset", f"{link}.npy"), allow_pickle=True).item()
        mesh = trimesh.load(os.path.join(PT, "Meshes", f"{link}.stl"), force="mesh")
        model = torch.load(mp, map_location="cpu", weights_only=False)
        sf = float(d["mesh_scale_factor"]); c = np.asarray(d["mesh_centroid_offset"]).reshape(3)
        surf, _ = trimesh.sample.sample_surface(mesh, args.surf_samples)
        tree = cKDTree(surf)
        qn = d["query_points"]; qsdf = d["query_sdf"].astype(np.float64)
        idx = rng.choice(len(qn), args.n_samples, replace=False)
        pred_m = eval_model_sdf(model, qn[idx], device) * sf           # meters
        true_m = tree.query(qn[idx] * sf + c)[0]                        # meters, unsigned
        ext = np.abs(qsdf[idx]) * sf > 0.01                             # clearly exterior
        err = np.abs(pred_m[ext]) - true_m[ext]
        corr = np.corrcoef(np.abs(pred_m[ext]), true_m[ext])[0, 1]
        print(f"{link:16s} {err.mean():12.4f} {np.abs(err).mean():10.4f} {corr:6.3f} "
              f"{np.percentile(np.abs(err),95):12.4f}")
        worst = max(worst, np.abs(err).mean())
    print(f"\nworst |err| across links: {worst:.4f} m")


if __name__ == "__main__":
    main()
