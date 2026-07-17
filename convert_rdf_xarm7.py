#!/usr/bin/env python3
"""Convert RDF xarm7 assets into the panda_test workspace format.

Sources (third_party/RDF):
  models/BP_8_xarm7.pt                      per-link n_func=8 Bernstein weights
  data/sdf_points_xarm7/voxel_128_xarm7_*.npy  DeepSDF-style samples (unit-sphere frame)
  panda_layer/meshes/voxel_128_xarm7/*.stl  voxelized link meshes (link frame)

Outputs (panda_test, same format as the panda production models):
  Dataset/xarm7_{link}.npy   near/query points + mesh normalization
  Meshes/xarm7_{link}.stl    link-frame mesh for normal-ray fine-tuning
  Models/xarm7_{link}_w.pt   n_func=8 seed model (weights + fitted ellipsoid)

The RDF weights are validated against the dataset SDF through THIS repo's
Bernstein basis before being trusted. All 6 axis-order permutations of the
(8,8,8) weight tensor are tried (basis conventions differ only by ordering);
the panda nfunc8 backup evaluated on the panda dataset provides the reference
error bar. If validation fails, the link is refit from scratch instead.

After conversion, fine-tune + degree-elevate to n_func=12 with:
  python3 train_rdf_weights.py --links xarm7_link1 ... xarm7_link7 \
      --n-func 12 --output-dir panda_test/Models
"""

import argparse
import itertools
import os
import shutil

import numpy as np
import torch
import trimesh

from src.core.math.fit_ellipsoid import compute_ellipsoid_parameters
from src.core.train.weight_train import BernsteinWeightsTrain

ROOT = os.path.dirname(os.path.abspath(__file__))
RDF_ROOT = os.path.normpath(os.path.join(ROOT, "..", "RDF"))
PANDA_TEST = os.path.join(ROOT, "panda_test")

BP_PATH = os.path.join(RDF_ROOT, "models", "BP_8_xarm7.pt")
DATA_DIR = os.path.join(RDF_ROOT, "data", "sdf_points_xarm7")
MESH_DIR = os.path.join(RDF_ROOT, "panda_layer", "meshes", "voxel_128_xarm7")

DEFAULT_LINKS = [
    "link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", nargs="+", default=DEFAULT_LINKS,
                        help="RDF mesh names to convert (output prefixed xarm7_).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mae-tolerance-factor", type=float, default=3.0,
                        help="Accept RDF weights if MAE <= factor * panda nfunc8 reference MAE.")
    parser.add_argument("--scratch-epochs", type=int, default=400,
                        help="Epochs for the from-scratch fallback fit.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def evaluate_weights(weights, points, device):
    """SDF prediction of flat n_func=8 weights on normalized-frame points."""
    poly = BernsteinWeightsTrain(n_func=8, domain_min=-1, domain_max=1,
                                 device=device, dtype=torch.float32)
    p = torch.from_numpy(points).float().to(device)
    w = weights.reshape(-1).float().to(device)
    preds = []
    for chunk in torch.split(p, 65536):
        phi, _ = poly.basis_function_from_3Dpoints(chunk, use_derivative=False)
        preds.append(phi @ w)
    return torch.cat(preds).cpu().numpy()


def best_permutation_mae(weights, points, sdf, device):
    """Try all axis orderings of the (8,8,8) weight tensor; return best."""
    w3 = weights.reshape(8, 8, 8)
    best = (None, np.inf)
    for perm in itertools.permutations(range(3)):
        pred = evaluate_weights(w3.permute(*perm).contiguous(), points, device)
        mae = float(np.mean(np.abs(pred - sdf)))
        if mae < best[1]:
            best = (perm, mae)
    return best


def reference_mae(device):
    """Panda nfunc8 backup on the panda dataset = 'known good' error bar."""
    backup_dir = None
    models_dir = os.path.join(PANDA_TEST, "Models")
    for name in sorted(os.listdir(models_dir)):
        if name.startswith("nfunc8_production_backup"):
            backup_dir = os.path.join(models_dir, name)
    if backup_dir is None:
        return None
    model = torch.load(os.path.join(backup_dir, "panda_link4_w.pt"),
                       map_location="cpu", weights_only=False)
    data = np.load(os.path.join(PANDA_TEST, "Dataset", "panda_link4.npy"),
                   allow_pickle=True).item()
    mask = np.linalg.norm(data["near_points"], axis=1) <= 1.0
    pred = evaluate_weights(model["weights"], data["near_points"][mask], device)
    return float(np.mean(np.abs(pred - data["near_sdf"][mask])))


def fit_from_scratch(dataset, epochs, device):
    trainer = BernsteinWeightsTrain(n_func=8, domain_min=-1, domain_max=1,
                                    device=device, dtype=torch.float32)
    weights = trainer.train(
        dataset["near_points"], dataset["near_sdf"],
        dataset["query_points"], dataset["query_sdf"],
        epoches=epochs,
    )
    return weights.detach().cpu()


def convert_link(rdf_name, bp_entries, ref_mae, args):
    out_name = f"xarm7_{rdf_name}"
    print(f"\n=== {rdf_name} -> {out_name} ===")

    raw = np.load(os.path.join(DATA_DIR, f"voxel_128_xarm7_{rdf_name}.npy"),
                  allow_pickle=True).item()
    mesh_src = os.path.join(MESH_DIR, f"{rdf_name}.stl")
    mesh = trimesh.load(mesh_src, force="mesh")

    # Link-frame -> unit-sphere normalization, same formula as RDF bf_sdf.py
    # (bounding-box centroid + max vertex distance) and mesh_to_sdf's
    # scale_to_unit_sphere, so it matches the frame of the sampled points.
    offset = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    scale = float(np.max(np.linalg.norm(mesh.vertices - offset, axis=1)))

    bp = next(v for v in bp_entries.values() if v["mesh_name"] == rdf_name)
    bp_offset = bp["offset"].numpy().astype(np.float64)
    d_off = float(np.linalg.norm(offset - bp_offset))
    d_scale = abs(scale - float(bp["scale"]))
    print(f"normalization: offset {offset.round(4)} scale {scale:.4f} "
          f"(vs BP_8: d_offset {d_off:.2e}, d_scale {d_scale:.2e})")

    near_points = raw["near_points"].astype(np.float32)
    near_sdf = raw["near_sdf"].astype(np.float32)
    mask = np.linalg.norm(near_points, axis=1) <= 1.0
    near_points, near_sdf = near_points[mask], near_sdf[mask]

    # Validate the RDF weights through this repo's basis (ordering-agnostic).
    perm, mae = best_permutation_mae(bp["weights"], near_points, near_sdf, args.device)
    verdict = "OK" if ref_mae is not None and mae <= args.mae_tolerance_factor * ref_mae else "FAIL"
    print(f"RDF weight validation: best perm {perm} MAE {mae:.5f} "
          f"(panda ref {ref_mae:.5f}) -> {verdict}")

    dataset = {
        "file_name": out_name,
        "near_points": near_points,
        "near_sdf": near_sdf,
        "query_points": raw["random_points"].astype(np.float64),
        "query_sdf": raw["random_sdf"].astype(np.float32),
        "mesh_centroid_offset": offset,
        "mesh_scale_factor": np.float64(scale),
        "sdf_domain_min": -1,
        "sdf_domain_max": 1,
    }
    np.save(os.path.join(PANDA_TEST, "Dataset", f"{out_name}.npy"), dataset)
    shutil.copyfile(mesh_src, os.path.join(PANDA_TEST, "Meshes", f"{out_name}.stl"))

    if verdict == "OK":
        weights = bp["weights"].reshape(8, 8, 8).permute(*perm).contiguous().reshape(-1).float()
    else:
        print("falling back to from-scratch n_func=8 fit...")
        weights = fit_from_scratch(dataset, args.scratch_epochs, args.device).reshape(-1).float()
        scratch_pred = evaluate_weights(weights, near_points, args.device)
        print(f"from-scratch MAE: {float(np.mean(np.abs(scratch_pred - near_sdf))):.5f}")

    # Ellipsoid over inside points, same as TrainWrapper.fit_ellipsoid.
    inside = near_points[near_sdf < 0]
    if len(inside) > 5000:
        inside = inside[np.random.choice(len(inside), 5000, replace=False)]
    axes, center, eigvec = compute_ellipsoid_parameters(inside)

    model = {
        "file_name": out_name,
        "file_suffix": "_w",
        "domain_min": -1,
        "domain_max": 1,
        "scale_factor": np.float64(scale),
        "centroid_offset": offset,
        "center_ellipsoid": np.asarray(center, dtype=np.float64),
        "axes_ellipsoid": np.asarray(axes, dtype=np.float64),
        "scales_ellipsoid": None,
        "eigen_vector_ellipsoid": np.asarray(eigvec, dtype=np.float64),
        "n_func": 8,
        "device": "cpu",
        "dtype": torch.float32,
        "weights": weights.cpu(),
    }
    out_path = os.path.join(PANDA_TEST, "Models", f"{out_name}_w.pt")
    torch.save(model, out_path)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    bp_entries = torch.load(BP_PATH, map_location="cpu", weights_only=False)
    ref = reference_mae(args.device)
    print(f"panda nfunc8 reference MAE (panda_link4): {ref:.5f}")
    for rdf_name in args.links:
        convert_link(rdf_name, bp_entries, ref, args)
    print("\nDone. Next step:")
    print("  python3 train_rdf_weights.py --links "
          + " ".join(f"xarm7_{l}" for l in args.links if l != "link_base")
          + " --n-func 12 --output-dir panda_test/Models")


if __name__ == "__main__":
    main()
