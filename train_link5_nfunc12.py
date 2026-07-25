#!/usr/bin/env python3
"""Fit a fresh higher-order (n_func=12) Bernstein SDF for a single link.

Two stages:
  1. Recursive least squares (`train`) for a fast, accurate value-fit init.
  2. Geometric fine-tune (`train_geometric`) adding ray/eikonal/direction terms.

The candidate is written with the new n_func so the visualizer/CBF can load it
via --model-override-dir without touching the production model.
"""

import argparse
import os

import numpy as np
import torch
import trimesh

from src.core.train.weight_train import (
    BernsteinWeightsTrain,
    build_normal_ray_training_data,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
PANDA_TEST = os.path.join(ROOT, "panda_test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--link", default="panda_link3")
    p.add_argument("--n-func", type=int, default=12)
    p.add_argument("--rls-iters", type=int, default=80)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ray-points", type=int, default=20000)
    p.add_argument("--ray-max-distance", type=float, default=0.05)
    p.add_argument("--lambda-ray", type=float, default=2.0)
    p.add_argument("--lambda-eikonal", type=float, default=0.05)
    p.add_argument("--lambda-direction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--output-dir",
        default=os.path.join(PANDA_TEST, "Models", "nfunc12_candidate"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    link = args.link

    dataset = np.load(
        os.path.join(PANDA_TEST, "Dataset", f"{link}.npy"), allow_pickle=True
    ).item()
    near_mask = np.linalg.norm(dataset["near_points"], axis=1) <= 1.0
    dataset["near_points"] = dataset["near_points"][near_mask]
    dataset["near_sdf"] = dataset["near_sdf"][near_mask]
    mesh = trimesh.load(os.path.join(PANDA_TEST, "Meshes", f"{link}.stl"), force="mesh")
    model = torch.load(
        os.path.join(PANDA_TEST, "Models", f"{link}_w.pt"),
        map_location="cpu",
        weights_only=False,
    )

    print(f"Link {link}: training n_func={args.n_func} ({args.n_func**3} params) on {device}")
    trainer = BernsteinWeightsTrain(
        n_func=args.n_func,
        domain_min=float(model["domain_min"]),
        domain_max=float(model["domain_max"]),
        device=str(device),
        dtype=torch.float32,
    )

    # Stage 1: recursive least squares value-fit (good init for the higher order).
    print(f"\n[Stage 1] Recursive least squares for {args.rls_iters} iters...")
    init_weights = trainer.train(
        dataset["near_points"],
        dataset["near_sdf"],
        dataset["query_points"],
        dataset["query_sdf"],
        epoches=args.rls_iters,
        sample_near=2048,
        sample_rand=512,
    )

    # Stage 2: geometric fine-tune with ray/eikonal/direction supervision.
    print(f"\nGenerating {args.ray_points:,} verified normal-ray samples...")
    ray_points, ray_sdf, ray_directions = build_normal_ray_training_data(
        mesh,
        dataset["mesh_centroid_offset"],
        dataset["mesh_scale_factor"],
        number_of_points=args.ray_points,
        max_distance_m=args.ray_max_distance,
        seed=args.seed,
    )
    print(f"\n[Stage 2] Geometric fine-tune for {args.epochs} epochs...")
    weights = trainer.train_geometric(
        dataset["near_points"],
        dataset["near_sdf"],
        dataset["query_points"],
        dataset["query_sdf"],
        ray_points,
        ray_sdf,
        ray_directions,
        initial_weights=init_weights,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_ray=args.lambda_ray,
        lambda_eikonal=args.lambda_eikonal,
        lambda_direction=args.lambda_direction,
        seed=args.seed,
    )

    candidate = dict(model)
    candidate["n_func"] = int(args.n_func)
    candidate["weights"] = weights.cpu()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{link}_w.pt")
    torch.save(candidate, out_path)
    print(f"\nSaved candidate: {out_path}  (n_func={args.n_func}, weights={tuple(weights.shape)})")


if __name__ == "__main__":
    main()
