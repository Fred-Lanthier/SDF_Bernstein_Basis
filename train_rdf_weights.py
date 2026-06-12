#!/usr/bin/env python3
"""Fine-tune Bernstein robot-link SDFs with mesh-normal supervision."""

import argparse
import os

import numpy as np
import torch
import trimesh

from src.core.train.weight_train import (
    BernsteinWeightsTrain,
    build_normal_ray_training_data,
    elevate_bernstein_weights,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
PANDA_TEST = os.path.join(ROOT, "panda_test")
DEFAULT_OUTPUT_DIR = os.path.join(
    PANDA_TEST,
    "Models",
    "near_surface_gradient_experiment",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune existing Bernstein weights using SDF values, Eikonal "
            "regularization, and verified outward mesh-normal directions."
        )
    )
    parser.add_argument(
        "--links",
        nargs="+",
        default=["panda_link4"],
        help="Link names to train. Defaults to panda_link4.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--n-func",
        type=int,
        help=(
            "Optional larger basis count. Existing weights are degree-elevated "
            "exactly before fine-tuning."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-near", type=int, default=2048)
    parser.add_argument("--batch-rand", type=int, default=1024)
    parser.add_argument("--batch-ray", type=int, default=1024)
    parser.add_argument("--ray-points", type=int, default=20000)
    parser.add_argument(
        "--ray-min-distance",
        type=float,
        default=0.001,
        help="Minimum outward ray offset in meters.",
    )
    parser.add_argument(
        "--ray-focus-distance",
        type=float,
        default=0.01,
        help="Upper edge of the near-surface shell receiving extra supervision.",
    )
    parser.add_argument(
        "--ray-focus-fraction",
        type=float,
        default=0.75,
        help="Fraction of generated and batched rays assigned to the focus shell.",
    )
    parser.add_argument(
        "--ray-max-distance",
        type=float,
        default=0.05,
        help="Maximum outward mesh-normal offset in meters.",
    )
    parser.add_argument("--lambda-ray", type=float, default=2.0)
    parser.add_argument("--lambda-eikonal", type=float, default=0.05)
    parser.add_argument("--lambda-direction", type=float, default=0.2)
    parser.add_argument("--lambda-gradient-vector", type=float, default=0.25)
    parser.add_argument(
        "--near-surface-weight",
        type=float,
        default=4.0,
        help="Additional exponential weight applied near the robot surface.",
    )
    parser.add_argument(
        "--ray-normal-tolerance",
        type=float,
        default=5.0,
        help="Reject training rays whose exact gradient turns by more than this many degrees.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Candidate model directory. Existing production models are not overwritten.",
    )
    return parser.parse_args()


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but CUDA is unavailable.")
    return torch.device(value)


def load_training_inputs(link_name):
    dataset_path = os.path.join(PANDA_TEST, "Dataset", f"{link_name}.npy")
    mesh_path = os.path.join(PANDA_TEST, "Meshes", f"{link_name}.stl")
    model_path = os.path.join(PANDA_TEST, "Models", f"{link_name}_w.pt")
    for path in (dataset_path, mesh_path, model_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    dataset = np.load(dataset_path, allow_pickle=True).item()
    near_mask = np.linalg.norm(dataset["near_points"], axis=1) <= 1.0
    dataset["near_points"] = dataset["near_points"][near_mask]
    dataset["near_sdf"] = dataset["near_sdf"][near_mask]
    mesh = trimesh.load(mesh_path, force="mesh")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    return dataset, mesh, model


def train_link(args, link_name, device):
    dataset, mesh, model = load_training_inputs(link_name)
    baseline_n_func = int(model["n_func"])
    n_func = args.n_func or baseline_n_func
    if n_func < baseline_n_func:
        raise ValueError(
            f"--n-func {n_func} cannot be lower than the existing "
            f"n_func={baseline_n_func} for {link_name}."
        )
    initial_weights = elevate_bernstein_weights(
        model["weights"],
        baseline_n_func,
        n_func,
    )
    if n_func != baseline_n_func:
        print(
            f"Degree-elevated {link_name}: n_func {baseline_n_func} -> {n_func} "
            "without changing its initial SDF."
        )
    trainer = BernsteinWeightsTrain(
        n_func=n_func,
        domain_min=float(model["domain_min"]),
        domain_max=float(model["domain_max"]),
        device=str(device),
        dtype=torch.float32,
    )

    print(f"\nGenerating {args.ray_points:,} verified normal-ray samples for {link_name}...")
    ray_points, ray_sdf, ray_directions = build_normal_ray_training_data(
        mesh,
        dataset["mesh_centroid_offset"],
        dataset["mesh_scale_factor"],
        number_of_points=args.ray_points,
        min_distance_m=args.ray_min_distance,
        focus_distance_m=args.ray_focus_distance,
        focus_fraction=args.ray_focus_fraction,
        max_distance_m=args.ray_max_distance,
        normal_tolerance_degrees=args.ray_normal_tolerance,
        seed=args.seed,
    )
    print(
        f"Training pools: near={len(dataset['near_points']):,}, "
        f"random={len(dataset['query_points']):,}, rays={len(ray_points):,}"
    )

    weights = trainer.train_geometric(
        dataset["near_points"],
        dataset["near_sdf"],
        dataset["query_points"],
        dataset["query_sdf"],
        ray_points,
        ray_sdf,
        ray_directions,
        initial_weights=initial_weights,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        sample_near=args.batch_near,
        sample_rand=args.batch_rand,
        sample_ray=args.batch_ray,
        lambda_ray=args.lambda_ray,
        lambda_eikonal=args.lambda_eikonal,
        lambda_direction=args.lambda_direction,
        lambda_gradient_vector=args.lambda_gradient_vector,
        ray_focus_distance=(
            args.ray_focus_distance / float(dataset["mesh_scale_factor"])
        ),
        ray_focus_fraction=args.ray_focus_fraction,
        near_surface_weight=args.near_surface_weight,
        seed=args.seed,
    )

    candidate = dict(model)
    candidate["n_func"] = int(n_func)
    candidate["weights"] = weights.cpu()
    output_path = os.path.join(args.output_dir, f"{link_name}_w.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(candidate, output_path)
    print(f"Saved candidate model: {output_path}")


def main():
    args = parse_args()
    positive_integer_args = {
        "--epochs": args.epochs,
        "--batch-near": args.batch_near,
        "--batch-rand": args.batch_rand,
        "--batch-ray": args.batch_ray,
        "--ray-points": args.ray_points,
    }
    if args.n_func is not None:
        positive_integer_args["--n-func"] = args.n_func
    invalid = [name for name, value in positive_integer_args.items() if value <= 0]
    if invalid:
        raise ValueError(f"Must be positive: {', '.join(invalid)}")
    if args.learning_rate <= 0.0 or args.ray_max_distance <= 0.0:
        raise ValueError("--learning-rate and --ray-max-distance must be positive.")
    if not 0.0 < args.ray_focus_fraction <= 1.0:
        raise ValueError("--ray-focus-fraction must be in (0, 1].")
    if not 0.0 < args.ray_min_distance < args.ray_focus_distance:
        raise ValueError(
            "Expected 0 < --ray-min-distance < --ray-focus-distance."
        )
    if args.ray_focus_distance >= args.ray_max_distance:
        raise ValueError(
            "--ray-focus-distance must be smaller than --ray-max-distance."
        )
    nonnegative_values = {
        "--lambda-ray": args.lambda_ray,
        "--lambda-eikonal": args.lambda_eikonal,
        "--lambda-direction": args.lambda_direction,
        "--lambda-gradient-vector": args.lambda_gradient_vector,
        "--near-surface-weight": args.near_surface_weight,
    }
    invalid_nonnegative = [
        name for name, value in nonnegative_values.items() if value < 0.0
    ]
    if invalid_nonnegative:
        raise ValueError(f"Must be nonnegative: {', '.join(invalid_nonnegative)}")
    if args.ray_normal_tolerance <= 0.0:
        raise ValueError("--ray-normal-tolerance must be positive.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Candidate output directory: {os.path.abspath(args.output_dir)}")
    for link_name in args.links:
        train_link(args, link_name, device)


if __name__ == "__main__":
    main()
