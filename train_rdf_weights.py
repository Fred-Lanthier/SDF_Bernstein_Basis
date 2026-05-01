#!/usr/bin/env python3

import argparse

import torch

from src.rdf_weights import RDF_Weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train weights-only RDF models.")
    parser.add_argument("--ws-path", required=True, help="Workspace root containing Meshes/ and Models/.")
    parser.add_argument("--robot-name", default="", help="Optional prefix for workspace subfolders.")
    parser.add_argument("--links", nargs="+", required=True, help="Link names to train.")
    parser.add_argument("--n-func", type=int, default=24)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-near", type=int, default=1024)
    parser.add_argument("--batch-rand", type=int, default=64)
    parser.add_argument("--n-points", type=int, default=100000)
    parser.add_argument("--sample-point-count", type=int, default=1000000)
    parser.add_argument("--scan-count", type=int, default=100)
    parser.add_argument("--scan-resolution", type=int, default=400)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to cuda if available.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rdf = RDF_Weights(device=device)
    rdf.init_robot_folder(args.ws_path, robot_name=args.robot_name)
    rdf.train_links(
        link_names=args.links,
        n_func=args.n_func,
        iters=args.iters,
        batch_near=args.batch_near,
        batch_rand=args.batch_rand,
        robot_name=args.robot_name,
        debug=args.debug,
        n_points=args.n_points,
        sample_point_count=args.sample_point_count,
        scan_count=args.scan_count,
        scan_resolution=args.scan_resolution,
    )


if __name__ == "__main__":
    main()

