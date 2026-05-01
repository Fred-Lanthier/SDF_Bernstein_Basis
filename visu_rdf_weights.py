#!/usr/bin/env python3

import argparse

import torch

from src.rdf_weights import RDF_Weights


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize weights-only RDF models.")
    parser.add_argument("--ws-path", required=True, help="Workspace root containing Models/.")
    parser.add_argument("--robot-name", default="", help="Optional prefix for workspace subfolders.")
    parser.add_argument("--links", nargs="+", required=True, help="Link names to visualize.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to cuda if available.")
    parser.add_argument("--mesh-color", default="#302E2E")
    parser.add_argument("--mesh-opacity", type=float, default=1.0)
    parser.add_argument("--rdf-color", default="#c9c9bf")
    parser.add_argument("--rdf-opacity", type=float, default=0.35)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rdf = RDF_Weights(device=device)
    rdf.init_robot_folder(args.ws_path, robot_name=args.robot_name)
    rdf.add_models(link_names=args.links, robot_name=args.robot_name)

    forward_dict = {}
    for idx, link in enumerate(args.links):
        tf = torch.eye(4, device=device, dtype=rdf.dtype)
        tf[0, 3] = float(idx) * 3.0
        forward_dict[link] = tf

    rdf.visualize_scene(
        forward_as_dict=forward_dict,
        links_as_mesh=False,
        links_as_rdf=True,
        rdf_link_names=args.links,
        mesh_color=args.mesh_color,
        mesh_opacity=args.mesh_opacity,
        rdf_color=args.rdf_color,
        rdf_opacity=args.rdf_opacity,
    )


if __name__ == "__main__":
    main()
