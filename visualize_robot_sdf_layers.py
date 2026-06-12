#!/usr/bin/env python3
"""
Visualize and evaluate the Bernstein SDF used by the CBF safety node.

Examples:
  python3 visualize_robot_sdf_layers.py
  python3 visualize_robot_sdf_layers.py --links all --levels 0 0.01 0.05 0.10
  python3 visualize_robot_sdf_layers.py --from-joint-states --grid-spacing 0.015
  python3 visualize_robot_sdf_layers.py --eval-point 0.45,0.0,0.55 --no-show
  python3 visualize_robot_sdf_layers.py --benchmark --no-show

The arrows show +grad SDF in world coordinates, i.e. the direction of
increasing signed distance away from the closest robot link.
"""

import argparse
import os
import sys
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "vision_processing_matplotlib"),
)

import numpy as np
import torch
import trimesh
import xacro

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


SDF_BERNSTEIN_PATH = os.path.dirname(os.path.abspath(__file__))
PKG_PATH = os.path.abspath(os.path.join(SDF_BERNSTEIN_PATH, os.pardir, os.pardir))

sys.path.insert(0, PKG_PATH)
sys.path.insert(0, SDF_BERNSTEIN_PATH)

from third_party.RDF.urdf_layer import URDFLayer
from third_party.SDF_Bernstein_Basis.bernstein_core import BernsteinCore
from third_party.SDF_Bernstein_Basis.src.core.assets.load_model_wrapper import (
    load_link_weight_model,
)
from third_party.SDF_Bernstein_Basis.src.rdf_weights import RDF_Weights


ALL_LINKS = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
    "fork_tip",
]

CBF_PROTECTED_LINKS = [
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "fork_tip",
]

DEFAULT_Q = np.array(
    [-0.000059, -0.125928, 0.000117, -2.193312, -0.000251, 2.064780, 0.785511],
    dtype=np.float32,
)

SHELL_COLORS = [
    "#2f2f2f",
    "#d7191c",
    "#fdae61",
    "#2c7bb6",
    "#1a9641",
    "#7b3294",
    "#008080",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Bernstein SDF distance shells and world-frame SDF gradients."
    )
    parser.add_argument(
        "--q",
        nargs=7,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=DEFAULT_Q.tolist(),
        help="Seven Panda joint positions in radians. Defaults to the launch spawn pose.",
    )
    parser.add_argument(
        "--from-joint-states",
        action="store_true",
        help="Read one /joint_states message and use panda_joint1..7 instead of --q.",
    )
    parser.add_argument(
        "--joint-states-topic",
        default="/joint_states",
        help="JointState topic used with --from-joint-states.",
    )
    parser.add_argument(
        "--links",
        default="protected",
        help="Link set: 'protected', 'all', or a comma-separated list of SDF link names.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.05, 0.10],
        help="SDF shell distances in meters. Example: --levels 0 0.01 0.05 0.10",
    )
    parser.add_argument(
        "--gradient-levels",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10],
        help="Shell distances in meters where gradient arrows are sampled.",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.02,
        help="SDF sampling grid spacing in meters.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding around the selected robot mesh bounds in meters.",
    )
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Manual SDF grid bounds in world meters. Overrides automatic mesh bounds.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32768,
        help="Number of grid/query points evaluated per SDF batch.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device for SDF evaluation.",
    )
    parser.add_argument(
        "--model-override-dir",
        help=(
            "Optional directory containing candidate <link>_w.pt files. "
            "Links without a candidate continue using the production model."
        ),
    )
    parser.add_argument(
        "--finger-position",
        type=float,
        default=0.001,
        help="Finger joint value appended to q for URDF FK.",
    )
    parser.add_argument(
        "--eval-point",
        action="append",
        default=[],
        help="Evaluate one world point 'x,y,z'. May be passed multiple times.",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=400,
        help="Maximum total gradient arrows to draw across all gradient levels.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=0.04,
        help="PyVista arrow scale in meters.",
    )
    parser.add_argument(
        "--robot-opacity",
        type=float,
        default=0.35,
        help="Opacity of the transformed robot visual mesh.",
    )
    parser.add_argument(
        "--shell-opacity",
        type=float,
        default=0.30,
        help="Opacity of non-zero SDF distance shells.",
    )
    parser.add_argument(
        "--surface-opacity",
        type=float,
        default=0.55,
        help="Opacity of the zero SDF surface.",
    )
    parser.add_argument(
        "--save-screenshot",
        help="Optional PNG path. Enables off-screen rendering if --no-show is also used.",
    )
    parser.add_argument(
        "--xvfb",
        action="store_true",
        help="Start a virtual framebuffer before PyVista rendering. Useful on headless machines.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive PyVista window.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Generate four Bernstein-versus-mesh-ground-truth benchmark plots.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        default="sdf_benchmark_plots",
        help="Directory where benchmark PNG files are written.",
    )
    parser.add_argument(
        "--ground-truth-links",
        default=None,
        help=(
            "Link set used to build the EXACT mesh ground truth. Default: match "
            "--links, i.e. measure the SDF against exactly the links it represents "
            "(apples-to-apples). Pass 'all' to measure against the complete robot "
            "(then junctions with links outside --links count as error), 'protected', "
            "or a comma-separated list."
        ),
    )
    parser.add_argument(
        "--benchmark-points",
        type=int,
        default=500,
        help="Number of verified external near-surface points used in the accuracy plots.",
    )
    parser.add_argument(
        "--benchmark-timing-points",
        type=int,
        default=32,
        help="Number of query points in each timing measurement.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=10,
        help="Number of measured repetitions in each timing boxplot.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=2,
        help="Number of unmeasured warmup repetitions per method.",
    )
    parser.add_argument(
        "--benchmark-surface-noise",
        type=float,
        default=0.05,
        help="Maximum outward normal offset from sampled mesh surfaces in meters.",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=7,
        help="Random seed used to generate benchmark points.",
    )
    parser.add_argument(
        "--benchmark-gradient-ray-tolerance",
        "--benchmark-gradient-jump-threshold",
        dest="benchmark_gradient_ray_tolerance",
        type=float,
        default=5.0,
        help=(
            "Maximum angle in degrees between the exact mesh gradient and "
            "the originating surface normal in the stable-ray comparison."
        ),
    )
    return parser.parse_args()


def resolve_link_names(link_arg):
    if link_arg == "protected":
        return list(CBF_PROTECTED_LINKS)
    if link_arg == "all":
        return list(ALL_LINKS)
    links = [item.strip() for item in link_arg.split(",") if item.strip()]
    unknown = sorted(set(links) - set(ALL_LINKS))
    if unknown:
        raise ValueError("Unknown SDF link name(s): " + ", ".join(unknown))
    return links


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def parse_eval_points(values):
    points = []
    for value in values:
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) != 3:
            raise ValueError(f"Expected --eval-point x,y,z, got '{value}'")
        points.append([float(piece) for piece in pieces])
    return np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)


def read_joint_states(topic):
    import rospy
    from sensor_msgs.msg import JointState

    rospy.init_node("robot_sdf_layer_visualizer", anonymous=True, disable_signals=True)
    msg = rospy.wait_for_message(topic, JointState, timeout=5.0)
    positions = {name: value for name, value in zip(msg.name, msg.position)}
    missing = [f"panda_joint{i}" for i in range(1, 8) if f"panda_joint{i}" not in positions]
    if missing:
        raise RuntimeError(f"JointState is missing: {', '.join(missing)}")
    return np.array([positions[f"panda_joint{i}"] for i in range(1, 8)], dtype=np.float32)


def build_robot_layer(device):
    xacro_path = os.path.join(PKG_PATH, "urdf", "panda_camera.xacro")
    doc = xacro.process_file(xacro_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as handle:
        handle.write(doc.toxml())
        urdf_path = handle.name

    return URDFLayer(
        urdf_path=urdf_path,
        device=device,
        package_dir=PKG_PATH,
        voxel_dir=os.path.join(PKG_PATH, "third_party", "RDF", "panda_layer", "meshes", "voxel_128"),
    )


def build_sdf_stack(device, link_names, model_override_dir=None):
    robot_layer = build_robot_layer(device)
    weights_dir = os.path.join(PKG_PATH, "third_party", "SDF_Bernstein_Basis", "panda_test")
    weights = RDF_Weights(device=device, dtype=torch.float32)
    weights.init_robot_folder(weights_dir, robot_name="panda")
    weights.add_models(link_names, robot_name="panda")

    if model_override_dir:
        override_dir = os.path.abspath(model_override_dir)
        if not os.path.isdir(override_dir):
            raise FileNotFoundError(f"Model override directory not found: {override_dir}")
        for link_name in link_names:
            candidate_path = os.path.join(override_dir, f"{link_name}_w.pt")
            if not os.path.isfile(candidate_path):
                continue
            candidate_dict = torch.load(
                candidate_path,
                map_location=device,
                weights_only=False,
            )
            candidate = load_link_weight_model(
                candidate_dict,
                device=device,
                dtype=torch.float32,
            )
            setattr(weights, link_name + weights.model_extension, candidate)
            print(f"Using candidate model for {link_name}: {candidate_path}")

    core = BernsteinCore(weights, robot_layer, device, link_names)
    return robot_layer, weights, core


def q7_to_q9(q7_np, finger_position, device):
    q7 = torch.as_tensor(q7_np, dtype=torch.float32, device=device).reshape(1, 7)
    fingers = torch.full((1, 2), float(finger_position), dtype=torch.float32, device=device)
    return torch.cat([q7, fingers], dim=1)


def link_matches(mesh_link_name, sdf_link_name):
    mesh_name = mesh_link_name.replace("panda_", "")
    sdf_name = sdf_link_name.replace("panda_", "")
    return mesh_name == sdf_name or mesh_name in sdf_name or sdf_name in mesh_name


def selected_robot_mesh(robot_layer, pose, q9, link_names):
    all_meshes = robot_layer.get_forward_robot_mesh(pose, q9)[0]
    selected = []
    for info, mesh in zip(robot_layer.meshes_info, all_meshes):
        if any(link_matches(info["link_name"], link_name) for link_name in link_names):
            selected.append(mesh)
    if not selected:
        raise RuntimeError("No URDF visual meshes matched the selected SDF link names.")
    if len(selected) == 1:
        return selected[0]
    return trimesh.util.concatenate(selected)


def matched_link_transforms(robot_layer, pose, q9, link_names):
    transformations = robot_layer.get_transformations_each_link(pose, q9)
    matched = []
    for link_name in link_names:
        match_index = None
        for index, info in enumerate(robot_layer.meshes_info):
            if link_matches(info["link_name"], link_name):
                match_index = index
                break
        if match_index is None:
            raise ValueError(f"Link {link_name} not found in URDF visuals")
        transform = transformations[match_index]
        if transform.ndim == 3:
            if transform.shape[0] != 1:
                raise ValueError("Benchmarking currently supports one robot configuration at a time.")
            transform = transform[0]
        matched.append(transform.detach().cpu().numpy())
    return matched


def build_ground_truth_meshes(weights, robot_layer, pose, q9, link_names):
    weights.add_mesh(link_names, robot_name="panda")
    transforms = matched_link_transforms(robot_layer, pose, q9, link_names)
    meshes = []
    for link_name, transform in zip(link_names, transforms):
        mesh = getattr(weights, link_name + weights.mesh_extension).mesh.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def ground_truth_distance(meshes, points_np):
    per_link_distance = []
    for mesh in meshes:
        _, distance, _ = trimesh.proximity.closest_point(mesh, points_np)
        per_link_distance.append(distance)
    return np.min(np.stack(per_link_distance, axis=0), axis=0).astype(np.float32)


def ground_truth_distance_and_gradient(meshes, points_np, return_external_mask=False):
    per_link_distance = []
    per_link_gradient = []
    per_link_signed_distance = []

    for mesh in meshes:
        closest, unsigned_distance, triangle_ids = trimesh.proximity.closest_point(
            mesh,
            points_np,
        )
        displacement = points_np - closest
        nearest_normals = mesh.face_normals[triangle_ids]
        inside = np.einsum("ij,ij->i", displacement, nearest_normals) < 0.0
        signed_distance = unsigned_distance.copy()
        signed_distance[inside] *= -1.0

        norms = np.linalg.norm(displacement, axis=1)
        gradient = np.zeros_like(displacement)
        valid = norms > 1e-10
        gradient[valid] = displacement[valid] / norms[valid, None]

        on_surface = ~valid
        if np.any(on_surface):
            gradient[on_surface] = mesh.face_normals[triangle_ids[on_surface]]

        per_link_distance.append(unsigned_distance)
        per_link_gradient.append(gradient)
        per_link_signed_distance.append(signed_distance)

    distance_by_link = np.stack(per_link_distance, axis=0)
    gradient_by_link = np.stack(per_link_gradient, axis=0)
    nearest_link = np.argmin(distance_by_link, axis=0)
    point_indices = np.arange(len(points_np))
    result = (
        distance_by_link[nearest_link, point_indices].astype(np.float32),
        gradient_by_link[nearest_link, point_indices].astype(np.float32),
    )
    if not return_external_mask:
        return result

    signed_distance_by_link = np.stack(per_link_signed_distance, axis=0)
    external_mask = np.all(signed_distance_by_link >= 0.0, axis=0)
    return result[0], result[1], external_mask


def make_external_benchmark_dataset(meshes, count, surface_noise, seed):
    rng = np.random.default_rng(seed)
    minimum_offset = min(max(surface_noise * 0.02, 1e-5), surface_noise * 0.5)
    accepted_points = []
    accepted_distance = []
    accepted_gradient = []

    for round_index in range(4):
        accepted_count = sum(len(points) for points in accepted_points)
        remaining = count - accepted_count
        if remaining <= 0:
            break

        candidate_count = max(remaining * 2, len(meshes) * 16)
        per_mesh_count = int(np.ceil(candidate_count / len(meshes)))
        candidates = []
        for mesh_index, mesh in enumerate(meshes):
            sample_seed = seed + round_index * 1000 + mesh_index
            surface_points, face_indices = trimesh.sample.sample_surface(
                mesh,
                per_mesh_count,
                seed=sample_seed,
            )
            normals = mesh.face_normals[face_indices]
            offsets = rng.uniform(
                minimum_offset,
                surface_noise,
                size=(per_mesh_count, 1),
            )
            candidates.append(surface_points + normals * offsets)

        candidate_points = np.vstack(candidates).astype(np.float32)
        distance, gradient, external = ground_truth_distance_and_gradient(
            meshes,
            candidate_points,
            return_external_mask=True,
        )
        valid = external & (distance > minimum_offset * 0.25)
        valid &= distance <= surface_noise * 1.05
        accepted_points.append(candidate_points[valid])
        accepted_distance.append(distance[valid])
        accepted_gradient.append(gradient[valid])

    if not accepted_points:
        raise RuntimeError("Could not generate external benchmark points.")

    points = np.concatenate(accepted_points, axis=0)
    distance = np.concatenate(accepted_distance, axis=0)
    gradient = np.concatenate(accepted_gradient, axis=0)
    if len(points) < count:
        raise RuntimeError(
            f"Generated only {len(points)} verified external points; requested {count}."
        )

    selection = rng.permutation(len(points))[:count]
    return points[selection], distance[selection], gradient[selection]


def make_external_benchmark_rays(sample_meshes, truth_meshes, ray_count, offsets, seed):
    """Generate verified outward normal rays.

    Rays are sampled from sample_meshes (the links under test) and their geometric
    cleanliness (monotone, distance approximately equal to the marched offset) is
    judged against those same meshes, so rays heading toward a junction with a link
    outside the tested set are still kept. The returned ground-truth distance and
    gradient, however, are measured against truth_meshes (the complete robot), so a
    closer neighboring link correctly reduces the reference distance.
    """
    rng = np.random.default_rng(seed)
    accepted_points = []
    accepted_distances = []
    accepted_gradients = []
    accepted_normals = []

    max_distance_error = max(0.003, 0.20 * float(offsets[-1]))
    progress = (
        _tqdm(total=ray_count, desc="Verified rays", unit="ray")
        if _tqdm is not None
        else None
    )
    for round_index in range(12):
        remaining = ray_count - len(accepted_points)
        if remaining <= 0:
            break

        per_mesh_count = max(int(np.ceil(remaining * 3 / len(sample_meshes))), 8)
        for mesh_index, mesh in enumerate(sample_meshes):
            sample_seed = seed + 10000 + round_index * 1000 + mesh_index
            surface_points, face_indices = trimesh.sample.sample_surface(
                mesh,
                per_mesh_count,
                seed=sample_seed,
            )
            normals = mesh.face_normals[face_indices]
            for surface_point, normal in zip(surface_points, normals):
                ray_points = (
                    surface_point[None, :] + offsets[:, None] * normal[None, :]
                ).astype(np.float32)
                # Cleanliness is judged against the sampled link only.
                sample_distance, _, sample_external = ground_truth_distance_and_gradient(
                    sample_meshes,
                    ray_points,
                    return_external_mask=True,
                )
                if not np.all(sample_external):
                    continue
                if np.max(np.abs(sample_distance - offsets)) > max_distance_error:
                    continue
                if not np.all(np.diff(sample_distance) >= -max_distance_error):
                    continue
                # Ground truth uses the complete robot, so a closer link outside the
                # tested set lowers the reference distance instead of being ignored.
                distance, gradient, truth_external = ground_truth_distance_and_gradient(
                    truth_meshes,
                    ray_points,
                    return_external_mask=True,
                )
                if not np.all(truth_external):
                    continue
                accepted_points.append(ray_points)
                accepted_distances.append(distance.astype(np.float32))
                accepted_gradients.append(gradient.astype(np.float32))
                accepted_normals.append(normal.astype(np.float32))
                if progress is not None:
                    progress.update(1)
                if len(accepted_points) >= ray_count:
                    break
            if len(accepted_points) >= ray_count:
                break

    if progress is not None:
        progress.close()

    if len(accepted_points) < ray_count:
        raise RuntimeError(
            f"Generated only {len(accepted_points)} verified external normal rays; "
            f"requested {ray_count}."
        )

    selection = rng.permutation(len(accepted_points))[:ray_count]
    return (
        np.stack([accepted_points[index] for index in selection], axis=0),
        np.stack([accepted_distances[index] for index in selection], axis=0),
        np.stack([accepted_gradients[index] for index in selection], axis=0),
        np.stack([accepted_normals[index] for index in selection], axis=0),
    )


def normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.zeros_like(vectors)
    valid = norms[:, 0] > 1e-10
    normalized[valid] = vectors[valid] / norms[valid]
    return normalized, valid


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_runtime_ms(operation, repeats, warmup, device):
    durations = []
    for iteration in range(warmup + repeats):
        synchronize_device(device)
        start = time.perf_counter()
        operation()
        synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= warmup:
            durations.append(elapsed_ms)
    return np.asarray(durations, dtype=np.float64)


def save_distance_comparison_plot(offsets, ground_truth, bernstein, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = bernstein - ground_truth
    ground_truth_flat = ground_truth.reshape(-1)
    bernstein_flat = bernstein.reshape(-1)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    correlation = float(np.corrcoef(ground_truth_flat, bernstein_flat)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    for ray in bernstein:
        axes[0].plot(offsets, ray, color="#2c7bb6", alpha=0.20, linewidth=1.0)
    axes[0].plot(
        offsets,
        np.mean(bernstein, axis=0),
        color="#2c7bb6",
        linewidth=2.5,
        label="Bernstein mean",
    )
    axes[0].plot(
        offsets,
        np.mean(ground_truth, axis=0),
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Exact mesh mean",
    )
    axes[0].plot(offsets, offsets, color="#d7191c", linestyle=":", linewidth=1.8, label="Ideal offset")
    axes[0].set_xlabel("Outward normal offset from mesh surface [m]")
    axes[0].set_ylabel("Distance [m]")
    axes[0].set_title("Distance Profiles Along Surface Normals")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    mean_error = np.mean(errors, axis=0)
    min_error = np.min(errors, axis=0)
    max_error = np.max(errors, axis=0)
    for ray in errors:
        axes[1].plot(offsets, ray, color="#2c7bb6", alpha=0.20, linewidth=1.0)
    axes[1].fill_between(
        offsets, min_error, max_error, color="#2c7bb6", alpha=0.12, label="min-max envelope"
    )
    axes[1].plot(offsets, mean_error, color="#2c7bb6", linewidth=2.5, label="Mean error")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Outward normal offset from mesh surface [m]")
    axes[1].set_ylabel("Bernstein - exact distance [m]")
    axes[1].set_title("Distance Error (per ray vs complete-robot SDF)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        "External Distance Comparison\n"
        f"MAE={mae:.5f} m, bias={bias:.5f} m, correlation={correlation:.3f}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return mae, rmse, bias, correlation


def save_gradient_comparison_plot(
    offsets,
    ground_truth,
    bernstein,
    ray_normals,
    ray_tolerance_degrees,
    output_path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat_ground_truth = ground_truth.reshape(-1, 3)
    flat_bernstein = bernstein.reshape(-1, 3)
    ground_truth_unit, ground_truth_valid = normalize_rows(flat_ground_truth)
    bernstein_unit, bernstein_valid = normalize_rows(flat_bernstein)
    valid = ground_truth_valid & bernstein_valid
    if not np.any(valid):
        raise RuntimeError("No non-zero gradients were available for comparison.")

    cosine = np.sum(ground_truth_unit[valid] * bernstein_unit[valid], axis=1)
    angle_degrees = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    mean_angle = float(np.mean(angle_degrees))
    median_angle = float(np.median(angle_degrees))
    direction_rmse = float(
        np.sqrt(np.mean((bernstein_unit[valid] - ground_truth_unit[valid]) ** 2))
    )
    component_correlation = float(
        np.corrcoef(
            ground_truth_unit[valid].reshape(-1),
            bernstein_unit[valid].reshape(-1),
        )[0, 1]
    )

    R, S, _ = ground_truth.shape
    all_ground_truth_unit, all_ground_truth_valid = normalize_rows(flat_ground_truth)
    all_bernstein_unit, all_bernstein_valid = normalize_rows(flat_bernstein)
    all_valid = all_ground_truth_valid & all_bernstein_valid
    all_cosine = np.sum(all_ground_truth_unit * all_bernstein_unit, axis=1)
    all_cosine = np.where(all_valid, all_cosine, np.nan).reshape(R, S)
    angle_by_ray = np.degrees(np.arccos(np.clip(all_cosine, -1.0, 1.0)))
    mean_angle_by_offset = np.nanmean(angle_by_ray, axis=0)
    median_angle_by_offset = np.nanmedian(angle_by_ray, axis=0)
    mean_cosine_by_offset = np.nanmean(all_cosine, axis=0)

    ray_normal_unit, ray_normal_valid = normalize_rows(ray_normals)
    ray_normal_by_sample = np.broadcast_to(
        ray_normal_unit[:, None, :],
        (R, S, 3),
    )
    ground_truth_ray_cosine = np.sum(
        all_ground_truth_unit.reshape(R, S, 3) * ray_normal_by_sample,
        axis=2,
    )
    ground_truth_ray_angle = np.degrees(
        np.arccos(np.clip(ground_truth_ray_cosine, -1.0, 1.0))
    )
    stable_ray_sample = (
        all_ground_truth_valid.reshape(R, S)
        & ray_normal_valid[:, None]
        & (ground_truth_ray_angle <= ray_tolerance_degrees)
    )

    fully_stable_ray = np.all(stable_ray_sample, axis=1)
    filtered_valid = (
        all_valid.reshape(R, S)
        & fully_stable_ray[:, None]
    )
    filtered_cosine = np.where(filtered_valid, all_cosine, np.nan)
    filtered_angle = np.degrees(
        np.arccos(np.clip(filtered_cosine, -1.0, 1.0))
    )
    filtered_values = filtered_angle[np.isfinite(filtered_angle)]
    if len(filtered_values) == 0:
        raise RuntimeError("Gradient jump filtering removed every benchmark sample.")
    filtered_mean_angle = float(np.mean(filtered_values))
    filtered_median_angle = float(np.median(filtered_values))
    retained_fraction = float(len(filtered_values) / np.count_nonzero(all_valid))

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5), sharex="col")

    def plot_comparison_row(
        row,
        row_angles,
        row_cosine,
        row_values,
        row_mean,
        row_median,
        label,
    ):
        mean_by_offset = np.nanmean(row_angles, axis=0)
        median_by_offset = np.nanmedian(row_angles, axis=0)
        cosine_by_offset = np.nanmean(row_cosine, axis=0)

        for ray in row_angles:
            axes[row, 0].plot(
                offsets, ray, color="#2c7bb6", alpha=0.20, linewidth=1.0
            )
        axes[row, 0].plot(
            offsets, mean_by_offset, color="#d7191c", linewidth=2.5, label="Mean"
        )
        axes[row, 0].plot(
            offsets,
            median_by_offset,
            color="black",
            linestyle=":",
            linewidth=2.0,
            label="Median",
        )
        axes[row, 0].set_ylabel("Angular error [deg]")
        axes[row, 0].set_title(f"{label}: Direction Error")
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 0].legend()

        for ray in row_cosine:
            axes[row, 1].plot(
                offsets, ray, color="#2c7bb6", alpha=0.20, linewidth=1.0
            )
        axes[row, 1].plot(
            offsets,
            cosine_by_offset,
            color="#2c7bb6",
            linewidth=2.5,
            label="Mean cosine",
        )
        axes[row, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.5)
        axes[row, 1].set_ylim(-1.05, 1.05)
        axes[row, 1].set_ylabel("cos(ground truth, Bernstein)")
        axes[row, 1].set_title(f"{label}: Direction Cosine")
        axes[row, 1].grid(True, alpha=0.25)
        axes[row, 1].legend()
        axes[row, 1].set_ylim(0.5, 1.05)

        axes[row, 2].hist(
            row_values,
            bins=np.linspace(0.0, 180.0, 37),
            color="#2c7bb6",
            alpha=0.75,
        )
        axes[row, 2].axvline(
            row_mean,
            color="#d7191c",
            linestyle="--",
            label=f"mean={row_mean:.1f} deg",
        )
        axes[row, 2].axvline(
            row_median,
            color="black",
            linestyle=":",
            label=f"median={row_median:.1f} deg",
        )
        axes[row, 2].set_ylabel("Point count")
        axes[row, 2].set_title(f"{label}: Error Distribution")
        axes[row, 2].grid(True, axis="y", alpha=0.25)
        axes[row, 2].legend()

    plot_comparison_row(
        0,
        angle_by_ray,
        all_cosine,
        angle_degrees,
        mean_angle,
        median_angle,
        "Raw",
    )
    plot_comparison_row(
        1,
        filtered_angle,
        filtered_cosine,
        filtered_values,
        filtered_mean_angle,
        filtered_median_angle,
        "Fully stable rays",
    )
    for column in range(2):
        axes[1, column].set_xlabel("Outward normal offset from mesh surface [m]")
    axes[1, 2].set_xlabel("Angular error [deg]")
    axes[0, 2].set_xlabel("Angular error [deg]")

    fig.suptitle(
        "Gradient Direction Comparison\n"
        f"raw mean={mean_angle:.2f} deg; fully-stable mean={filtered_mean_angle:.2f} deg; "
        f"retained={retained_fraction:.1%}; ray tolerance={ray_tolerance_degrees:.1f} deg",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return (
        direction_rmse,
        component_correlation,
        mean_angle,
        median_angle,
        filtered_mean_angle,
        filtered_median_angle,
        retained_fraction,
    )


def save_gradient_outlier_location_plot(
    meshes,
    offsets,
    ray_points,
    ground_truth_gradient,
    bernstein_gradient,
    ray_normals,
    ray_tolerance_degrees,
    output_path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ground_truth_unit, _ = normalize_rows(ground_truth_gradient.reshape(-1, 3))
    bernstein_unit, bernstein_valid = normalize_rows(bernstein_gradient.reshape(-1, 3))
    ray_normal_unit, _ = normalize_rows(ray_normals)
    ray_count, sample_count, _ = ground_truth_gradient.shape
    ground_truth_unit = ground_truth_unit.reshape(ray_count, sample_count, 3)
    bernstein_unit = bernstein_unit.reshape(ray_count, sample_count, 3)

    ray_alignment = np.sum(
        ground_truth_unit * ray_normal_unit[:, None, :],
        axis=2,
    )
    ray_alignment_angle = np.degrees(
        np.arccos(np.clip(ray_alignment, -1.0, 1.0))
    )
    fully_stable = np.all(ray_alignment_angle <= ray_tolerance_degrees, axis=1)
    valid_at_nearest = bernstein_valid.reshape(ray_count, sample_count)[:, 0]
    selected = fully_stable & valid_at_nearest
    if not np.any(selected):
        raise RuntimeError("No fully stable rays were available for the location plot.")

    nearest_points = ray_points[selected, 0]
    nearest_ground_truth = ground_truth_unit[selected, 0]
    nearest_bernstein = bernstein_unit[selected, 0]
    cosine = np.sum(nearest_ground_truth * nearest_bernstein, axis=1)
    angular_error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    worst_order = np.argsort(angular_error)[::-1]
    highlighted = worst_order[: min(5, len(worst_order))]

    all_bounds = np.stack([mesh.bounds for mesh in meshes], axis=0)
    lower = np.min(all_bounds[:, 0], axis=0)
    upper = np.max(all_bounds[:, 1], axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.55 * float(np.max(upper - lower))

    fig = plt.figure(figsize=(15.0, 7.0))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    normalization = Normalize(
        vmin=0.0,
        vmax=max(30.0, float(np.ceil(np.max(angular_error) / 5.0) * 5.0)),
    )
    colormap = colormaps["turbo"]

    for axis, view in zip(axes, [(22, -60), (18, 125)]):
        for mesh in meshes:
            triangle_step = max(1, int(np.ceil(len(mesh.faces) / 20000)))
            collection = Poly3DCollection(
                mesh.triangles[::triangle_step],
                facecolor="#b8b8b8",
                edgecolor="none",
                alpha=0.22,
            )
            axis.add_collection3d(collection)

        axis.scatter(
            nearest_points[:, 0],
            nearest_points[:, 1],
            nearest_points[:, 2],
            c=angular_error,
            cmap=colormap,
            norm=normalization,
            s=34,
            depthshade=False,
        )
        arrow_length = max(0.012, radius * 0.18)
        for rank, point_index in enumerate(highlighted, start=1):
            point = nearest_points[point_index]
            axis.text(
                point[0],
                point[1],
                point[2],
                f" #{rank}",
                fontsize=9,
                color="black",
            )
            axis.quiver(
                *point,
                *nearest_ground_truth[point_index],
                length=arrow_length,
                color="#1a9641",
                linewidth=1.8,
                normalize=True,
            )
            axis.quiver(
                *point,
                *nearest_bernstein[point_index],
                length=arrow_length,
                color="#d7191c",
                linewidth=1.8,
                normalize=True,
            )

        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel("World X [m]")
        axis.set_ylabel("World Y [m]")
        axis.set_zlabel("World Z [m]")
        axis.view_init(elev=view[0], azim=view[1])

    axes[0].set_title("1 mm Gradient Error Locations")
    axes[1].set_title("Opposite View")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=axes,
        shrink=0.75,
        pad=0.04,
    )
    colorbar.set_label("Gradient angular error [deg]")
    fig.text(
        0.5,
        0.03,
        "Arrows at the five worst points: green = exact mesh, red = Bernstein",
        ha="center",
    )
    fig.suptitle(
        f"Fully Stable Rays at {float(offsets[0]) * 1000.0:.1f} mm "
        f"(median={np.median(angular_error):.2f} deg, max={np.max(angular_error):.2f} deg)",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    outliers = []
    for rank, point_index in enumerate(highlighted, start=1):
        outliers.append(
            {
                "rank": rank,
                "point": nearest_points[point_index],
                "error": float(angular_error[point_index]),
                "gradient_norm": float(
                    np.linalg.norm(bernstein_gradient[selected, 0][point_index])
                ),
            }
        )
    return outliers


def save_timing_boxplot(bernstein_ms, ground_truth_ms, title, batch_size, output_path):
    import inspect
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_key = "tick_labels" if "tick_labels" in inspect.signature(plt.Axes.boxplot).parameters else "labels"
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    boxplot = ax.boxplot(
        [bernstein_ms, ground_truth_ms],
        patch_artist=True,
        showmeans=True,
        **{label_key: ["Bernstein", "Mesh ground truth"]},
    )
    for patch, color in zip(boxplot["boxes"], ["#2c7bb6", "#d7191c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_yscale("log")
    ax.set_ylabel(f"Computation time for {batch_size} points [ms]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def validate_benchmark_args(args):
    positive_values = {
        "--benchmark-points": args.benchmark_points,
        "--benchmark-timing-points": args.benchmark_timing_points,
        "--benchmark-repeats": args.benchmark_repeats,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
    if args.benchmark_warmup < 0:
        raise ValueError("--benchmark-warmup cannot be negative.")
    if args.benchmark_surface_noise <= 0.0:
        raise ValueError("--benchmark-surface-noise must be greater than zero.")
    if args.benchmark_gradient_ray_tolerance <= 0.0:
        raise ValueError(
            "--benchmark-gradient-ray-tolerance must be greater than zero."
        )


def run_benchmark(args, robot_layer, weights, core, pose, q9, link_names, device):
    validate_benchmark_args(args)
    output_dir = os.path.abspath(args.benchmark_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading mesh ground truth...")
    sample_meshes = build_ground_truth_meshes(
        weights,
        robot_layer,
        pose,
        q9,
        link_names,
    )
    ground_truth_link_names = (
        list(link_names)
        if args.ground_truth_links is None
        else resolve_link_names(args.ground_truth_links)
    )
    print("Exact ground truth uses links: " + ", ".join(ground_truth_link_names))
    truth_meshes = build_ground_truth_meshes(
        weights,
        robot_layer,
        pose,
        q9,
        ground_truth_link_names,
    )
    # Timing and the outlier-location plot reference the complete robot.
    meshes = truth_meshes
    ray_sample_count = min(25, max(5, args.benchmark_points))
    ray_count = max(1, int(np.ceil(args.benchmark_points / ray_sample_count)))
    minimum_offset = min(
        max(args.benchmark_surface_noise * 0.02, 1e-5),
        args.benchmark_surface_noise * 0.5,
    )
    ray_offsets = np.linspace(
        minimum_offset,
        args.benchmark_surface_noise,
        ray_sample_count,
        dtype=np.float32,
    )
    print(
        f"Generating {ray_count} verified external normal rays with "
        f"{ray_sample_count} samples each..."
    )
    ray_points, ground_truth_distance_values, ground_truth_gradient, ray_normals = (
        make_external_benchmark_rays(
            sample_meshes,
            truth_meshes,
            ray_count,
            ray_offsets,
            args.benchmark_seed,
        )
    )
    accuracy_points = ray_points.reshape(-1, 3)

    if args.benchmark_timing_points <= len(accuracy_points):
        timing_indices = np.linspace(
            0,
            len(accuracy_points) - 1,
            args.benchmark_timing_points,
            dtype=np.int64,
        )
        timing_points = accuracy_points[timing_indices]
    else:
        timing_points, _, _ = make_external_benchmark_dataset(
            meshes,
            args.benchmark_timing_points,
            args.benchmark_surface_noise,
            args.benchmark_seed + 1,
        )

    print(f"Evaluating accuracy on {len(accuracy_points)} normal-ray points...")
    bernstein_distance, bernstein_gradient = evaluate_sdf_and_grad(
        core,
        accuracy_points,
        pose,
        q9,
        args.chunk_size,
        device,
    )
    bernstein_distance = bernstein_distance.reshape(ray_count, ray_sample_count)
    bernstein_gradient = bernstein_gradient.reshape(ray_count, ray_sample_count, 3)

    distance_plot = os.path.join(output_dir, "01_distance_comparison.png")
    gradient_plot = os.path.join(output_dir, "02_gradient_comparison.png")
    gradient_location_plot = os.path.join(
        output_dir,
        "05_gradient_1mm_outlier_locations.png",
    )
    distance_mae, distance_rmse, distance_bias, distance_correlation = (
        save_distance_comparison_plot(
            ray_offsets,
            ground_truth_distance_values,
            bernstein_distance,
            distance_plot,
        )
    )
    (
        direction_rmse,
        gradient_correlation,
        mean_angle,
        median_angle,
        filtered_mean_angle,
        filtered_median_angle,
        retained_fraction,
    ) = save_gradient_comparison_plot(
        ray_offsets,
        ground_truth_gradient,
        bernstein_gradient,
        ray_normals,
        args.benchmark_gradient_ray_tolerance,
        gradient_plot,
    )
    gradient_outliers = save_gradient_outlier_location_plot(
        meshes,
        ray_offsets,
        ray_points,
        ground_truth_gradient,
        bernstein_gradient,
        ray_normals,
        args.benchmark_gradient_ray_tolerance,
        gradient_location_plot,
    )

    print(
        f"Benchmarking {len(timing_points)} points for "
        f"{args.benchmark_repeats} measured repetitions..."
    )
    bernstein_distance_ms = measure_runtime_ms(
        lambda: evaluate_sdf(
            core,
            timing_points,
            pose,
            q9,
            args.chunk_size,
            device,
        ),
        args.benchmark_repeats,
        args.benchmark_warmup,
        device,
    )
    ground_truth_distance_ms = measure_runtime_ms(
        lambda: ground_truth_distance(meshes, timing_points),
        args.benchmark_repeats,
        args.benchmark_warmup,
        device,
    )
    bernstein_gradient_ms = measure_runtime_ms(
        lambda: evaluate_sdf_and_grad(
            core,
            timing_points,
            pose,
            q9,
            args.chunk_size,
            device,
        ),
        args.benchmark_repeats,
        args.benchmark_warmup,
        device,
    )
    ground_truth_gradient_ms = measure_runtime_ms(
        lambda: ground_truth_distance_and_gradient(meshes, timing_points),
        args.benchmark_repeats,
        args.benchmark_warmup,
        device,
    )

    distance_timing_plot = os.path.join(output_dir, "03_distance_timing_boxplot.png")
    gradient_timing_plot = os.path.join(output_dir, "04_gradient_timing_boxplot.png")
    save_timing_boxplot(
        bernstein_distance_ms,
        ground_truth_distance_ms,
        "External Distance Computation Time",
        len(timing_points),
        distance_timing_plot,
    )
    save_timing_boxplot(
        bernstein_gradient_ms,
        ground_truth_gradient_ms,
        "External Distance and Gradient Computation Time",
        len(timing_points),
        gradient_timing_plot,
    )

    print("\nBenchmark summary:")
    print(f"  Distance MAE: {distance_mae:.6f} m")
    print(f"  Distance RMSE: {distance_rmse:.6f} m")
    print(f"  Distance bias: {distance_bias:.6f} m")
    print(f"  Distance correlation: {distance_correlation:.4f}")
    print(f"  Unit-gradient component RMSE: {direction_rmse:.6f}")
    print(f"  Unit-gradient component correlation: {gradient_correlation:.4f}")
    print(f"  Mean gradient angular error: {mean_angle:.3f} deg")
    print(f"  Median gradient angular error: {median_angle:.3f} deg")
    print(
        "  Fully-stable-ray gradient error: mean={:.3f} deg, median={:.3f} deg "
        "({:.1%} samples retained)".format(
            filtered_mean_angle,
            filtered_median_angle,
            retained_fraction,
        )
    )
    print("  Worst fully stable points at the nearest sampled offset:")
    for outlier in gradient_outliers:
        point = outlier["point"]
        print(
            "    #{rank}: error={error:.2f} deg, |grad|={gradient_norm:.3f}, "
            "world=({x:.5f}, {y:.5f}, {z:.5f}) m".format(
                rank=outlier["rank"],
                error=outlier["error"],
                gradient_norm=outlier["gradient_norm"],
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
            )
        )
    print(
        "  Median distance time: Bernstein={:.3f} ms, ground truth={:.3f} ms".format(
            float(np.median(bernstein_distance_ms)),
            float(np.median(ground_truth_distance_ms)),
        )
    )
    print(
        "  Median gradient time: Bernstein={:.3f} ms, ground truth={:.3f} ms".format(
            float(np.median(bernstein_gradient_ms)),
            float(np.median(ground_truth_gradient_ms)),
        )
    )
    print("Saved plots:")
    for path in [
        distance_plot,
        gradient_plot,
        distance_timing_plot,
        gradient_timing_plot,
        gradient_location_plot,
    ]:
        print(f"  {path}")


def automatic_bounds(mesh, padding):
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    bounds[0] -= padding
    bounds[1] += padding
    return bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1], bounds[0, 2], bounds[1, 2]


def make_grid(bounds, spacing):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    xs = np.arange(x_min, x_max + 0.5 * spacing, spacing, dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * spacing, spacing, dtype=np.float32)
    zs = np.arange(z_min, z_max + 0.5 * spacing, spacing, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack((gx, gy, gz), axis=-1)
    points = coords.reshape(-1, 3, order="F")
    dims = (len(xs), len(ys), len(zs))
    return points, dims, (float(xs[0]), float(ys[0]), float(zs[0]))


@torch.no_grad()
def evaluate_sdf(core, points_np, pose, q9, chunk_size, device):
    values = []
    for start in range(0, len(points_np), chunk_size):
        chunk_np = points_np[start:start + chunk_size]
        chunk = torch.as_tensor(chunk_np, dtype=torch.float32, device=device)
        sdf = core.get_whole_body_sdf_batch(chunk, pose, q9)
        values.append(sdf.reshape(-1).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def evaluate_sdf_and_grad(core, points_np, pose, q9, chunk_size, device):
    sdf_values = []
    grad_values = []
    for start in range(0, len(points_np), chunk_size):
        chunk_np = points_np[start:start + chunk_size]
        points = torch.as_tensor(chunk_np, dtype=torch.float32, device=device).clone().detach()
        points.requires_grad_(True)
        sdf = core.get_whole_body_sdf_batch(points, pose, q9).reshape(-1)
        grad = torch.autograd.grad(sdf.sum(), points, create_graph=False, retain_graph=False)[0]
        sdf_values.append(sdf.detach().cpu().numpy())
        grad_values.append(grad.detach().cpu().numpy())
    return np.concatenate(sdf_values).astype(np.float32), np.vstack(grad_values).astype(np.float32)


def trimesh_to_pyvista(mesh):
    import pyvista as pv

    faces = np.hstack((np.full((mesh.faces.shape[0], 1), 3), mesh.faces)).astype(np.int64)
    return pv.PolyData(mesh.vertices, faces.reshape(-1))


def new_image_data():
    import pyvista as pv

    if hasattr(pv, "ImageData"):
        return pv.ImageData()
    if hasattr(pv, "UniformGrid"):
        return pv.UniformGrid()
    from pyvista.core import UniformGrid

    return UniformGrid()


def maybe_start_xvfb(pv, should_start):
    if not should_start:
        return
    try:
        pv.start_xvfb()
    except Exception as exc:
        print(f"[warn] Could not start PyVista Xvfb: {exc}")


def build_pyvista_grid(sdf_values, dims, origin, spacing):
    grid = new_image_data()
    grid.dimensions = dims
    grid.origin = origin
    grid.spacing = (spacing, spacing, spacing)
    grid.point_data["sdf_m"] = sdf_values
    return grid


def contour_for_level(grid, level):
    contour = grid.contour([float(level)], scalars="sdf_m")
    if contour.n_points == 0:
        return None
    return contour


def sample_points(points, max_count):
    if len(points) <= max_count:
        return points
    indices = np.linspace(0, len(points) - 1, max_count, dtype=np.int64)
    return points[indices]


def add_distance_shells(plotter, grid, levels, shell_opacity, surface_opacity):
    contours = {}
    for index, level in enumerate(levels):
        contour = contour_for_level(grid, level)
        if contour is None:
            print(f"[warn] No contour for SDF level {level:.4f} m; level is outside the sampled range.")
            continue
        color = SHELL_COLORS[index % len(SHELL_COLORS)]
        opacity = surface_opacity if abs(level) < 1e-9 else shell_opacity
        label = "SDF 0 cm" if abs(level) < 1e-9 else f"SDF {level * 100.0:.1f} cm"
        plotter.add_mesh(
            contour,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            label=label,
        )
        contours[float(level)] = contour
    return contours


def add_gradient_arrows(plotter, core, contours, gradient_levels, pose, q9, args, device):
    available_levels = sorted(contours.keys())
    requested = []
    for level in gradient_levels:
        closest = min(available_levels, key=lambda candidate: abs(candidate - level), default=None)
        if closest is not None and abs(closest - level) < 1e-9:
            requested.append(closest)

    if not requested:
        return

    arrows_per_level = max(1, args.max_arrows // len(requested))
    for level in requested:
        contour = contours[level]
        points = sample_points(np.asarray(contour.points, dtype=np.float32), arrows_per_level)
        _, gradients = evaluate_sdf_and_grad(
            core,
            points,
            pose,
            q9,
            chunk_size=min(args.chunk_size, 4096),
            device=device,
        )
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        if not np.any(valid):
            continue
        directions = gradients[valid] / norms[valid]
        color = SHELL_COLORS[(list(contours.keys()).index(level)) % len(SHELL_COLORS)]
        plotter.add_arrows(points[valid], directions, mag=args.arrow_scale, color=color)


def print_point_evaluations(core, points_np, pose, q9, chunk_size, device):
    if len(points_np) == 0:
        return
    sdf, grad = evaluate_sdf_and_grad(core, points_np, pose, q9, chunk_size, device)
    print("\nPoint evaluations:")
    for point, value, gradient in zip(points_np, sdf, grad):
        norm = float(np.linalg.norm(gradient))
        direction = gradient / norm if norm > 1e-8 else gradient
        print(
            "  p=({:.4f}, {:.4f}, {:.4f})  sdf={:.5f} m  grad=({:.4f}, {:.4f}, {:.4f})".format(
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(value),
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
            )
        )


def main():
    args = parse_args()
    link_names = resolve_link_names(args.links)
    eval_points = parse_eval_points(args.eval_point)
    device = choose_device(args.device)

    q7 = read_joint_states(args.joint_states_topic) if args.from_joint_states else np.asarray(args.q, dtype=np.float32)
    print(f"Using device: {device}")
    print("Using q:", np.array2string(q7, precision=6, separator=", "))
    print("SDF links:", ", ".join(link_names))

    robot_layer, weights, core = build_sdf_stack(
        device,
        link_names,
        model_override_dir=args.model_override_dir,
    )
    pose = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    q9 = q7_to_q9(q7, args.finger_position, device)

    if args.benchmark:
        run_benchmark(args, robot_layer, weights, core, pose, q9, link_names, device)
        return

    robot_mesh = selected_robot_mesh(robot_layer, pose, q9, link_names)
    bounds = tuple(args.bounds) if args.bounds else automatic_bounds(robot_mesh, args.padding)
    grid_points, dims, origin = make_grid(bounds, args.grid_spacing)
    print(
        "Grid: {} x {} x {} = {:,} points, spacing={:.3f} m".format(
            dims[0], dims[1], dims[2], len(grid_points), args.grid_spacing
        )
    )
    print(
        "Bounds: x=[{:.3f}, {:.3f}] y=[{:.3f}, {:.3f}] z=[{:.3f}, {:.3f}]".format(
            bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
        )
    )

    sdf_values = evaluate_sdf(core, grid_points, pose, q9, args.chunk_size, device)
    print(f"SDF range on grid: [{float(np.min(sdf_values)):.5f}, {float(np.max(sdf_values)):.5f}] m")
    print_point_evaluations(core, eval_points, pose, q9, args.chunk_size, device)

    if args.no_show and not args.save_screenshot:
        return

    import pyvista as pv

    off_screen = bool(args.no_show and args.save_screenshot)
    maybe_start_xvfb(pv, args.xvfb or off_screen)
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    plotter.add_mesh(
        trimesh_to_pyvista(robot_mesh),
        color="#bdbdbd",
        opacity=args.robot_opacity,
        smooth_shading=True,
        label="URDF visual mesh",
    )

    grid = build_pyvista_grid(sdf_values, dims, origin, args.grid_spacing)
    contours = add_distance_shells(
        plotter,
        grid,
        sorted(set(float(level) for level in args.levels)),
        shell_opacity=args.shell_opacity,
        surface_opacity=args.surface_opacity,
    )
    add_gradient_arrows(plotter, core, contours, args.gradient_levels, pose, q9, args, device)

    plotter.add_axes()
    plotter.add_legend()
    plotter.camera_position = "xy"
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 25
    plotter.reset_camera()

    if args.save_screenshot:
        plotter.screenshot(args.save_screenshot)
        print(f"Saved screenshot: {args.save_screenshot}")
    if not args.no_show:
        plotter.show()


if __name__ == "__main__":
    main()
