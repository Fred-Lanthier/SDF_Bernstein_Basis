import time

import mesh_to_sdf  # noqa: F401
import numpy as np
import torch

from panda_test.panda_fk import fk_panda_two_robots
from src.rdf_weights import RDF_Weights

import rospkg

rospack = rospkg.RosPack()


WS_PATH = rospack.get_path('vision_processing') + '/third_party/SDF_Bernstein_Basis/panda_test'
ROBOT_NAME = 'panda_robot'

BASE_LINK_NAMES = [
    'panda_link0',
    'panda_link1',
    'panda_link2',
    'panda_link3',
    'panda_link4',
    'panda_link5',
    'panda_link6',
    'panda_link7',
    'panda_hand',
    'panda_leftfinger',
    'panda_rightfinger',
    'fork_tip'
]
NAMESPACES = ['p1_', 'p2_']
LINK_NAMES = [ns + ln for ns in NAMESPACES for ln in BASE_LINK_NAMES]

RUN_GD_VISU = False
GD_ITERS = 10

MESH_COLOR = '#C7C7C7'
MESH_OPACITY = 1.0
RDF_COLOR = '#C7C7C7'
RDF_OPACITY = 1.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    rdf = RDF_Weights(device=device)
    rdf.init_robot_folder(WS_PATH, robot_name=ROBOT_NAME)

    # Load unique models ONLY ONCE (no namespace)
    rdf.add_models(link_names=BASE_LINK_NAMES, robot_name=ROBOT_NAME)
    rdf.add_mesh(link_names=BASE_LINK_NAMES, robot_name=ROBOT_NAME)

    # Batch parameters for inference (use base names)
    rdf.set_ordered_batch_params(link_names=BASE_LINK_NAMES)

    points = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=dtype,
        device=device,
    )
    batch_points = points.unsqueeze(0).repeat(len(BASE_LINK_NAMES), 1, 1)

    points_rdm = torch.rand((550, 3), dtype=dtype, device=device)
    for _ in range(3):
        rdf.inference_link(points=points_rdm, link_name=BASE_LINK_NAMES[3], get_grad=False, get_min=False)
    time_ = time.time()
    sdf, grad, _ = rdf.inference_link(points=points_rdm, link_name=BASE_LINK_NAMES[3], get_grad=False, get_min=False)
    print('TIME SDF GRAD 550 POINTS:', time.time() - time_)

    sdf, grad, points = rdf.inference_link_batch(points=batch_points, get_grad=False, get_min=True)
    print('SDF BATCH LINK')
    print(sdf)
    print('GRAD BATCH LINK')
    print(grad)

    q = np.array([0.0, -0.7, -0.0, -2.3, -0.0, 1.57, 0.0], dtype=np.float32)
    theta = -np.pi / 10.0
    y_offset = 0.4

    rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    base_a = np.eye(4, dtype=np.float32)
    base_a[:3, :3] = rz
    base_a[:3, 3] = np.array([0.0, -y_offset / 2.0, 0.0], dtype=np.float32)

    base_b = np.eye(4, dtype=np.float32)
    base_b[:3, :3] = rz
    base_b[:3, 3] = np.array([0.0, +y_offset / 2.0, 0.0], dtype=np.float32)

    fk_two = fk_panda_two_robots(
        q_a=q,
        q_b=q,
        base_a=base_a,
        base_b=base_b,
        include_gripper=True,
    )
    if fk_two.shape[1] != len(BASE_LINK_NAMES):
        raise ValueError(
            f'FK returned {fk_two.shape[1]} links, expected {len(BASE_LINK_NAMES)} '
            f'for {BASE_LINK_NAMES}.'
        )

    fw_dict = {}
    for idx_ns, ns in enumerate(NAMESPACES):
        for i, base_name in enumerate(BASE_LINK_NAMES):
            fw_dict[ns + base_name] = torch.as_tensor(fk_two[idx_ns, i], device=device, dtype=rdf.dtype)

    if RUN_GD_VISU:
        points_for_gd = (
            torch.tensor([[1.0, 0.0, 0.5]], dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(len(BASE_LINK_NAMES), 1, 1)
        )
        rdf.visualize_gradient_descent(
            initial_points=points_for_gd,
            forward_dict=fw_dict,
            num_of_iteration=GD_ITERS,
            mesh_color=MESH_COLOR,
            mesh_opacity=MESH_OPACITY,
            rdf_color=RDF_COLOR,
            rdf_opacity=RDF_OPACITY,
        )

    mesh_links = [name for name in fw_dict.keys() if name.startswith('p1_')]
    rdf_links = [name for name in fw_dict.keys()] # Visualize all links as RDF

    # Custom loop to handle instancing without duplicating models
    from src.utils.MeshUtils import trimesh_to_pyvista
    from src.visu.plots_3d import RobotSene
    
    rdf._visualizer = RobotSene()
    
    # 1. Add Meshes for first robot
    for ln in BASE_LINK_NAMES:
        tf = fw_dict['p1_' + ln]
        mesh = rdf.to_mesh(ln, type='trimesh')
        rdf._visualizer.add_mesh(
            trimesh_to_pyvista(mesh, tf),
            opacity=MESH_OPACITY,
            color=MESH_COLOR,
        )

    # 2. Add RDFs for second robot (REUSING SAME MESHES)
    for ln in BASE_LINK_NAMES:
        tf = fw_dict['p2_' + ln]
        mesh = rdf.to_mesh(ln, type='trimesh') # Cache hit here
        rdf._visualizer.add_mesh(
            trimesh_to_pyvista(mesh, tf),
            opacity=RDF_OPACITY,
            color=RDF_COLOR,
        )
    
    # 3. Superpose the real robot CAD meshes in red on top of Robot 1 RDF
    for ln in BASE_LINK_NAMES:
        tf = fw_dict['p1_' + ln]
        cad_mesh = getattr(rdf, ln + rdf.mesh_extension).mesh
        rdf._visualizer.add_mesh(
            trimesh_to_pyvista(cad_mesh, tf),
            opacity=0.4,
            color='red',
        )
    
    rdf._visualizer.show()


if __name__ == '__main__':
    main()
