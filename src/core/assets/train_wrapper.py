import numpy as np
import torch

from src.visu.debug_visu_utils import show_mesh, plot_ellipsoid
from src.core.train.weight_train import BernsteinWeightsTrain
from src.core.train.sphere_train import build_spheres_from_mesh
from src.core.math.fit_ellipsoid import compute_ellipsoid_parameters
from src.utils.sdf_utils import sdf_to_mesh
from src.visu.pv_mesh_spheres_viewer import show_mesh_with_spheres
from src.core.assets.entities.models import WeightsLinkModel, SphereModel


class TrainWrapper():
    def __init__(self, n_func=8, domain_min=-1, domain_max=1, device='cuda', dtype=torch.float32):
        self._domain_min = domain_min
        self._domain_max = domain_max
        self.n_func = n_func
        self._berstein_train = BernsteinWeightsTrain(
            n_func=n_func,
            domain_min=domain_min,
            domain_max=domain_max,
            device=device,
            dtype=dtype,
        )
        self.device = device
        self.dtype = dtype

        self.name = 'generc_model'
        self.instance_type = 'Model'
        self.debug = False
        self.model_pt = None

    def set_weight_model(self):
        self.model_pt = WeightsLinkModel()

    def set_sphere_model(self):
        self.model_pt = SphereModel()

    def _to_device_dtype(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=self.device, dtype=self.dtype)
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    def set_scale_and_offset(self, scale, translate):
        self.scale_factor = self._to_device_dtype(scale)
        self.centroid_offset = self._to_device_dtype(translate)

    def initialize_model(self, dataset: dict):
        self.model_pt.file_name = dataset['file_name']
        self.model_pt.domain_min = dataset['sdf_domain_min']
        self.model_pt.domain_max = dataset['sdf_domain_max']
        self._domain_min = dataset['sdf_domain_min']
        self._domain_max = dataset['sdf_domain_max']
        self.model_pt.scale_factor = dataset['mesh_scale_factor']
        self.model_pt.centroid_offset = dataset['mesh_centroid_offset']

    def fit_ellipsoid(self, points_inside: np.ndarray):
        N = 5000

        if len(points_inside) > N:
            indices = np.random.choice(len(points_inside), size=N, replace=False)
            points_inside = points_inside[indices]
        A, c, eigen_vector = compute_ellipsoid_parameters(points_inside)
        self.model_pt.axes_ellipsoid = A
        self.model_pt.center_ellipsoid = c
        self.model_pt.eigen_vector_ellipsoid = eigen_vector

        if self.debug:
            plot_ellipsoid(
                center=self.model_pt.center_ellipsoid,
                axes_lengths=self.model_pt.axes_ellipsoid,
                eigenvectors=self.model_pt.eigen_vector_ellipsoid,
                points=points_inside,
            )

    def filter_dataset(self, dataset: dict):
        near_points = dataset['near_points']
        near_sdf = dataset['near_sdf']
        near_dist = np.linalg.norm(near_points, axis=1)
        near_mask = near_dist <= 1

        dataset['near_points'] = near_points[near_mask]
        dataset['near_sdf'] = near_sdf[near_mask]

        return dataset

    def train(self, dataset: dict, n_func, epoches: int = 200, sample_near: int = 1024, sample_rand: int = 64):
        if not isinstance(dataset, dict):
            raise ValueError("[91mThe dataset must be a dictionary[0m")

        for key in ['near_points', 'near_sdf', 'query_points', 'query_sdf', 'mesh_scale_factor', 'mesh_centroid_offset', 'file_name']:
            if key not in dataset.keys():
                raise ValueError("[91m" + f"The dataset must have the key: {key}" + "[0m")

        print('[95m Training Bernstain function for ' + dataset['file_name'] + ' Using Device: ' + self.device + '[0m')

        self._berstein_train.set_points_domain(dataset['sdf_domain_min'], dataset['sdf_domain_max'])
        self._berstein_train.set_number_of_functions(n_func)

        self.weights = self._berstein_train.train(
            dataset['near_points'],
            dataset['near_sdf'],
            dataset['query_points'],
            dataset['query_sdf'],
            epoches=epoches,
            sample_near=sample_near,
            sample_rand=sample_rand,
        )

        self.model_pt.n_func = int(n_func)
        self.model_pt.weights = self.weights

        if self.debug:
            mesh = sdf_to_mesh(
                weights=self.weights,
                nbData=128,
                domain_max=self._domain_max,
                domain_min=self._domain_min,
                scaling_factor=self.model_pt.scale_factor,
                centroid_offset=self.model_pt.centroid_offset,
                basis_function_from_3Dpoints=self._berstein_train.basis_function_from_3Dpoints,
            )
            show_mesh(mesh)

    def train_sphere(self, mesh, n_points: int = 1000, n_spheres: int = 50):
        sphere_dict = build_spheres_from_mesh(mesh, n_points=n_points, n_spheres=n_spheres)

        self.model_pt.centers = sphere_dict['centers']
        self.model_pt.radii = sphere_dict['radii']
        self.model_pt.file_name = mesh.name

        if self.debug:
            show_mesh_with_spheres(
                mesh_path=mesh.path,
                centers=self.model_pt.centers,
                radii=self.model_pt.radii,
                mesh_opacity=0.9,
                sphere_opacity=0.6,
            )

    def get_model_pt(self):
        return self.model_pt.to_dict()
