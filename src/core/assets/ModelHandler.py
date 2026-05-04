from typing import Union

import os

import torch
from torch import load as torch_load
from torch import save as torch_save

from src.core.assets.FolderManage import FolderManage
from src.core.assets.entities.models import SphereModel, WeightsLinkModel
from src.core.assets.load_model_wrapper import load_link_sphere_model, load_link_weight_model
from src.core.assets.train_wrapper import TrainWrapper
from src.core.train.train_config import TrainConfig


def _infer_kind(d: dict) -> str:
    if "weights" in d:
        return "link_w"
    if "centers" in d and "radii" in d:
        return "link_sphere"
    raise ValueError(f"Tipo modello non riconosciuto. Chiavi trovate: {list(d.keys())[:40]}")


ModelT = Union[WeightsLinkModel, SphereModel]


class ModelHandler(FolderManage):
    def __init__(self, ws_path, extention="pt"):
        super().__init__(ws_path, "Models", extention)

    def load_model(self, model_name, device, dtype) -> ModelT:
        path = super().get_file_path(model_name, debug=True)

        if path is None:
            raise FileNotFoundError(
                "\033[91m"
                + (
                    f'Model "{model_name}" not found in {self.get_path()}.\n'
                    "No fallback is performed.\n"
                    "Fix: point the RDF workspace (ws_path) to the folder that contains the expected Models/, "
                    "or copy/rename the model files so the expected filename exists."
                )
                + "\033[0m"
            )

        model_dict = torch_load(path, map_location=torch.device(device))
        kind = _infer_kind(model_dict)

        if kind == "link_w":
            return load_link_weight_model(model_dict, device=device, dtype=dtype)
        if kind == "link_sphere":
            return load_link_sphere_model(model_dict, device=device, dtype=dtype)

        raise RuntimeError("Error in loading model")

    def save(self, model_pt):
        path = super().get_path()
        model_name = model_pt["file_name"] + model_pt.get("file_suffix", "")
        path = os.path.join(path, model_name + "." + self.extension)
        torch_save(model_pt, path)
        print("\033[92m" + f"[SAVED] FILE: {model_name} in --> {path}" + "\033[0m")

    def create_weights(self, dataset: dict, cfg: TrainConfig, device, dtype) -> dict:
        print("\033[1m" + f'CREATING MODEL FILE: {dataset["file_name"]}' + "\033[0m")
        model = TrainWrapper(device=device, dtype=dtype)
        model.debug = cfg.debug
        model.set_weight_model()
        model.initialize_model(dataset)
        dataset = model.filter_dataset(dataset)
        points_inside = dataset["near_points"][dataset["near_sdf"] < 0]
        model.fit_ellipsoid(points_inside)

        model.train(
            dataset,
            cfg.classic.n_func,
            epoches=cfg.classic.iters,
            sample_near=cfg.classic.batch_near,
            sample_rand=cfg.classic.batch_rand,
        )

        self.save(model.get_model_pt())

    def create_spheres(self, mesh, n_points, n_spheres, debug, device="cuda", dtype=torch.float32) -> dict:
        model = TrainWrapper(device=device, dtype=dtype)
        model.debug = debug
        model.set_sphere_model()
        model.train_sphere(mesh, n_points=n_points, n_spheres=n_spheres)
        self.save(model.get_model_pt())
    
