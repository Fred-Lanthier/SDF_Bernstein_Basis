import os
from typing import Union

import torch
from torch import load as torch_load
from torch import save as torch_save

from src.core.assets.FolderManage import FolderManage
from src.core.assets.entities.models import SphereModel, WeightsLinkModel
from src.core.assets.load_model_wrapper import load_link_sphere_model, load_link_weight_model


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

        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")

        model_dict = torch_load(path, map_location=target_device)
        kind = _infer_kind(model_dict)

        if kind == "link_w":
            return load_link_weight_model(model_dict, device=target_device, dtype=dtype)
        if kind == "link_sphere":
            return load_link_sphere_model(model_dict, device=target_device, dtype=dtype)

        raise RuntimeError("Error in loading model")

    def save(self, model_pt):
        path = super().get_path()
        model_name = model_pt["file_name"] + model_pt.get("file_suffix", "")
        path = os.path.join(path, model_name + "." + self.extension)
        torch_save(model_pt, path)
        print("\033[92m" + f"[SAVED] FILE: {model_name} in --> {path}" + "\033[0m")
