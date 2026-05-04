from src.core.assets.entities.models import SphereModel, WeightsLinkModel

import numpy as np
import torch

_C_METHOD = {
    "W3D": "\033[38;2;120;220;120m",
    "SPHERE": "\033[38;2;210;210;210m",
}
_C_RESET = "\033[0m"


def _method_print(method: str, text: str):
    print(_C_METHOD.get(method, "\033[38;2;255;165;0m") + text + _C_RESET)


def _log_loading(method: str, file_name: str, details: str = ""):
    _method_print(method, f"LOADING[{method}] MODEL: {file_name}")
    if details:
        _method_print(method, details)


def _to_tensor(x, device, dtype) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, (np.ndarray, list, float, int)):
        return torch.as_tensor(x, device=device, dtype=dtype)
    return x


def load_link_weight_model(model_dict, device="cuda", dtype=torch.float32) -> WeightsLinkModel:
    d = dict(model_dict)

    for k in [
        "scale_factor",
        "centroid_offset",
        "center_ellipsoid",
        "axes_ellipsoid",
        "scales_ellipsoid",
        "eigen_vector_ellipsoid",
        "weights",
    ]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)

    _log_loading("W3D", d["file_name"], f"Number of BASIS functions: {d['n_func']}")
    return WeightsLinkModel.from_dict(d)


def load_link_sphere_model(model_dict, device="cuda", dtype=torch.float32) -> SphereModel:
    d = dict(model_dict)

    for k in ["centers", "radii"]:
        if k in d:
            d[k] = _to_tensor(d[k], device, dtype)

    _log_loading("SPHERE", d["file_name"])
    return SphereModel.from_dict(d)
