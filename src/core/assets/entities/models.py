from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

ArrayLike = Union[float, np.ndarray, torch.Tensor, list]


@dataclass
class BaseLinkModel:
    file_name: str = ""
    file_suffix: str = field(default="", init=False)
    domain_min: float = 0.0
    domain_max: float = 0.0
    scale_factor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    centroid_offset: ArrayLike = 0.0

    center_ellipsoid: torch.Tensor = field(default_factory=lambda: torch.empty((3,)))
    axes_ellipsoid: Optional[Any] = None
    scales_ellipsoid: Optional[Any] = None
    eigen_vector_ellipsoid: Optional[Any] = None

    n_func: Optional[int] = None

    device: str = field(default="cpu", init=False)
    dtype: torch.dtype = field(default=torch.float32, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        allowed = {f.name for f in fields(cls) if f.init}
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)


@dataclass
class WeightsLinkModel(BaseLinkModel):
    file_suffix: str = field(default="_w", init=False)
    weights: torch.Tensor = field(default_factory=lambda: torch.empty((0,)))

    def to(self, device, dtype):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cur = torch.device(self.device) if isinstance(self.device, str) else self.device

        if (cur == dev) and (self.dtype == dtype):
            return self

        self.weights = self.weights.to(device=device, dtype=dtype)
        self.centroid_offset = self.centroid_offset.to(device=device, dtype=dtype)
        self.scale_factor = self.scale_factor.to(device=device, dtype=dtype)
        self.device = str(dev)
        self.dtype = dtype
        return self


@dataclass
class SphereModel:
    centers: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3)))
    radii: torch.Tensor = field(default_factory=lambda: torch.empty((0,)))
    file_name: str = ""
    file_suffix: str = field(default="_spheres", init=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        allowed = {f.name for f in fields(cls) if f.init}
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)
