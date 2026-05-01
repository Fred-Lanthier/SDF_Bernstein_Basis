from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

import torch


def _as_tensor(value, device=None, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if device is not None or dtype is not None else value
    return torch.as_tensor(value, device=device, dtype=dtype)


@dataclass
class BaseLinkModel:
    file_name: str = ""
    file_suffix: str = field(default="", init=False)
    domain_min: float = 0.0
    domain_max: float = 0.0
    scale_factor: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    centroid_offset: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
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
    weights: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def to(self, device, dtype):
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cur = torch.device(self.device) if isinstance(self.device, str) else self.device
        if cur == dev and self.dtype == dtype:
            return self

        self.weights = _as_tensor(self.weights, device=dev, dtype=dtype)
        self.centroid_offset = _as_tensor(self.centroid_offset, device=dev, dtype=dtype)
        self.scale_factor = _as_tensor(self.scale_factor, device=dev, dtype=dtype)
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
