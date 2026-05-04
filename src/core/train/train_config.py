from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class Train_W:
    run: bool = False
    n_func: int = 8
    iters: int = 200
    batch_near: int = 1024
    batch_rand: int = 64


@dataclass
class Train_Sphere:
    run: bool = False
    n_points: int = 3000
    n_spheres: int = 3


@dataclass
class TrainConfig:
    links_to_train: List[str]
    fit_ellipsoid: bool = True
    debug: bool = False
    fk_matrices: Optional[List[ArrayLike]] = None

    classic: Train_W = field(default_factory=Train_W)
    sphere: Train_Sphere = field(default_factory=Train_Sphere)

    @property
    def train(self) -> bool:
        return self.classic.run or self.sphere.run
