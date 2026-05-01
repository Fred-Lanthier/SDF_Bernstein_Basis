# RDF

Public subset centered on `src/rdf_weights.py`.

This repository keeps the weights-based RDF inference and visualization stack:

- `RDF_Weights` for loading and evaluating `*_w.pt` models
- `train_rdf_weights.py` for training only weights models
- `visu_rdf_weights.py` for visualizing trained weights models
- mesh/point-cloud visualization helpers
- mesh loading and SDF-to-mesh utilities needed by the public API

Removed from this public tree:

- CP/TT training and decomposition code
- Panda and UR Robotnik example assets
- publication-only example scripts and benchmarks

## Quick Start

```python
from src.rdf_weights import RDF_Weights

rdf = RDF_Weights(device="cpu")
rdf.init_robot_folder("/path/to/workspace", robot_name="robot")
rdf.add_models(["link0", "link1"], robot_name="robot")
```

The workspace is expected to contain a `Models/` folder with the serialized
weight models for the links you want to load.

Examples:

```bash
python train_rdf_weights.py --ws-path /path/to/workspace --links link0 link1
python visu_rdf_weights.py --ws-path /path/to/workspace --links link0 link1
```
