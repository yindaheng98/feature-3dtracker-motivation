# LAPA

## Role

Transformer-based multi-camera point tracking with TAPVid-3D-MC,
PointOdyssey-MC, and joint checkpoints.

## Entrypoints

- `inference_lapa.py`: inference for one scene and selected cameras.
- `evaluate_lapa.py`: bounded evaluation.
- `train_lapa.py`: TAPVid-3D, PointOdyssey, or joint training.
- `python -m lapa.models.lapa`: built-in dependency-free synthetic model smoke.

## Harness rules

- Use the root `.venv`; the declared Python imports are covered by the shared
  manifest and pass on Torch 2.6.0+cu124.
- Real inference requires a matching `lapa.pt`, multi-camera metadata, per-camera
  HDF5 feature caches, and track NPZ files. Do not infer those paths from the
  large root `data/` tree.
- Feature precomputation may download DINOv2 or CoTracker through Hugging Face or
  Torch Hub. Treat that as an explicit model-download action.
- Start with `--max_points` and `--num_frames` bounded to a small smoke before a
  full evaluation or training run.
