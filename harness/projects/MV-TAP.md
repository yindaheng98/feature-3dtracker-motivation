# MV-TAP

## Role

Multi-view 3D point tracking for datasets including DexYCB, Panoptic, Kubric, and
Harmony4D, with view-sampling utilities.

## Entrypoints

Run evaluation from `MV-TAP/` with explicit inputs and output:

```bash
../.venv/bin/python experiment.py \
  mode=eval \
  ckpt_path=<checkpoint> \
  datasets.data_root=<explicit-data-root> \
  experiment_path=../output/harness-runs/<experiment-id>
```

View-sampling helpers are under `scripts/sample_*_views.py`.

## Harness rules

- Full training (DDP/W&B) and full evaluation are expensive; create a bounded
  single-dataset or small-sample smoke path before using them.
- Pass explicit absolute/normalized dataset and destination paths. Do not assume
  the submodule's missing default dataset directory maps to root `data/`.
- Do not use a sampling `--overwrite` option without explicit authorization; it
  may delete the destination. Prefer symlinks to copying large image trees.
- Avoid W&B or other network side effects unless requested.
- The documented Python/PyTorch versions differ from the root environment;
  preflight imports before running.

