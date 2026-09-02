# SpaTrackerV2

## Role

Offline monocular RGB or RGBD 3D point tracking inference. The upstream README
does not provide a general released training/evaluation workflow.

## Entrypoints

Run from `SpaTrackerV2/` with the root environment referenced explicitly:

```bash
../.venv/bin/python inference.py \
  --data_type RGB --data_dir <explicit-input> --video_name <name> --fps 3
```

`app.py` starts a Gradio application and is not a minimal validation command.
`evaluation/eval_dyn.py` contains environment-specific paths and is not a general
harness validation entrypoint.

## Harness rules

- Preflight Python, torch, CUDA, network/cache availability, and input paths.
- Start with a short video, sparse frames, larger `fps`, smaller grid, and fewer
  visual-odometry points when supported.
- The code may download and load two large Hugging Face models and use
  CUDA/bfloat16; treat this as an OOM/download risk.
- Upstream output uses `<data_dir>/results/result.npz`; protect existing output
  with a unique input/output arrangement or first add an explicit output option.
- Its nested `examples` submodule may be uninitialized; do not initialize or
  download it without a task that requires it.

