# TrackerSplat historical dependency pins

Cutoff: TrackerSplat HEAD `7e6c485570adfbb6892d595580a99e8adac9f467`,
committed at `2025-11-23T20:24:30-08:00` (`2025-11-24T04:24:30Z`).
The external commits below are the last commits at or before that cutoff:

| Package | Commit | Commit time (UTC) |
| --- | --- | --- |
| gaussian-splatting | `017fe9b04015dc71a3eb153840e7937c7fa76f77` | 2025-11-19T04:21:41Z |
| InstantSplat | `303e98cec6180ee7484782c23edef6eb990171bd` | 2025-11-02T05:03:29Z |
| reduced-3dgs | `f8d65eb171925d04dace3f68c175d609fd4ccec1` | 2025-11-15T18:42:30Z |
| ExtrinsicInterpolator | `5de703b258d65c39c394a8e7a08fa6391a66155c` | 2025-08-13T00:02:36Z |
| CoTracker | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` | 2025-01-21T21:30:41Z |

TrackerSplat's own nested gitlinks are already exact historical pins:

| Path | Commit |
| --- | --- |
| `submodules/dot` | `8c6ef1e1521eff9df8c7030f10adc57a4cddee13` |
| `submodules/featurefusion` | `2bc82c1dec3333f5b51dc772e43c1938b1e6312f` |
| `submodules/motionfusion` | `959f3efcaa6f73718643d4a451eff18f81d2e035` |
| `submodules/simple-knn` | `5735b33bd249b9ed5495b23d058cf9807e0d9a64` |

Use `.venv/bin/python -m pip`, `--no-deps`, and `--no-build-isolation` so
legacy package metadata cannot replace the protected PyTorch stack. These CUDA
packages still require an actual CUDA toolkit with `nvcc` and `CUDA_HOME`.
