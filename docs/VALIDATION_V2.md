# V2 validation — 2026-10-15

Tests ran against the new v2 source tree, using the existing installed environments
and private input/weight files outside Git. No capture data or test output is published.

| Check | Result |
| --- | --- |
| Packaging, configuration, stage commands and Gradio construction | 9 tests passed |
| CPU preparation and actual FFmpeg sampling | 3 tests passed |
| Separate hand calibration and camera ordering | 18 tests passed |
| Qualitative WebUI panel | 59 tests passed |
| Single-sequence export helpers | 9 tests passed |
| Render/export validation | 7 tests passed |
| WAPR candidate flow, geometry, SOR, recovery, output history and live preview | 36 tests passed |
| Demo HTTP endpoints, manual query points, lossless export and video timing | 30 tests passed; optional MoSE3 source was available |
| Browser: main Gradio UI | WAPR selection, live-preview section and hand calibration present; no JavaScript errors |
| Browser: WAPR and MoSE3 demo pages | Navigation, forms, validation, reset and mobile width passed; no JavaScript errors |
| WAPR imports and CUDA | Passed; imports resolve to the bundled v2 source |
| WAPR installer | Dry run passed; no environment installation performed |
| Publication index | Python/shell syntax and source/data/credential-pattern audit passed |

Total: **171 unit/integration tests passed**. The browser checks and GPU checks are additional.

## Real GPU checks

On one NVIDIA H200, with the PyTorch backend and CUDA/nvdiffrast renderer:

- A private calibrated RGB-D frame supplied its largest valid-mask camera and
  triangulated center to all 96 initial rotations. Maximum initial center error
  was `1.45e-8 m`.
- All 96 poses received masked WAPR (3 updates), SAPR (2 updates), and one
  96-candidate WBPS group. The next-frame call used a three-candidate scoring group.
- The same packaged source performed FoundationPose registration with exactly
  240 rotations and 5 refinement iterations, then the complete six-candidate
  WAPR/SAPR scoring step. The resulting pose was finite with a valid rotation.
- The independent preview process published an image and metadata. It fused
  30 calibrated cameras, filtered 610,942 masked-depth points to 601,617 using SOR,
  and displayed sparse red points with the emphasized green pose mesh.

## Limits

This validates the targeted tracking and packaging paths. It does not establish
accuracy against ground truth, full-sequence throughput, a new end-to-end Hands/Final
GPU run, or a clean installation on another machine. Existing FoundationPose native
binaries, installed dependencies and model weights were reused through ignored local
paths. A fresh install still requires the documented CUDA builds and licensed assets.
TensorRT engines were not tested; PyTorch is the default.

Gradio emitted existing event-loop cleanup warnings, and upstream model/video libraries
emitted deprecation/resource warnings. These did not fail the checks. Browser readiness
was checked using rendered controls because the live UI polls continuously.

Reproduce the non-GPU suite with `bash scripts/validate.sh` after installing the required
environments. Run `bash scripts/check_env.sh --component wapr --skip-models` to verify
local imports/CUDA, and `python scripts/audit_source.py --staged` before publication.
