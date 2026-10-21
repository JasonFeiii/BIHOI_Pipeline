# Per-person hand calibration

Calibration creates `data/rigid/<person>/calib/restored_flat_real_scale.pt` for Hands and Final. It is separate from sequence processing.

## Prerequisites

- Synchronized calibration videos and matching `intri.yml` and `extri.yml`.
- Licensed MANO left/right models obtained from the MANO website.
- Public FMHR GCN weights and templates: `bash scripts/download_models.sh --component fmhr --yes`. These downloads are pinned and hash-checked.
- The project-specific WiLoR source variant. It remains unpublished; ordinary upstream WiLoR alone cannot run this workflow.
- The `easymocap` and `wilor` environments and the separately licensed `psbody-mesh` dependency. Review its terms before `HOI_ACCEPT_PSBODY_MESH_LICENSE=1 bash scripts/install_hand_calibration_deps.sh`.

See [reproducibility](REPRODUCIBILITY.md) and [third-party terms](../THIRD_PARTY.md).

## Run

Put videos under `data/hand_calibration/<capture>/videos/`, with camera files beside `videos/`. Intrinsics must match the extracted 640x360 frames. Specify the intended person and an explicit 10 FPS interval:

```bash
bash run_hand_calibration_args.sh \
  --data_name calibration_capture --person_name participant \
  --start_sec 0 --end_sec 0.1 --hand_side right --check
```

Remove `--check` to run. Use `--dry-run` to inspect the plan. `configs/hand_calibration.json` contains no personal mapping; supply `--person_name` or your own `--mapping`. Identities are not guessed from names.

The workflow extracts frames in an isolated workspace, detects and triangulates joints, runs FMHR, restores MANO scale, validates the ten shape coefficients, and publishes the prior. `--replace-prior` keeps a backup of an existing prior. The WebUI provides the same separate action.

FMHR uses GLX for its external psbody import. Other GPU stages retain their configured backend. Successful installation or a command plan does not establish a successful clean-machine GPU run.
