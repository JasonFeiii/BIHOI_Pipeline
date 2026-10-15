# Hand Calibration implementation and validation

Calibration is a separate per-collector setup action in the **10 FPS version**. It preserves the five normal processing stages and never joins their automatic job graph.

## Original execution chain

`run_hand_calibration_args.sh` originally calls:

1. `rename_videos_31cams.py --strict-calibration`: copy the original 31 videos into an isolated workspace as `1.mp4` ... `31.mp4`, following `CALIB_CAMERA_ORDER` exactly (or preserving validated complete numeric/legacy rpizero indices). The source videos are never modified. Ambiguous, extra, or missing cameras are rejected.
2. `rename_videos_extract_frames_ffmpeg_pts_compat_resize.py --skip-rename`: select the exact half-open time interval, extract 640×360 images at 10 FPS. No second renaming takes place.
3. `WiLoR/demo.py`: hand-2D detections in `annots_fallback/<camera>/000000.json` (private licensed local import).
4. `apps/demo/mv1p713.py`: multiview triangulation and its existing fallback-2D recovery, with `handr/manor` for right or `handl/manol` for left. The actual entrypoint writes skeletons; it does not call its defined SMPL fitting helper.
5. `FMHR/multihands_mano.py` → `train_gcn.run_infer_json` → `models/gcn.py` and its transitive mesh/math utilities: reconstruct the original GCN MANO parameters from frame-zero joints.
6. `FMHR/restore_gcn_mano_flat_real_scale_v3_all_bones.py`: reconstruct posed MANO geometry, estimate scale from all 20 skeleton bone segments, generate a flat hand, and solve the original regularized least-squares real-scale beta fit.
7. The new publication helper validates the original result dictionary before saving the per-person prior.

The original `--pose_batch_size` and Sapiens path variables were unused; there is no active Sapiens dependency in this chain. There is no new calibration estimator, substitute detector, or mean-beta fallback.

## Calibration camera naming

The input folder is `<Hand Calibration Root>/<capture>/videos/`. Supply exactly **31 `.mp4` videos** named after the calibrated cameras. The conversion is 1-based: `node13_cam0.mp4` → `1.mp4`, `node13_cam1.mp4` → `2.mp4`, `node25_cam1.mp4` → `3.mp4`, ... `node47_cam1.mp4` → `31.mp4`. The full list is maintained in `pipeline/rename_videos_31cams.py` as `CALIB_CAMERA_ORDER`.

For compatibility, a complete `1.mp4` ... `31.mp4` or `rpizero000_test*.mp4` ... `rpizero030_test*.mp4` capture is also accepted. In strict calibration mode, filename guessing, unknown cameras and missing cameras are **errors**, not silently sorted. `intri.yml` and `extri.yml` must refer to the same numbered camera order. The WebUI command preview lists the rename stage; the source capture remains read-only.

## Person mapping and result contract

The original shell has no `hands_ck` → `ck` mapping. Its restoration would write `hands_ck/calib/`. Original `hand.md` lines 390, 400 and 407 use dataset `rigid/ck/hands_ck` and hand prior `rigid/ck`; that evidence is recorded in `provenance/hand-calibration.json` and the configurable `configs/hand_calibration.json`. Other names require an explicit mapping or `--person_name`; suffixes are never guessed.

Output: `$HOI_DATA_ROOT/rigid/<person_name>/calib/restored_flat_real_scale.pt`. The original `real_scale_betas10_lstsq` and `original_betas10` are finite CPU float tensors of shape `(10,)`; `real_scale` is finite and positive. Original pose/mesh/joint fields are preserved. Publication adds capture/person/side/interval/FPS metadata and is atomic. An existing prior is refused unless `--replace-prior` is provided, in which case its bytes are retained in a private backup.

EasyMocap's existing `load_betas_from_restored_pt` selects `real_scale_betas10_lstsq`; `set_fixed_betas_to_params` writes it to all frame shapes. Final optimization receives these MANO shapes, and its standardized export reader accepts the same ten beta values. As in the original pipeline, this is one per-person prior shared by subsequent jobs; the calibrated side is recorded rather than changing that contract.

## Validation performed

- **Calibration tests** cover argument/side/time propagation, verified mapping, rejection of unmapped names and path escapes, fixed 10 FPS, invalid intervals, isolated input copies and GCN paths, missing exact frame counts, real FFmpeg extraction from a generated video, beta validation, atomic publication/backups, the existing Hands loader and Final export reader, original scale math, FMHR's GLX import setting, and the WebUI preview's actual shell/Python round trip.
- **9 existing packaging/WebUI regression tests** passed after adding the separate UI action. They include UI construction, all ordinary stage commands, shared-take dependencies and existing FPS/GPU assignment checks. The prior 87-test packaging record remains historical; heavyweight processing tests were not repeated.
- Python and shell syntax and the final exact Git-index publication audit are checked before committing. Original numerical function/class ASTs match for restoration (apart from CLI/main path plumbing) and GCN inference (apart from the explicit output path). The unused fixed-bone-length table from a private measured hand is omitted; the original active triangulation branch disables this constraint.
- **Real calibration passed** using the available `hands_ck` capture, the exact interval **[0, 0.1)**, **right** side and one NVIDIA H200 GPU. Extraction produced **31 camera folders with exactly one frame each**. WiLoR produced all 31 annotation files with zero reported failures. The original fallback triangulation recovered **all 21 joints**, with no remaining missing joints. GCN inference, MANO restoration and prior publication completed. The published beta vector was loaded by the existing Hands loader and applied to frame shapes.
- The real run exposed an external `psbody-mesh` import failure under EGL: its unused MeshViewer loads GLUT font symbols. The launcher now uses GLX only for FMHR, and EGL for other stages. Validation resumed at FMHR with the already verified extraction/detections/triangulation, without changing the requested interval or repeating those GPU stages.
- The original source/script/document, capture inputs and existing person prior were fingerprinted; **all 47 checked originals remain unchanged** during the reported prior validation. The calibration renaming stage now copies input videos directly into the ignored run workspace and writes numbered output names, without modifying the original capture.

The real result remains private under the independent project's `data/rigid/ck/calib/`; intermediate meshes, GCN outputs, detections and logs remain ignored. No weights, template meshes, beta vectors or real capture images are published. Full raw logs are retained locally, not versioned.

## Limits

This establishes execution and beta format compatibility for one right-hand capture, not calibration accuracy for every collector. The least-squares real-scale beta vector remains the original algorithm's approximation; it is not a new global-scale parameterization. Real left-hand execution, fresh builds of the optional native psbody dependency, and new full hand/object reconstruction or final GPU optimization using the new prior were not run. Synthetic tests verify the downstream loader/export contracts; they do not establish full-stage numerical accuracy. All required models were already available, and no environment packages were changed for validation.

The adapted WiLoR detector remains excluded under its no-derivatives license. A fresh clone needs the exact separately licensed local source import, MANO registration/assets, FMHR templates/checkpoints and the external research-use psbody dependency. Clean installation and end-to-end reproducibility of the entire repository remain subject to the earlier documented limitations.
