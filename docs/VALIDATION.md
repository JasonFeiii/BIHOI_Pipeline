# Validation record

> Historical v1 record. For the current v2 snapshot, see [V2 changes](V2_CHANGES.md) and [v2 validation](VALIDATION_V2.md).

The interrupted packaging session's logs and generated records were inspected when work resumed. The existing repository was preserved. No checkpoint downloads, reconstruction jobs, clean GPU installations, or expensive GPU tests were repeated during final preparation.

## Successful checks

| Check | Evidence and scope |
| --- | --- |
| Packaged validation suite | **87 tests passed**: 9 packaging/command/path tests, 3 synthetic CPU stage tests, 59 media-panel tests, 9 rendering-geometry tests and 7 render CLI tests. The earlier count of 86 predates the GPU-assignment regression test. |
| Component imports and GPU visibility | All **8 stage environments passed** using their existing installed libraries. Local module origins were checked against the new source tree. Seven Torch environments detected all eight H200 devices; the WebUI environment is CPU-only. Sanitized records: `provenance/validated-imports.json`. |
| Fresh WebUI dependencies | Direct pins installed from official PyPI into a new isolated Python 3.10 virtual environment. `pip check` reported no broken requirements. The complete resolved UI package set is `environments/handobj_webui.lock.txt`. This does not test the Conda solver or the seven GPU installations. |
| Fresh WebUI source tests | 9 packaging tests, 59 media-panel tests and 7 render CLI tests passed with the new isolated WebUI dependencies. These overlap the 87 tests above; they are not additional unique tests. |
| HTTP startup | The new launcher started from an unrelated temporary working directory, served localhost `/config` with 294 components, and was stopped by the smoke harness. Both the existing and fresh WebUI dependencies passed this check during the packaging session. |
| FoundationPose C++ helper | The new package's `FoundationPose/mycpp` configured and compiled successfully against the existing FoundationPose build libraries. The resulting extension imported from the new package. Its binary and build tree are excluded from Git. |
| Synthetic video preprocessing | Temporary videos exercised camera numbering, shortest-duration selection, reference extraction, FFmpeg PTS sampling, equalization, four-ring mask-view selection and calibration synchronization. No real datasets were processed. |
| Pipeline command construction | Every selectable stage was constructed for primary and secondary takes and checked with `bash -n`; the shared-mesh dependency and 10 FPS hand/final restriction were verified. The hand GPU-selection test uses a mocked GPU inventory. |
| Syntax/configuration checks | Packaged Python parses, shell syntax checks, JSON/YAML configuration parsing and dataset path validation passed. Final syntax and staging audits are recorded in `docs/SECURITY.md`. |
| Source preservation | The prior session rechecked **754 copied original files** against their hashes recorded before edits and found **zero original source changes**. Packaged hashes were rechecked when the session resumed. See `provenance/original-preservation.json`. |

The resumed source audit identified two learned MANO joint tables stored as text. They are now excluded model assets. A small CPU-only conversion from the existing licensed MANO files reproduced both original matrices with a maximum absolute difference of **0.0**. Full checkpoints were not copied, and original files were only read.

## Unverified or unavailable work

| Area | Remaining limitation |
| --- | --- |
| Seven clean GPU environments | Observed direct dependency/build metadata is pinned, but fresh Conda solving and complete installation have not been run. Existing-environment imports do not establish clean-install reproducibility. |
| Native CUDA builds | Newly compiling PyTorch3D, nvdiffrast, gsplat, Kaolin-related components and HO-Cap MeshSDF has not been tested. Existing installed extensions imported successfully. Matching per-environment CUDA toolkits are required. |
| Model-dependent stages | SAM3/SAM2 inference, DA3 depth, MV-SAM3D reconstruction/texture, FoundationPose tracking/postoptimization, WiLoR/MANO fitting and joint refinement/export have not been executed end to end in this package. |
| Operator assets | Checkpoints, licensed MANO assets, personal beta priors, measured rig calibration and real captures are intentionally absent. Full model/calibration checks cannot pass until these are supplied. |
| WiLoR redistribution | The working adapted variant is excluded from publication under its no-derivatives license. Its private, hash-checked import was tested locally; an independent clone still needs that exact separately licensed source. See `THIRD_PARTY.md`. |
| Numerical equivalence | No newly processed capture or reference reconstruction was compared against the original pipeline. Packaging and lightweight smoke tests cannot establish numerical equivalence. |

GPU access and local sockets initially failed inside the filesystem/network sandbox; approved host-level checks succeeded. These sandbox failures were not evidence of broken hardware or package imports. Upstream Python docstring escape warnings and Gradio event-loop cleanup warnings appeared during lightweight checks without failing tests.

The full pipeline is **not yet independently proven reproducible**. This package is ready for source review and GitHub publication preparation with the above limitations disclosed. The initial commit and push remain pending the user's author details and publication confirmation.

## Reusing the checks

Run `bash scripts/validate.sh` when source changes justify retesting. Run `scripts/check_env.sh` after installing new environments/assets. The optional heavyweight procedure is in `docs/INTEGRATION.md`; it is intentionally separate from the default lightweight tests. `scripts/audit_source.py --staged` checks the exact Git index before committing.

## Hand Calibration integration

The later calibration integration ran 16 new lightweight calibration tests and reran 9 existing packaging/WebUI regression tests. A real 31-camera right-hand run at the exact [0, 0.1) interval and 10 FPS completed GCN inference, real-scale restoration and private per-person prior publication. The existing Hands loader accepted the result. See [the implementation, environment issue resolved, preservation checks and remaining limits](HAND_CALIBRATION.md). This does not establish a fresh native installation or a new full Hands/Final GPU run.
