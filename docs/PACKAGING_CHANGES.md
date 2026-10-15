# Changes from the working source snapshot

> Historical v1 record. For the current v2 snapshot, see [V2 changes](V2_CHANGES.md) and [v2 validation](VALIDATION_V2.md).

Processing math, model classes, loss functions, default reconstruction arguments, fitting/RANSAC/Kalman logic, and output schemas are preserved. Original file hashes are recorded separately from packaged hashes so modifications are reviewable.

- Project-relative code roots replace fixed deployment paths; data roots resolve through the existing ignored alias mechanism. Model/calibration/env paths come from local configuration.
- Each of eight Conda environments resolves to an independent prefix by default; explicit per-environment overrides are supported. Standardized export no longer contains a fixed user interpreter path.
- The four active shell wrappers locate the pipeline root from their own location and honor configured Conda prefixes and MANO directory. Legacy source examples/comments with machine-specific paths were removed or generalized.
- The hand shell filters its nvidia-smi candidates to GPU_PICK/CUDA_VISIBLE_DEVICES assigned by the scheduler, and honors its VRAM budget. It previously could select other GPUs despite the outer reservation. Triangulation/fitting logic is unchanged.
- The original hand/final scripts are 10 FPS presets. The UI now rejects incompatible FPS for those stages instead of silently producing inconsistent timing. Final export still resamples to 30 FPS with its original first-500 default.
- Default bind is localhost and sharing is off; the launcher rejects configured public unauthenticated access. Existing authenticated sharing remains supported. File serving is limited to configured data/run roots and explicit additional allowed roots, rather than all of /mnt.
- Dataset names `.` and `..` are rejected in addition to the existing name checks.
- The qualitative panel and all runner dependencies were included. Its helper paths and MANO rendering settings are portable. Rendering uses the operator's EGL libraries rather than an unpublished copy of OS shared objects.
- Gradio 6 column_count/CSS launch compatibility is applied while preserving the UI features; the deprecated row-count tuple is replaced with the equivalent dynamic default. Public unauthenticated CLI launch is also rejected.
- MV-SAM3D respects an explicit CUDA_HOME and uses the original MoGe model from the configured local checkpoint. Missing MoGe fails clearly instead of silently downloading at inference time.
- Minimal package build metadata replaces MV-SAM3D's missing development requirements files. DA3 metadata no longer requires its unused pre-commit tool or optional xformers accelerator; actual inference code is unchanged. Environment package pins are maintained separately.
- The modified WiLoR source is installed privately with a byte-preserving hash check rather than publicly redistributed. The corresponding hand stage remains present and fails on missing prerequisites. No proxy detector or solver is added.
- HO-Cap is an external pinned install; all licensing text and native build requirements are documented. No original installer that deletes build products is run.
- No original source/data file is rewritten or deleted, no existing Git history is rewritten, and no publication happens during packaging.

Final preparation excludes learned MANO regressor tables even though their format is text. A local conversion from licensed MANO models preserves their exact matrix values and the original side-specific fingertip selection. Model-directory symlinks are ignored as paths as well as directories; mixed-case asset extensions are covered. The exporter dependency report now lists the actual included fallback.

Hand Calibration adds the original FMHR import closure and EasyMocap `mv1p713.py`. Dataset/model/output paths are configurable and GCN outputs are isolated per run. Numeric inference and restoration functions are preserved; AST comparison against originals passed. The unused, disabled fixed-bone-length table in `mv1p713.py` contained private measured values and is omitted; its active calibration branch is unchanged. Per-person publication adds validation/metadata and protects existing priors with optional backups. FMHR alone uses GLX for the external psbody import. WiLoR calibration detection remains an exact private import. The separate UI action leaves stage keys/order unchanged.
