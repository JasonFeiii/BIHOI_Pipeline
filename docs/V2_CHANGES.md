# V2 source snapshot

Based on `JasonFeiii/Hand-Object-Interaction-Pipeline` at
`3c7c9ea`, retaining the portable installation, calibrated camera ordering, separate
hand calibration, seven-stage UI and hand/object export workflow. V2 is a new source
snapshot; measured calibration files uploaded to v1 are not included.

## Tracking and preview

The current local WebUI and FoundationPose changes were merged against their shared
original source, retaining v1 packaging adaptations. Added `webui/wapr_tracking.py`,
`foundationpose_reinit.py`, `masked_depth.py`, `wapr_parallel.py`, `wapr_multiview.py`,
`tracking_preview.py`, and `tracking_run.py`. The active runtime uses persistent
per-camera workers; the batched helper is retained for comparison.

[Tracking documentation](POSE_TRACKING_V2.md) specifies initialization, ordinary
tracking and no-consensus recovery separately. The first-frame depth centroid and
tracking/recovery triangulated center serve different purposes. Refinement can update
both rotation and translation. Frames, model crop resolution and refinement counts
are preserved by the speed changes.

## Packaging

- WAPR core source is pinned and bundled with its licenses and modified-file notice.
- Runtime environments and cache paths are configurable; no workstation paths are required.
- The standalone WAPR and MoSE3 WebUIs retain manual query points, 10 FPS sampling/playback,
  parallel lossless prediction export and on-demand ZIP generation. MoSE3 is optional
  and fetched separately at the documented revision.
- Models, native binaries, data, measured camera calibration, credentials and job
  histories are excluded. Licensed font assets have explicit SHA-256 allowlisting.

`provenance/v2-source-files.json` records the local source inputs and packaged hashes;
`docs/upload-manifest.tsv` records the published file set. The v1 import-graph reports
are historical; v2 adds the WAPR dependency path described above and in DEPENDENCIES.md.
