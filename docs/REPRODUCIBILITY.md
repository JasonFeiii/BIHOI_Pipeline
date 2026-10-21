# Reproducibility and release status

This is a research pipeline with mixed third-party terms. It is not yet a self-contained, independently reproducible end-to-end release.

| Component | Available to a new user | Remaining requirement |
| --- | --- | --- |
| WebUI, preprocessing and CPU exports | Source and synthetic regression tests | Follow environment setup; GPU inference is separate |
| Object reconstruction and tracking | Vendored or pinned source, upstream model downloads | Some weights are gated/noncommercial; matching camera calibration is required |
| WiLoR weights | Official model and detector downloads | They do not include the unpublished project-specific detector scripts |
| Hands | Fitting source | Exact compatible WiLoR variant, licensed MANO models and a person prior |
| FMHR calibration | Public GCN weights and templates, pinned and hash-checked | MANO and the WiLoR variant are still required |
| Final | Refinement/export source | Fitted hands, MANO and the separately licensed HO-Cap dependency |

## FMHR weights

`gcn_left.pth`, `gcn_right.pth`, `TEMPLATE_LEFT.obj` and `TEMPLATE_RIGHT.obj` are public in [the official FMHR repository](https://github.com/agnJason/FMHR/tree/ab624d62949eec26368bf73f73dc27b3c6a7fa4e/mano). They are not private project-trained checkpoints:

```bash
bash scripts/download_models.sh --component fmhr --yes
```

`configs/models/manifest.json` pins the commit and each file's SHA-256. MANO PKLs are excluded; obtain them under the MANO terms.

## Remaining publication blockers

1. Publish a lawfully distributable WiLoR adapter/variant or implement and validate a public alternative with the same output contract. Hashes of unavailable source do not solve reproducibility.
2. Resolve the license for project-owned integration code. A future parent-repository license must not silently replace third-party licenses.
3. Review the combined workflow's obligations: HO-Cap has AGPL terms, while FoundationPose and EasyMocap impose noncommercial restrictions. Separate installation does not establish license compatibility.
4. Run a clean installation and a complete sequence on a new machine, documenting source revisions, model sources, inputs and expected outputs. Unit tests do not establish GPU reproducibility.

Do not advertise “clone and run the full pipeline” until these conditions are met.

## Export timing

Inference uses 10 FPS. Standardized output is resampled to 30 FPS without shortening elapsed time. The exporter requires `--input_fps`; the WebUI passes the processed rate explicitly. `--max_frames 500` caps output samples and can intentionally truncate long sequences; use `0` to export all samples.

Exports generated with an incorrect 30 FPS input assumption must be regenerated. Changing only names or metadata cannot repair interpolation and timestamps.
