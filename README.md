# Hand-Object Interaction Pipeline v2

A **31-camera hand–object interaction processing pipeline** with a Gradio WebUI. From synchronized multi-view videos, it produces object masks, depth maps, 3D meshes, object poses, hand reconstruction, and standardized exports.

> **Reproducibility status:** this source release does not yet provide an independently reproducible full pipeline. **Hands requires a project-specific WiLoR source variant that is not published; Final requires its fitted MANO outputs.** Public WiLoR weights alone do not supply that variant. FMHR calibration weights and templates are publicly available and downloaded from a pinned official revision. MANO requires individual registration. See [reproducibility and remaining release blockers](docs/REPRODUCIBILITY.md) before installing the GPU stages.

> **10 FPS VERSION:** The current pipeline is designed for **10 FPS processing**. Input videos may have a different native FPS, but the **Hands** and **Final** stages require processing at **10 FPS**. The final export may be resampled to 30 FPS; that does **not** mean the inference pipeline runs at 30 FPS.

V2 integrates the current **FoundationPose / WAPR 6D object tracking** and live WebUI into the portable pipeline. It preserves the hand-calibration workflow from v1.

- First frame: best mask, fused masked-depth XYZ with outlier removal, **240 FoundationPose hypotheses → 5 refinements → best pose → six-candidate WAPR/SAPR selection**.
- Tracking: compare **WAPR(C), SAPR(C), SAPR(WAPR(C))** across views.
- No consensus: best mask + triangulated XYZ, **96 rotation hypotheses → WAPR → SAPR → WBPS**.
- Live preview: prominent green pose mesh and sparse red depth points at **35% opacity**. Full-sequence processing and result playback use the prepared 10 FPS timeline.
- Standalone [WAPR demo](web_demos/README.md) for RGB-D object pose estimation and result export.

See [6D tracking behavior and setup](docs/POSE_TRACKING_V2.md) and [reproducibility status](docs/REPRODUCIBILITY.md). Models, videos, measured calibration and runtime outputs are supplied separately.

## 1. Prepare your data

Place your videos under a data root in the following structure. **Each person must provide hand shape (beta) parameters** in `restored_flat_real_scale.pt`; use the separate [Hand Calibration setup](#hand-calibration-once-per-data-collector) below to generate them.

```text
data/
└── rigid/
    └── <person_name>/                        # Person / hand prior
        ├── calib/
        │   └── restored_flat_real_scale.pt  # Required for hand fitting
        └── <settings>/
            └── videos/
                ├── take01/
                │   ├── node13_cam0.mp4
                │   ├── node13_cam1.mp4
                │   └── ...          # Up to 31 synchronized camera videos
                └── take02/          # Optional additional take
                    └── ...
```

Prepare the **camera calibration** in a separate folder:

```text
calibration/
├── intri.yml
├── extri.yml
└── corrected_gsformat.pkl                 # Currently required by preparation scripts
```

The camera files must match your calibrated camera IDs and order; the system **does not calibrate cameras automatically**. The hand/final pipeline **requires 10 FPS processing** and defaults to **640×360**. See [exact camera names, calibration format, and optional inputs](docs/DATA_FORMAT.md).

## Hand Calibration (once per data collector)

**10 FPS setup:** Calibrate each person's hand once before running the main pipeline. This is a separate step, also available in the WebUI.

**Input** (example: `calibration_capture`):

```text
data/hand_calibration/calibration_capture/
├── videos/       
├── intri.yml
└── extri.yml
```

Use camera calibration matching **640×360** extracted frames.

**Setup & run** (from the repository root):

```bash
# After reviewing/accepting the psbody-mesh license:
HOI_ACCEPT_PSBODY_MESH_LICENSE=1 bash scripts/install_hand_calibration_deps.sh

bash run_hand_calibration_args.sh \
    --data_name calibration_capture \
    --person_name participant \
    --start_sec 0 \
    --end_sec 0.1 \
    --hand_side right
```

Requires the licensed **WiLoR/MANO** assets and **FMHR checkpoints/templates** in addition to the installed environments. See [full setup instructions](docs/HAND_CALIBRATION.md).

**Output:** `data/rigid/<person_name>/calib/restored_flat_real_scale.pt` (MANO beta parameters used by the Hands/Final stages). Specify `--person_name` explicitly, or supply your own mapping. No collector identities or calibration captures are distributed.

## 2. Install environments

**Requirements:** Linux, NVIDIA GPU(s), a compatible driver, Conda/Miniforge, Git, FFmpeg/ffprobe, and CUDA build tools for native extensions. High-memory GPUs are recommended.

```bash
git clone <repository-url> BIHOI_Pipeline
cd BIHOI_Pipeline
cp .env.example .env
```

Edit `.env` to set the paths and GPUs for your machine:

```dotenv
HOI_DATA_ROOT=/absolute/path/to/data
HOI_CALIBRATION_DIR=/absolute/path/to/calibration
WEBUI_ALLOWED_GPUS=0,1
WEBUI_PASSWORD=choose-a-password
```

Then install the project's **nine separate Conda environments**:

```bash
bash scripts/install.sh --component all --skip-native
```

Native GPU components also need CUDA toolkits matching their PyTorch builds:

```bash
conda install -p .environments/sam3d-objects -c nvidia cuda-toolkit=12.1
conda install -p .environments/foundationpose -c nvidia cuda-toolkit=12.4
conda install -p .environments/wapr -c nvidia cuda-toolkit=12.8
conda install -p .environments/hocap-annotation -c nvidia cuda-toolkit=11.8

bash scripts/build_native.sh sam3d-objects
bash scripts/build_native.sh foundationpose
bash scripts/build_native.sh wapr
bash scripts/build_native.sh hocap-annotation
```

These are setup instructions, **not a guarantee of a clean install on every GPU system**. For version-specific requirements and troubleshooting, see [DEPENDENCIES.md](DEPENDENCIES.md).

## 3. Download model weights

First accept any required upstream licenses and obtain Hugging Face access for the **gated SAM3 and SAM3D** models. Then run:

```bash
for model in sam3 sam2 da3 mv-sam3d moge wilor fmhr; do
    bash scripts/download_models.sh --component "$model" --yes
done
```

FMHR's GCN checkpoints and templates are downloaded from the official repository at a fixed commit and verified with SHA-256. They are not private project checkpoints. Two model families require **manual downloads**:

- **FoundationPose:** follow the [official weight instructions](https://github.com/NVlabs/FoundationPose#download-weights). Put `model_best.pth` in both `.models/foundationpose/2023-10-28-18-33-37/` and `.models/foundationpose/2024-01-11-20-02-45/`.
- **MANO:** register at [the official MANO website](https://mano.is.tue.mpg.de/). Place `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `.models/mano/`, then run `bash scripts/prepare_mano_regressors.sh`.

The **hand reconstruction stage** additionally requires the separately licensed, project-compatible WiLoR source variant; it is **not included in this repository**. Once you have permission and that variant, import it with:

```bash
.environments/handobj_webui/bin/python scripts/import_wilor.py --source /path/to/licensed/WiLoR
.environments/handobj_webui/bin/python scripts/setup_assets.py
bash scripts/check_env.sh
```

Checkpoints are stored in `.models/` by default. See [third-party licensing and restrictions](THIRD_PARTY.md).

## 4. Start the WebUI

```bash
bash scripts/run_webui.sh
```

Open **http://127.0.0.1:7860** and sign in with the credentials configured in `.env` (default username: `operator`).

Passwords are never printed to logs. If `WEBUI_PASSWORD` is unset, the launcher writes a generated password to an owner-only runtime file and prints only its location.

If running on a remote GPU server, forward port **7860** through SSH or VS Code. Do not expose the interface publicly without authentication.

In the WebUI:

1. Select your **data root**, dataset (e.g. `rigid/participant/my_sequence`), and take.
2. Set an **object text prompt**, **10 FPS** (required for Hands / Final), and **640×360**.
3. Choose individual stages: **Preprocess → Reconstruction → Object Tracking → Hands → Final Export**. Hands and the normal Final workflow remain unavailable to a fresh clone until the unpublished WiLoR variant and other listed prerequisites are supplied.
4. Inspect masks, depth, mesh, tracking, hand results, and previews directly in the interface.

Processed outputs are saved under the selected dataset (e.g. `images/`, `masks/`, `depth/`, `meshes/compressed.ply`, and `standardized_hoi_30fps/`).

> **Important:** The full GPU pipeline requires your own calibrated capture, weights, MANO/person prior, and licensed WiLoR variant. New-machine end-to-end installation and inference have not yet been fully validated.


The server accepts only `HOI_DATA_ROOT` and explicitly configured `HOI_ALLOWED_DATA_ROOTS`; browser inputs cannot add trusted roots. Exports use fresh folders under `data/exports/` (or `HOI_ALLOWED_EXPORT_ROOTS`) and do not delete existing destination files. See [security policy](docs/SECURITY.md).

Standardized export receives the processed input rate explicitly: **10 FPS input → 30 FPS output**, preserving elapsed time. Existing incorrectly timed exports must be regenerated.

**More details:** [Data format](docs/DATA_FORMAT.md) · [Dependencies](DEPENDENCIES.md) · [Model licenses](THIRD_PARTY.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Integration test guide](docs/INTEGRATION.md)
