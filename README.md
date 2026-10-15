# Hand-Object Interaction Pipeline v2

A **31-camera hand–object interaction processing pipeline** with a Gradio WebUI. From synchronized multi-view videos, it produces object masks, depth maps, 3D meshes, object poses, hand reconstruction, and standardized exports.

> **10 FPS VERSION:** The current pipeline is designed for **10 FPS processing**. Input videos may have a different native FPS, but the **Hands** and **Final** stages require processing at **10 FPS**. The final export may be resampled to 30 FPS; that does **not** mean the inference pipeline runs at 30 FPS.

V2 integrates the current **FoundationPose / WAPR 6D object tracking** and live WebUI into the portable pipeline. It preserves the hand-calibration workflow from v1.

- First frame: best mask, fused masked-depth XYZ with outlier removal, **240 FoundationPose hypotheses → 5 refinements → best pose → six-candidate WAPR/SAPR selection**.
- Tracking: compare **WAPR(C), SAPR(C), SAPR(WAPR(C))** across views.
- No consensus: best mask + triangulated XYZ, **96 rotation hypotheses → WAPR → SAPR → WBPS**.
- Live preview: prominent green pose mesh and sparse red depth points at **35% opacity**. Full-sequence processing and result playback use the prepared 10 FPS timeline.
- Standalone [WAPR / MoSE3 demos](web_demos/README.md), including manual frame-click query points and faster result export.

See [6D tracking behavior and setup](docs/POSE_TRACKING_V2.md), [changes](docs/V2_CHANGES.md), and [validation](docs/VALIDATION_V2.md). Models, videos, measured calibration and runtime outputs are supplied separately.

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

**Input** (example: `hands_ck`):

```text
data/hand_calibration/hands_ck/
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
    --data_name hands_ck \
    --start_sec 0 \
    --end_sec 0.1 \
    --hand_side right
```

Requires the licensed **WiLoR/MANO** assets and **FMHR checkpoints/templates** in addition to the installed environments. See [full setup instructions](docs/HAND_CALIBRATION.md).

**Output:** `data/rigid/<person_name>/calib/restored_flat_real_scale.pt` (MANO beta parameters used by the Hands/Final stages). In the included mapping, `hands_ck` belongs to person `ck`; use `--person_name` for other collectors. A right-hand calibration has been validated; left-hand execution is not yet verified.

## 2. Install environments

**Requirements:** Linux, NVIDIA GPU(s), a compatible driver, Conda/Miniforge, Git, FFmpeg/ffprobe, and CUDA build tools for native extensions. High-memory GPUs are recommended.

```bash
git clone https://github.com/JasonFeiii/Hand-Object-Interaction-Pipeline-v2.git
cd Hand-Object-Interaction-Pipeline-v2
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
for model in sam3 sam2 da3 mv-sam3d moge wilor; do
    bash scripts/download_models.sh --component "$model" --yes
done
```

Two model families require **manual downloads**:

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

If running on a remote GPU server, forward port **7860** through SSH or VS Code. Do not expose the interface publicly without authentication.

In the WebUI:

1. Select your **data root**, dataset (e.g. `rigid/ck/my_sequence`), and take.
2. Set an **object text prompt**, **10 FPS** (required for Hands / Final), and **640×360**.
3. Run the full pipeline or choose individual stages: **Preprocess → Reconstruction → Object Tracking → Hands → Final Export**.
4. Inspect masks, depth, mesh, tracking, hand results, and previews directly in the interface.

Processed outputs are saved under the selected dataset (e.g. `images/`, `masks/`, `depth/`, `meshes/compressed.ply`, and `standardized_hoi_30fps/`).

> **Important:** The full GPU pipeline requires your own calibrated capture, weights, MANO/person prior, and licensed WiLoR variant. New-machine end-to-end installation and inference have not yet been fully validated.


**More details:** [Data format](docs/DATA_FORMAT.md) · [Dependencies](DEPENDENCIES.md) · [Model licenses](THIRD_PARTY.md) · [Validation status](docs/VALIDATION.md) · [Integration test guide](docs/INTEGRATION.md)
