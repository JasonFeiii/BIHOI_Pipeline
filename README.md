# BIHOI

将多视角视频处理为物体掩码、深度、三维模型、物体位姿和手物交互结果，支持 WebUI 操作。

**运行条件：** Linux、NVIDIA GPU、Conda/Miniforge、Git、FFmpeg。处理使用 **10 FPS、640×360**。
**当前限制：** Hands / Final 仍需未公开的 WiLoR 适配代码；仅下载本仓库不能运行这两个阶段。

## 1. 安装

下载仓库后，在终端执行：

```bash
cd BIHOI_Pipeline
cp .env.example .env
bash scripts/install.sh --component all --skip-native
```

<details>
<summary>首次安装还需编译 CUDA 扩展（展开查看命令）</summary>

```bash
conda install -p .environments/sam3d-objects -c nvidia cuda-toolkit=12.1
conda install -p .environments/foundationpose -c nvidia cuda-toolkit=12.4
conda install -p .environments/wapr -c nvidia cuda-toolkit=12.8
conda install -p .environments/hocap-annotation -c nvidia cuda-toolkit=11.8

for component in sam3d-objects foundationpose wapr hocap-annotation; do
    bash scripts/build_native.sh "$component"
done
```

</details>

## 2. 准备模型

先申请 SAM3 / SAM3D 的 Hugging Face 访问权限，再下载：

```bash
for model in sam3 sam2 da3 mv-sam3d moge wilor fmhr; do
    bash scripts/download_models.sh --component "$model" --yes
done
```

另外手动准备：

- **FoundationPose：** 按[官方说明](https://github.com/NVlabs/FoundationPose#download-weights)下载，将两个 `model_best.pth` 分别放到 `.models/foundationpose/2023-10-28-18-33-37/` 和 `.models/foundationpose/2024-01-11-20-02-45/`。
- **MANO：** 在[官网](https://mano.is.tue.mpg.de/)注册下载，将 `MANO_LEFT.pkl`、`MANO_RIGHT.pkl` 放到 `.models/mano/`，执行 `bash scripts/prepare_mano_regressors.sh`。
- **WiLoR 适配代码：** 如已获得兼容版本及使用权限，执行下列命令；公开版 WiLoR 权重不能替代适配代码。

```bash
.environments/handobj_webui/bin/python scripts/import_wilor.py --source /path/to/licensed/WiLoR
.environments/handobj_webui/bin/python scripts/setup_assets.py
bash scripts/check_env.sh
```

## 3. 放置数据

默认目录结构如下，视频须同步，标定须匹配相机及处理分辨率：

```text
.calibration/
├── intri.yml
├── extri.yml
└── corrected_gsformat.pkl
data/rigid/participant/
├── calib/restored_flat_real_scale.pt
└── sequence/videos/take01/
    ├── node13_cam0.mp4
    ├── node13_cam1.mp4
    └── ...
```

`participant`、`sequence` 可自行命名。每位参与者需先做一次[手部标定](docs/HAND_CALIBRATION.md)，生成 `restored_flat_real_scale.pt`。
数据在其他位置时，在 `.env` 设置 `HOI_DATA_ROOT` 和 `HOI_CALIBRATION_DIR`。

## 4. 启动并运行

在 `.env` 设置 `WEBUI_PASSWORD`；用 `WEBUI_ALLOWED_GPUS=0` 指定 GPU（多卡可填 `0,1`）。

```bash
bash scripts/run_webui.sh
```

打开 **http://127.0.0.1:7860**，用户名为 `operator`，密码为 `.env` 中设置的值。远程服务器需先转发 7860 端口。

1. 选择数据目录、序列和 take，填写物体描述。
2. 设置 **10 FPS、640×360**，按顺序运行预处理、重建、物体跟踪、Hands、Final。
3. 在页面查看或导出结果；标准化结果位于序列目录的 `standardized_hoi_30fps/`，按 30 FPS 导出并保留原时长。

[安装问题](DEPENDENCIES.md) · [数据格式](docs/DATA_FORMAT.md) · [手部标定](docs/HAND_CALIBRATION.md) · [第三方许可证](THIRD_PARTY.md) · [当前可复现范围](docs/REPRODUCIBILITY.md)
