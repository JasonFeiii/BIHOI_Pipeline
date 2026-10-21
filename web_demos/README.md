# WAPR 本地 Web demo

上传 RGB-D、物体网格和目标 Mask（或框选区域），估计物体的 6D 位姿。
网页服务仅依赖 Python 标准库；模型在独立 Python 环境中执行。

## 安装

在仓库根目录执行：

```bash
bash scripts/install.sh --component wapr --skip-native
conda install -p .environments/wapr -c nvidia cuda-toolkit=12.8
bash scripts/build_native.sh wapr
bash scripts/check_env.sh --component wapr --skip-models
```

独立 demo 使用上游 OGL 渲染，需要 OpenGL/EGL 开发头文件、NVIDIA EGL
驱动和匹配的 CUDA 编译器。首次运行会编译扩展并下载缺失的官方权重。
缓存由 `WAPR_CACHE_DIR` 指定，默认在仓库的 `.models/wapr`。
源码及权重各自的许可证见 `third_party/WAPR/`。

## 启动

在本目录执行（脚本会读取仓库根目录 `.env`）：

```bash
bash run_wapr.sh
# http://127.0.0.1:7863/wapr
```

默认使用 `.environments/wapr/bin/python`。已有环境可在 `.env` 设置
`HOI_ENV_WAPR`，或通过 `--wapr-python /path/to/environment/bin/python` 指定。
源码位置可用 `--wapr-repo` 或 `WAPR_REPO` 配置。

仅查看页面可执行 `python3 server.py --port 7860`。服务默认监听
`127.0.0.1`，远程访问可使用 SSH 转发：
`ssh -L 7863:127.0.0.1:7863 user@server`。

## 输入和结果

- RGB：PNG/JPEG；深度：PNG/TIFF/NPY，与 RGB 尺寸相同。
- 物体网格：PLY/OBJ/STL/GLB，必须包含三角面片。
- 目标区域：上传与 RGB 同尺寸的前景 Mask，或在原图上拖动框选。
- 内参对应原始 RGB 分辨率。深度、网格按填写系数转换为米，直径也使用米。

运行后可查看位姿叠加图、4×4 物体到相机变换矩阵、排序分数，并下载
`pose.npy`、`mesh_camera.ply`、`result.json` 或全部结果 ZIP。
分数是排序信号，不是成功概率。文件保存在 `runtime/wapr/` 的
`uploads/` 和 `jobs/<任务 ID>/`。GPU 任务按顺序执行，可以取消。

## 验证

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
# 可选：需 Playwright 和运行中的服务
python3 tests/browser_smoke.py http://127.0.0.1:7863
# 已下载官方 pose_lmo 样例后，执行完整 HTTP 推理
python3 tests/run_official_sample.py wapr --url http://127.0.0.1:7863
```
