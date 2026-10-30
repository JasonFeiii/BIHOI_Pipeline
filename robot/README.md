# BIHOI 机器人阶段：Sharpa 重定向 + 双臂双手物理优化

接在采集流水线之后：把序列目录里的 `standardized_hoi_30fps/` 导出，先运动学重定向到双 Sharpa 手，
再放到双 UR3e + Sharpa 手的 MuJoCo 场景里做采样式 MPC 物理优化。

```text
standardized_hoi_30fps/ ──► retarget.py ──► sim/rank_takes.py ──► sim/prepare_task.py ──► run_batch.py / start_demo*.py
  (采集流水线导出)          Sharpa IK         按物体挑 take          CoACD + 场景 + 机械臂规划     物理优化 + Viser
```

## 目录

```text
robot/
├── retarget.py         MANO 导出 → Do as I Do Sharpa 双手 IK（只做运动学，不跑动力学）
├── run_batch.py        多个 take 同时做物理优化（spider_arm_mpc.py），自带 batch_viewer
├── start_demo.py       碗 demo：监督进程 + 检查点恢复（sim/run_from_zero.py → sim/arm_mpc.py）
├── start_demo1.py      碗 demo：SPIDER 式 MPC（sim/spider_arm_mpc.py），无监督进程
├── batch_viewer.py     run_batch 的实时 Viser 回放（只读）
├── opt_viewer.py       start_demo1 优化过程的 Viser 视图（只读）
└── sim/
    ├── robot_paths.py  所有运行目录/环境的唯一配置（见下）
    ├── prepare_task.py rank_takes.py hocap.py   任务准备
    ├── spider_arm_mpc.py arm_mpc.py run_from_zero.py ...  物理优化
    ├── render_*.py     出视频
    └── retargeting/    vendored Do as I Do（MIT，改动见 LOCAL_CHANGES.md）
```

## 1. 环境和资源

```bash
# Python 3.12 venv，放在 .environments/robot（或用 HOI_ENV_ROBOT 指向已有环境）
python3.12 -m venv .environments/robot
.environments/robot/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.environments/robot/bin/pip install -r environments/robot.requirements.txt

# Sharpa 机器人模型：取 Do as I Do 固定 commit 到 .external/do-as-i-do
python scripts/fetch_external.py do-as-i-do
```

上面的全新安装步骤尚未在干净机器上验证；当前服务器上 `.environments/robot` 链接到已有的 `physics-env`。

另外手动准备（都不进 Git）：

- **MANO：** 与采集流水线相同，`MANO_LEFT.pkl`、`MANO_RIGHT.pkl` 放在 `.models/mano/`。首次使用时会在
  `$BIHOI_ROBOT_DATA/mano/` 生成一份不依赖 chumpy 的副本（数组不变）。
- **UR3e 模型：** `ur3e.xml` 和 `assets/*.obj` 放在 `$BIHOI_ROBOT_DATA/demo/ur3e/`。
- **碗 demo 数据（只有 `start_demo*.py` 需要）：** `$BIHOI_ROBOT_DATA/demo/{plan,physics_contact_bowl}/`。

## 2. 运行目录

都在 `sim/robot_paths.py` 里定义，均已被 `.gitignore` 忽略；可在 `.env` 或环境变量里改：

| 变量 | 默认 | 内容 |
| --- | --- | --- |
| `BIHOI_ROBOT_DATA` | `data/robot` | `raw/` 导入的导出、`retargeted/`、`tasks/`、`demo/`、`mano/`、`manifest.json` |
| `BIHOI_ROBOT_OUTPUTS` | `outputs/robot` | `demo_runs/` 每次运行、`retarget_review/`、`retarget_videos/`、`results/` |
| `HOI_MANO_DIR` | `.models/mano` | 原始 MANO 模型 |
| `HOI_ENV_ROBOT` | `.environments/robot` | Python 环境；`run_batch.py`、`start_demo*.py`、viewer 和 `render_*.py` 会自动切换到它，其余脚本用它直接运行 |
| `HOI_EXTERNAL_ROOT` | `.external` | `do-as-i-do/` 固定 commit |

## 3. 重定向

```bash
PY=.environments/robot/bin/python
# 导入：递归查找 standardized_hoi_30fps/，按相对路径复制到 $BIHOI_ROBOT_DATA/raw，任务名为相对路径用 __ 连接
$PY robot/retarget.py prepare --source "$HOI_DATA_ROOT"          # 或 --archive exports.zip [--sha256 ...]
$PY robot/retarget.py batch --workers 8                          # 已完成的任务自动跳过；--tasks 只跑指定任务
$PY robot/retarget.py report                                     # outputs/robot/retarget_review/index.html
```

结果在 `$BIHOI_ROBOT_DATA/retargeted/sharpa/bimanual/<task>/0/`（`trajectory_kinematic.npz`、`scene_ik.xml`、`metrics.json`），
人手参考在 `retargeted/mano/bimanual/<task>/0/trajectory_keypoints.npz`。保留全部帧、30 Hz、原始物体位姿和桌面 Z=0。
没有可用的 EGL 离屏渲染时，加 `--no-video` 并设 `MUJOCO_GL=disable`（只跳过预览视频，IK 结果相同；之后不加 `--no-video` 重跑会补视频）。

## 4. 物理优化

```bash
$PY robot/sim/rank_takes.py                  # 每个物体挑最适合仿真的 take → tasks/take_ranking.json
$PY robot/run_batch.py --end 15 --parallel 4 # 每个物体最佳 take：prepare_task（CoACD、场景、机械臂规划）+ 物理优化
$PY robot/run_batch.py --tasks fwd_capture__0818_fwd_04__take05   # 指定 take
```

- 浏览器打开 `http://localhost:8150`（`--no-viewer` 不开）；每个仿真约需 18 GB 显存。
- 输出在 `outputs/robot/demo_runs/batch_<时间>/<物体>/`，`batch_status.json` 汇总状态。
- 碗 demo：`$PY robot/start_demo.py`（检查点恢复版）或 `$PY robot/start_demo1.py`（SPIDER 式），端口 8148。
- 出视频：`sim/render_retarget.py --task <task>`、`sim/render_run.py --run <run 目录>`、`sim/render_demo.py`（需要 EGL）。

单独调试一轮 `arm_mpc.py`（不经过监督进程）：

```bash
RUN=outputs/robot/demo_runs/<某次 start_demo 运行>
BIHOI_ARM_ROOT=$RUN PYTHONPATH=robot/sim CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MUJOCO_GL=disable \
$PY -u robot/sim/arm_mpc.py --name debug_0001 --port 8149 \
  --plan-result sqp_result_001.json --from-open-hand --hold-seconds 0 --continue-on-task-failure \
  --grasp-transport --contact-prior --native-screen --native-shortlist 12 --samples 96 --iterations 4 \
  --horizon 1.2 --ctrl-dt .2 --arm-jerk 1200 --arm-tracking-rate 40 --checkpoint-interval .2 \
  --native-retries 2 --max-seconds 24.5 --seed 63575
```

## 代码结构（物理部分）

| 用途 | 文件 |
|---|---|
| 场景 / 运动学 | `build_arm_scene.py` `arm_kinematics.py` `projector.py` `arm_servo.py` `smooth_targets.py` |
| 任务准备 / 规划 | `prepare_task.py` `rank_takes.py` `hocap.py` `prepare_reference.py` `search_placement.py` `plan_mink.py` `global_sqp.py` |
| 优化器 | `spider_arm_mpc.py`（SPIDER 式）、`arm_mpc.py` + `open_hand_pipeline.py` + `precontact_phase.py`（检查点恢复版） |
| 采样提议 | `wrench_search.py` `grasp_transport.py` `regrasp_candidates.py` `regrasp_guidance.py` `coherent_regrasp.py` |
| 安全 / 约束 | `contact_guard.py` `drop_guard.py` `wrist_clearance.py` `clearance_guidance.py` `collision_anticipation.py` `angular_budget.py` `angular_tracking.py` `handover_control.py` `temporal_support.py` |
| 求解器补丁 | `numerical_solver.py` `solver_convergence.py` `stable_linesearch.py` `solver_contacts.py` `cached_broadphase.py` |
| 恢复 / 审计 | `run_from_zero.py` `recover_contact.py` `recovery_beam.py` `trajectory_evidence.py` `frontier_backoff.py` `lineage.py` `motion_audit.py` |
| 进度 / 显示 | `compute_progress.py` `async_progress.py` `trajectory_store.py` `reference_overlay.py` `render_*.py` |

## 来源

- `retarget.py`：原 `BIHOI_Sharpa/bihoi_batch.py`；改为可配置路径、可导入任意 `standardized_hoi_30fps` 目录并合并 manifest、增加 `--tasks` / `--task` / `--no-video`。
- `sim/` 与入口脚本：原 `/workspace/bihoi`（v57 生效代码拍平后的版本）；只把数据/输出/环境路径改为 `robot_paths.py`。
- 第三方许可证见仓库根目录 [THIRD_PARTY.md](../THIRD_PARTY.md)。
