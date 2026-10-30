"""BIHOI standard MANO exports -> Do as I Do Sharpa bimanual IK.

    python robot/retarget.py prepare --source <dir with standardized_hoi_30fps/ exports>   # or --archive <zip>
    python robot/retarget.py batch [--workers 8] [--tasks TASK ...]
    python robot/retarget.py one --task TASK
    python robot/retarget.py report

Preserves every input frame, object pose, world coordinates and 30-Hz timing.
Physics optimization and rollout are deliberately not run (see robot/sim/prepare_task.py).
A task is named by the export's relative path under --source, '__'-joined
(fwd_capture/0818_fwd_03/take01 -> fwd_capture__0818_fwd_03__take01).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
import smplx
import trimesh
import hocap
import retargeting.pipeline.generate_scene as generator
from retargeting.pipeline.process_dataset import _build_side_qpos
from retargeting.pipeline.generate_scene import main as generate_scene
from retargeting.pipeline.solve_ik import main as solve_ik
from robot_paths import DATA, DO_AS_I_DO, OUTPUTS

OUT = DATA / "retargeted"
REPORT = OUTPUTS / "retarget_review"
MANIFEST = DATA / "manifest.json"
DO_AS_I_DO_COMMIT = "824591b808c342b20079c3b4198a8c2bdf88c74e"  # scripts/fetch_external.py do-as-i-do
# Sharpa robot assets come from the pinned upstream checkout (they are not vendored).
generator.ROOT = str(DO_AS_I_DO)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
torch.set_num_threads(1)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def mano_layers():
    # Chumpy-free copies of the operator's MANO pickles (same arrays), shared with sim/hocap.py.
    return {s: smplx.MANO(str(hocap.mano_model(s)), is_rhand=(s == "right"),
                          use_pca=False, flat_hand_mean=True).eval()
            for s in ["right", "left"]}


def fk(layer, params):
    n = len(params["global_orient"])
    vertices, joints = [], []
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        betas = params["betas"]
        if betas.ndim == 1:
            betas = np.tile(betas[None], (stop - start, 1))
        elif len(betas) == 1:
            betas = np.tile(betas, (stop - start, 1))
        else:
            betas = betas[start:stop]
        with torch.no_grad():
            result = layer(
                global_orient=torch.as_tensor(params["global_orient"][start:stop], dtype=torch.float32),
                hand_pose=torch.as_tensor(params["hand_pose"][start:stop], dtype=torch.float32),
                transl=torch.as_tensor(params["transl"][start:stop], dtype=torch.float32),
                betas=torch.as_tensor(betas, dtype=torch.float32), return_verts=True,
            )
        j = result.joints.cpu().numpy()
        if j.shape[1] == 16:
            # SMPL-X MANO exposes the 16 skeletal joints; add its documented
            # MANO fingertip vertices in thumb/index/middle/ring/pinky order.
            from smplx.vertex_ids import vertex_ids
            tip_ids = [vertex_ids["mano"][f] for f in FINGERS]
            j = np.concatenate([j, result.vertices[:, tip_ids].cpu().numpy()], axis=1)
        if j.shape[1] != 21:
            raise ValueError(f"Expected 21 SMPL-X MANO joints, got {j.shape}")
        joints.append(j[:, MANO_TO_OPENPOSE])
        vertices.append(result.vertices.cpu().numpy())
    return np.concatenate(joints), np.concatenate(vertices)


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []


def import_exports(source=None, archive=None, sha256=None):
    """Copy standardized_hoi_30fps exports into DATA/raw (same relative paths); return the imported ones."""
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if archive is not None:
        archive = Path(archive)
        if sha256:
            with archive.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == sha256
        with zipfile.ZipFile(archive) as z:
            names = [info.filename for info in z.infolist()]
            for name in names:
                (raw / name).resolve().relative_to(raw.resolve())
            z.extractall(raw)
        return sorted({raw / Path(name).parent for name in names if Path(name).name == "metadata.json"
                       and Path(name).parent.name == "standardized_hoi_30fps"})
    source = Path(source).resolve()
    found = sorted(source.rglob("standardized_hoi_30fps"))
    if source.resolve() == raw.resolve():
        return found
    imported = []
    for directory in found:
        dest = raw / directory.relative_to(source)
        shutil.copytree(directory, dest, dirs_exist_ok=True)
        imported.append(dest)
    return imported


def prepare(source=None, archive=None, sha256=None):
    raw = DATA / "raw"
    exports = import_exports(source, archive, sha256)
    layers = mano_layers()
    rows = []
    for k, directory in enumerate(exports):
        meta = json.loads((directory / "metadata.json").read_text())
        relative = directory.relative_to(raw).as_posix()
        task = "__".join(directory.relative_to(raw).parts[:-1])
        n, fps = meta["num_frames"], float(meta["fps"])
        assert fps == 30 and meta["world_frame"]["up_axis"] == "+Z"
        out_dir = OUT / "mano/bimanual" / task / "0"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = np.load(directory / "frames.npy", allow_pickle=False)
        assert len(frames) == n
        data = {}
        for side in ["right", "left"]:
            params = dict(np.load(directory / f"mano_{side}.npz", allow_pickle=False))
            assert params["hand_pose"].shape == (n, 45)
            assert meta[f"mano_{side}"]["hand_pose"]["flat_hand_mean"] is True
            j, v = fk(layers[side], params)
            wrist, fingers = _build_side_qpos(j, j[:, [4, 8, 12, 16, 20]],
                                            params["global_orient"], np.zeros(3), side == "right")
            data.update({f"qpos_wrist_{side}": wrist,
                         f"qpos_finger_{side}": fingers,
                         f"qpos_pip_{side}": j[:, [2, 6, 10, 14, 18]],
                         f"qpos_dip_{side}": j[:, [3, 7, 11, 15, 19]],
                         f"mano_joints_{side}": j,
                         f"mano_verts_{side}": v,
                         f"mano_faces_{side}": layers[side].faces.astype(np.int32),
                         f"mano_finger_pose_{side}": params["hand_pose"],
                         f"mano_global_orient_{side}": params["global_orient"],
                         f"mano_betas_{side}": params["betas"],
                         f"contact_{side}": np.zeros((n, 5)),
                         f"contact_pos_{side}": np.zeros((5, 3))})
        obj = dict(np.load(directory / "obj_pose.npz", allow_pickle=False))
        trans = obj.get("trans", obj.get("transl"))
        quat = obj["quat"]
        assert trans.shape == (n, 3) and quat.shape == (n, 4), obj.keys()
        assert np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4)
        data["qpos_obj_right"] = np.c_[trans, quat]
        # Keep the upstream shared-object convention for bimanual scenes.
        data["qpos_obj_left"] = data["qpos_obj_right"].copy()
        data["frames"] = frames
        data["source_frame_ids"] = np.load(directory / "source_frame_ids.npy", allow_pickle=False)
        data["timestamps_sec"] = np.load(directory / "timestamps_sec.npy", allow_pickle=False)
        assert all(np.isfinite(x).all() for x in data.values())
        np.savez_compressed(out_dir / "trajectory_keypoints.npz", **data)
        mesh_path = directory / "object_mesh.ply"
        mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        mesh_dir = OUT / "assets/objects" / mesh_hash[:16]
        if not (mesh_dir / "visual.obj").exists():
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh = trimesh.load(mesh_path, process=False)
            assert isinstance(mesh, trimesh.Trimesh)
            mesh.export(mesh_dir / "visual.obj")
            shutil.copy2(mesh_path, mesh_dir / "source_mesh.ply")
        task_info = {"task": task, "dataset_name": "bihoi", "robot_type": "mano",
                     "embodiment_type": "bimanual", "data_id": 0,
                     "right_object_mesh_dir": mesh_dir.relative_to(OUT).as_posix(),
                     "left_object_mesh_dir": None, "ref_dt": 1/fps,
                     "n_frames": n, "source_relative_path": relative,
                     "source_metadata": meta, "table": json.loads((directory / "table.json").read_text()),
                     "collision_preparation": "visual convex hull for IK only; CoACD required before physics"}
        dump(out_dir.parent / "task_info.json", task_info)
        rows.append({"task": task, "source": relative, "n_frames": n, "fps": fps,
                     "duration_sec": (n-1)/fps, "mesh_sha256": mesh_hash})
        print(f"PREPARED {k+1}/{len(exports)} {task} frames={n}", flush=True)
    assert rows, f"no standardized_hoi_30fps export found under {source or archive}"
    # Merge by task so that later imports keep earlier tasks.
    merged = {row["task"]: row for row in load_manifest()} | {row["task"]: row for row in rows}
    dump(MANIFEST, sorted(merged.values(), key=lambda row: row["task"]))
    dump(DATA / "provenance.json", {
        "source": str(archive or source), "archive_sha256": sha256, "source_count": len(rows),
        "do_as_i_do_commit": DO_AS_I_DO_COMMIT,
        "method": "Do as I Do original generate_scene + solve_ik; dynamics disabled through explicit option",
        "world_policy": "Preserve source metres, +Z up, tabletop Z=0 and exact object transforms",
        "mano_policy": "SMPL-X MANO, non-PCA, flat_hand_mean=True; do not add the pose mean",
        "time_policy": "Every input frame, native 30 Hz; no crop, smoothing or synthetic lift",
        "torch_version": torch.__version__, "mujoco_version": mujoco.__version__})
    print("PREPARE COMPLETE", sum(x["n_frames"] for x in rows), "frames", flush=True)


def evaluate_and_render(row, render=True):
    task, n = row["task"], row["n_frames"]
    directory = OUT / "sharpa/bimanual" / task / "0"
    model = mujoco.MjModel.from_xml_path(str(directory / "scene_ik.xml"))
    data = mujoco.MjData(model)
    trajectory = dict(np.load(directory / "trajectory_kinematic.npz"))
    ref = dict(np.load(OUT / "mano/bimanual" / task / "0/trajectory_keypoints.npz"))
    qpos = trajectory["qpos"]
    assert qpos.shape == (n, model.nq), (qpos.shape, n, model.nq)
    assert np.isfinite(qpos).all() and np.isfinite(trajectory["qvel"]).all()
    assert float(trajectory["frequency"]) == 30.0
    assert np.allclose(qpos[:, -14:-7], ref["qpos_obj_right"], atol=1e-8)
    assert np.allclose(qpos[:, -7:], ref["qpos_obj_left"], atol=1e-8)
    limits = model.jnt_limited.astype(bool)
    limit_excess = 0.0
    for i in np.flatnonzero(limits):
        values = qpos[:, model.jnt_qposadr[i]]
        lo, hi = model.jnt_range[i]
        limit_excess = max(limit_excess, float(np.max(np.maximum(lo-values, values-hi))))
    errors, wrist_errors = [], []
    site_ids = [model.site(f"{s}_{f}_tip").id for s in ["right", "left"] for f in FINGERS]
    wrist_ids = [model.site(f"{s}_palm").id for s in ["right", "left"]]
    targets = np.concatenate([ref["qpos_finger_right"][:,:,:3],ref["qpos_finger_left"][:,:,:3]],axis=1)
    wrists = np.stack([ref["qpos_wrist_right"][:,:3], ref["qpos_wrist_left"][:,:3]],axis=1)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        errors.append(np.linalg.norm(data.site_xpos[site_ids] - targets[t], axis=1))
        wrist_errors.append(np.linalg.norm(data.site_xpos[wrist_ids] - wrists[t], axis=1))
    errors, wrist_errors = np.array(errors), np.array(wrist_errors)
    metrics = {**row, "status": "complete", "output_frames": len(qpos), "nq":model.nq,"nv":model.nv,
               "fingertip_mean_mm": float(errors.mean()*1000),
               "fingertip_p95_mm": float(np.quantile(errors,.95)*1000),
               "fingertip_max_mm": float(errors.max()*1000),
               "wrist_mean_mm": float(wrist_errors.mean()*1000),
               "joint_limit_excess_rad": max(0.0,limit_excess),
               "physics_optimized": False, "collision_model": "convex visual hull placeholder",
               # Only a rendered review video makes the result final (run_one re-renders otherwise).
               "preview_version": 2 if render else 0}
    np.savez_compressed(directory / "tracking_errors.npz", fingertip_m=errors, wrist_m=wrist_errors,
                        frames=ref["frames"], timestamps_sec=ref["timestamps_sec"])
    if render:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT / "videos").mkdir(exist_ok=True)
        model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        all_points = np.concatenate([targets.reshape(-1,3), wrists.reshape(-1,3),ref["qpos_obj_right"][:,:3]])
        lo, hi = np.quantile(all_points, [.01,.99], axis=0)
        camera.lookat[:] = (lo+hi)/2
        camera.distance = max(.65, float(np.linalg.norm(hi-lo))*1.5)
        camera.azimuth, camera.elevation = 90, -30
        options = mujoco.MjvOption()
        options.geomgroup[:] = [1,1,0,0,0,0]
        options.sitegroup[:] = [0,0,0,0,0,0]
        ref_options = mujoco.MjvOption()
        ref_options.geomgroup[:] = [1,0,0,0,0,0]
        ref_options.sitegroup[:] = [0,0,0,0,0,0]
        import cv2
        bones = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                 (15,16),(0,17),(17,18),(18,19),(19,20)]
        stride = max(1, math.ceil(n/180))
        with imageio.get_writer(REPORT / "videos" / f"{task}.mp4", fps=30/stride, codec="libx264", quality=7) as writer:
            for t in range(0,n,stride):
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera,scene_option=options)
                robot_frame = renderer.render().copy()
                renderer.update_scene(data,camera=camera,scene_option=ref_options)
                for side in ["right","left"]:
                    joints = ref[f"mano_joints_{side}"][t]
                    for a,b in bones:
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,
                                           np.zeros(3),np.zeros(3),np.eye(3).flatten(),
                                           np.array([1.,.35,0.,1.],dtype=np.float32))
                        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0025,joints[a],joints[b])
                        renderer.scene.ngeom += 1
                human_frame = renderer.render().copy()
                frame = np.concatenate([human_frame,robot_frame],axis=1)
                cv2.putText(frame,"Human reference",(16,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,180,80),2,cv2.LINE_AA)
                cv2.putText(frame,"Sharpa IK",(656,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,230,230),2,cv2.LINE_AA)
                writer.append_data(frame)
                if t == (n//2//stride)*stride:
                    imageio.imwrite(REPORT / "videos" / f"{task}.jpg",frame)
        renderer.close()
    dump(directory / "metrics.json",metrics)
    return metrics


def run_one(row,video=True):
    start=time.time()
    task=row["task"]
    directory=OUT / "sharpa/bimanual" / task / "0"
    metrics_path=directory/"metrics.json"
    if metrics_path.exists():
        existing=json.loads(metrics_path.read_text())
        if existing.get("preview_version")==2:
            return existing
        return evaluate_and_render(row,video)
    try:
        generate_scene(output_root_dir=str(OUT),dataset_name="bihoi",robot_type="sharpa",
                       embodiment_type="bimanual",task=task,data_id=0,force=True,
                       use_visual_mesh_as_collision=True,object_floor_collision=True,
                       hand_floor_collision=False,use_pedestal=False,use_support=False,
                       show_viewer=False,add_ur3_arm=False)
        # Default end_idx=-1 drops a frame; an explicit N preserves the full clip.
        solve_ik(output_root_dir=str(OUT),dataset_name="bihoi",robot_type="sharpa",
                 embodiment_type="bimanual",task=task,data_id=0,end_idx=row["n_frames"],
                 ref_dt=1/row["fps"],sim_dt=1/(row["fps"]*7),show_viewer=False,
                 save_video=False,force=True,smoothing=False,run_forward_validation=False)
        metrics=evaluate_and_render(row,video)
        metrics["elapsed_sec"]=time.time()-start
        dump(metrics_path,metrics)
        return metrics
    except Exception:
        result={**row,"status":"failed","error":traceback.format_exc(),"elapsed_sec":time.time()-start}
        dump(directory/"error.json",result)
        return result


def report(rows):
    results=[]
    for row in rows:
        d=OUT/"sharpa/bimanual"/row["task"]/"0"
        p=d/"metrics.json"
        e=d/"error.json"
        results.append(json.loads(p.read_text()) if p.exists() else json.loads(e.read_text()) if e.exists() else {**row,"status":"pending"})
    REPORT.mkdir(parents=True,exist_ok=True)
    dump(REPORT/"results.json",results)
    fields=["task","status","n_frames","output_frames","duration_sec","fingertip_mean_mm","fingertip_p95_mm","wrist_mean_mm","joint_limit_excess_rad","elapsed_sec"]
    with (REPORT/"results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(results)
    complete=[x for x in results if x["status"]=="complete"]
    dump(REPORT/"summary.json",{"total":len(rows),"complete":len(complete),
                                "failed":sum(x["status"]=="failed" for x in results),
                                "pending":sum(x["status"]=="pending" for x in results),
                                "frames":sum(x["n_frames"] for x in complete),
                                "physics_optimized":False})
    cards=[]
    for x in results:
        name=html.escape(x["task"])
        if x["status"]=="complete":
            cards.append(f'<article data-name="{name}"><h3>{name}</h3><p>{x["n_frames"]} frames · {x["duration_sec"]:.1f}s · fingertip mean {x["fingertip_mean_mm"]:.1f} mm · p95 {x["fingertip_p95_mm"]:.1f} mm</p><video controls preload="none" poster="videos/{name}.jpg" src="videos/{name}.mp4"></video><label><input type="checkbox" value="{name}" onchange="selected()"> Select for later physics optimization</label></article>')
        else:
            cards.append(f'<article data-name="{name}"><h3>{name}</h3><p>{x["status"]}</p></article>')
    page='''<!doctype html><meta charset="utf-8"><title>BIHOI · Sharpa retargeting review</title>
<style>body{font:15px system-ui;background:#111827;color:#e5e7eb;margin:28px}header{position:sticky;top:0;background:#111827;padding:10px;z-index:2}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}article{background:#1f2937;padding:16px;border-radius:12px}h3{font-size:14px;overflow-wrap:anywhere}video{width:100%;background:#000}input[type=search]{padding:10px;width:45%}button{padding:10px}a{color:#93c5fd}p{color:#cbd5e1}textarea{width:100%;height:90px}</style>
<header><h1>BIHOI → Sharpa · kinematic review</h1><p>Do as I Do IK · bimanual · full 30 Hz references · orange dots: human fingertip targets. Physics optimization has not been run. Collisions/grasps are not validated. Source geometry, poses and table height are preserved.</p><input type="search" placeholder="Filter by task" oninput="document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.name.toLowerCase().includes(this.value.toLowerCase()))"> <a href="results.csv">Metrics CSV</a> · <a href="results.json">JSON</a><details><summary>Selected clips</summary><textarea id="selection" readonly></textarea><button onclick="navigator.clipboard.writeText(document.querySelector('#selection').value)">Copy list</button></details></header><main>'''+"\n".join(cards)+'''</main><script>function selected(){document.querySelector('#selection').value=[...document.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.value).join('\n')}</script>'''
    (REPORT/"index.html").write_text(page,encoding="utf-8")
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode",choices=["prepare","one","batch","report"])
    parser.add_argument("--source",type=Path,help="prepare: directory searched for standardized_hoi_30fps/ exports (e.g. $HOI_DATA_ROOT)")
    parser.add_argument("--archive",type=Path,help="prepare: zip of standardized_hoi_30fps/ exports")
    parser.add_argument("--sha256",help="prepare: expected sha256 of --archive")
    parser.add_argument("--index",type=int,default=0,help="one: manifest row")
    parser.add_argument("--task",help="one: task name (instead of --index)")
    parser.add_argument("--tasks",nargs="*",help="batch: only these tasks (default: whole manifest)")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--no-video",action="store_true",help="skip the review videos (no offscreen rendering; use with MUJOCO_GL=disable)")
    args=parser.parse_args()
    if args.mode=="prepare":
        if (args.source is None)==(args.archive is None):parser.error("prepare needs exactly one of --source / --archive")
        prepare(args.source,args.archive,args.sha256);return
    rows=load_manifest()
    if args.mode=="one":
        row=next(r for r in rows if r["task"]==args.task) if args.task else rows[args.index]
        result=run_one(row,not args.no_video);print(json.dumps(result,indent=2),flush=True);report(rows)
        if result["status"]!="complete":raise SystemExit(1)
    elif args.mode=="batch":
        from concurrent.futures import ProcessPoolExecutor,as_completed
        todo=[r for r in rows if not args.tasks or r["task"] in args.tasks]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_one,row,not args.no_video):row for row in todo}
            for i,future in enumerate(as_completed(futures),1):
                result=future.result()
                print(f'{i}/{len(todo)} {result["status"]} {result["task"]}',flush=True)
                report(rows)
        results=report(rows)
        if any(x["status"]!="complete" for x in results if x["task"] in {r["task"] for r in todo}):raise SystemExit(1)
    else:report(rows)


if __name__=="__main__":main()
