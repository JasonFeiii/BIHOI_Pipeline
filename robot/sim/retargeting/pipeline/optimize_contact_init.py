"""Experimental contact-timed initialization using the Do-as-I-Do MPC.

The original trajectory is read-only.  Pre-contact kinematic approach is followed
by physical hand control, temporary object weld tracking, gravity release, and a
strict unsupported-hold gate before the remaining demonstration can execute.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import mujoco_warp as mjwarp
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import warp as wp

from retargeting.config import Config, process_config
from retargeting.pipeline.generate_scene import main as generate_scene
from retargeting.utils import mjwp as physics
from retargeting.utils import viser_viewer as live
from retargeting.utils.sampling import make_optimize_fn, make_optimize_once_fn, make_rollout_fn
from robot_paths import DATA

TASK = "fwd_capture__8.27_big_steel_bowl_fwd__take01"
ROOT = DATA / "demo" / "physics_contact_bowl"


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda x:x.item() if isinstance(x,np.generic) else str(x)), encoding="utf-8")
    temp.replace(path)


def phase_at(seconds, schedule):
    """The same phase rule is used by candidate rollouts and execution."""
    locked = seconds < schedule["release_s"] - 1e-8
    fraction = np.clip((seconds-schedule["release_s"])/.5, 0, 1)
    return locked, float(fraction*fraction*(3-2*fraction))


def evaluate_grasp_gate(rows):
    checks={"two_second_window":len(rows)>=59,
            "all_unwelded":bool(rows) and all(not x["weld_active"] for x in rows),
            "full_gravity":bool(rows) and all(x["gravity_fraction"]>.999 for x in rows),
            "position_within_20mm":bool(rows) and all(x["object_position_error_m"]<=.02 for x in rows),
            "rotation_within_15deg":bool(rows) and all(x["object_rotation_error_deg"]<=15 for x in rows),
            "above_table_5mm":bool(rows) and all(x["object_bottom_m"]>.005 for x in rows),
            "hand_contact_present":bool(rows) and np.mean([x["hand_object_contacts"]>0 for x in rows])>=.95,
            "penetration_within_3mm":bool(rows) and all(x["hand_object_penetration_m"]<=.003 for x in rows)}
    checks={key:bool(value) for key,value in checks.items()}
    return bool(all(checks.values())),checks


def append_continuous_controls(previous, appended):
    """Upstream blend_reference rule: preserve the optimized splice boundary."""
    if len(previous) and len(appended):
        appended=appended.clone()
        delta=previous[-1]-appended[0]
        decay=torch.linspace(1.,0.,len(appended),device=appended.device)[:,None]
        appended+=decay*delta
    return torch.cat([previous,appended],dim=0)


def seed_recorded_controls(controls, trial_name, dt):
    """Warm start from recorded controls only; never copy future states."""
    chain=[]
    current=trial_name
    while current:
        if current in chain:
            raise ValueError('Cyclic control-seed lineage')
        chain.append(current)
        current=json.loads((ROOT/current/'parameters.json').read_text()).get('resume_from')
    chain.reverse()
    controls=np.array(controls,copy=True)
    counts={}
    last_index=-1
    last_control=None
    for i,name in enumerate(chain):
        path=ROOT/name/'native_trajectory.npz'
        if not path.exists():
            continue
        saved=np.load(path)
        cutoff=(float(np.load(ROOT/chain[i+1]/'trajectory_physics.npz')['time'][0])
                if i+1<len(chain) else float('inf'))
        # A recorded control at t was applied over (t-dt,t].
        indices=np.rint(saved['time']/dt).astype(int)-1
        keep=(saved['time']<=cutoff+1e-7)&(indices>=0)&(indices<len(controls))
        controls[indices[keep]]=saved['ctrl'][keep]
        counts[name]=int(keep.sum())
        if keep.any() and int(indices[keep][-1])>last_index:
            last_index=int(indices[keep][-1])
            last_control=saved['ctrl'][keep][-1]
    result_path=ROOT/trial_name/'result.json'
    if (last_control is not None and result_path.exists()
            and json.loads(result_path.read_text()).get('completed_full_reference')):
        # Keep the optimized final grip through the padded planning interval.
        controls[last_index+1:]=last_control
    return controls,counts


def prediction_feasible_mask(metrics,scores):
    return (torch.isfinite(scores) & (metrics['tracking_bad']<1e-6)
            & (metrics['penetration_bad']<1e-6) & (metrics['contact_miss']<=.02)
            & (metrics['unexpected_floor']<1e-6))


def native_state_failure(model,data,q,reference,object_geoms,hand_geoms,vertices,airborne):
    """Check the actual post-integration state before committing a candidate."""
    if not np.isfinite(q).all():return 'non-finite state'
    data.qpos[:]=q
    mujoco.mj_forward(model,data)
    penetration=max((max(0.,-float(c.dist)) for c in data.contact
        if (int(c.geom[0]) in object_geoms and int(c.geom[1]) in hand_geoms)
        or (int(c.geom[1]) in object_geoms and int(c.geom[0]) in hand_geoms)),default=0.)
    if penetration>=.005:return f'penetration {penetration*1000:.3f} mm'
    if airborne:
        position=float(np.linalg.norm(q[-14:-11]-reference[-14:-11]))
        if position>=.05:return f'position error {position*1000:.3f} mm'
        actual=Rotation.from_quat(q[[-10,-9,-8,-11]])
        target=Rotation.from_quat(reference[[-10,-9,-8,-11]])
        rotation=float(np.degrees((actual.inv()*target).magnitude()))
        if rotation>=30:return f'rotation error {rotation:.3f} deg'
        bottom=float((vertices@actual.as_matrix().T+q[-14:-11])[:,2].min())
        if bottom<=.005:return f'object bottom {bottom*1000:.3f} mm'
    return None


def capture_dynamic_state(env):
    names=['qpos','qvel','qacc','time','ctrl','act','act_dot','qacc_warmstart',
           'qfrc_applied','xfrc_applied','mocap_pos','mocap_quat']
    return {name:wp.to_torch(getattr(env.data_wp,name)).clone() for name in names
            if getattr(env.data_wp,name).size}


def restore_dynamic_state(env,state):
    for name,value in state.items():wp.to_torch(getattr(env.data_wp,name)).copy_(value)


def contact_aware_reward(config,env,ref,airborne,loss_scale,tracking_guard_scale=0.):
    """Same pose/velocity/penetration objective, one contact-buffer readback.

    This pipeline has no object actuators, pedestal costs, or reference contact
    sites. A shared-object contact-loss term is added only in reference holds.
    """
    q=wp.to_torch(env.data_wp.qpos)
    v=wp.to_torch(env.data_wp.qvel)
    # Branchless quaternion subtraction avoids a CPU synchronization for each
    # object at every simulation substep. The meshless dummy has zero weight.
    qa=q[:,-11:-7];qb=ref[0][-11:-7]
    rw=(qa*qb).sum(dim=1,keepdim=True)
    rv=qb[0]*qa[:,1:]-qa[:,:1]*qb[1:]-torch.linalg.cross(qb[1:].expand_as(qa[:,1:]),qa[:,1:],dim=1)
    sine=torch.linalg.vector_norm(rv,dim=1,keepdim=True)
    angle=2*torch.atan2(sine,rw)
    angle=torch.where(angle>torch.pi,angle-2*torch.pi,angle)
    rotation=rv*angle/torch.clamp(sine,min=1e-12)
    weights=config._qpos_weight_cache
    diff=torch.cat([(q[:,:config.nu]-ref[0][:config.nu])*weights[:config.nu],
                    (q[:,-14:-11]-ref[0][-14:-11])*weights[-12:-9],rotation*weights[-9:-6]],dim=1)
    pd=torch.linalg.vector_norm(diff,dim=1)
    vd=torch.linalg.vector_norm(v-ref[1],dim=1)
    pen=torch.zeros(env.num_worlds,device=config.device)
    hand=torch.zeros_like(pen)
    floor=torch.zeros_like(pen)
    contacts=env.data_wp.contact
    dist=wp.to_torch(contacts.dist).reshape(-1)
    geoms=wp.to_torch(contacts.geom).reshape(-1,2).long().clamp(0,len(config._is_hand_geom)-1)
    ids=wp.to_torch(contacts.worldid).reshape(-1).long().clamp(0,env.num_worlds-1)
    if len(dist):
        if not hasattr(config,'_contact_slot_indices') or len(config._contact_slot_indices)!=len(dist):
            config._contact_slot_indices=torch.arange(len(dist),device=config.device)
        valid=config._contact_slot_indices<wp.to_torch(env.data_wp.nacon).reshape(-1)[0]
        g1,g2=geoms[:,0],geoms[:,1]
        ho=valid&((config._is_hand_geom[g1]&config._is_right_object_geom[g2])|(config._is_right_object_geom[g1]&config._is_hand_geom[g2]))
        fo=valid&((config._is_floor_geom[g1]&config._is_right_object_geom[g2])|(config._is_right_object_geom[g1]&config._is_floor_geom[g2]))
        touching=dist<=0
        depths=torch.nan_to_num(torch.clamp(-dist-config.penetration_margin,min=0.))**2
        pen.scatter_reduce_(0,ids,torch.where(ho,depths,0.),reduce='amax',include_self=True)
        hand.scatter_reduce_(0,ids,(touching&ho).float(),reduce='amax',include_self=True)
        floor.scatter_reduce_(0,ids,(touching&fo).float(),reduce='amax',include_self=True)
    pen*=config.penetration_penalty_scale
    lost=loss_scale*((1-hand)+floor) if airborne else torch.zeros_like(pen)
    position_error=torch.linalg.vector_norm(q[:,-14:-11]-ref[0][-14:-11],dim=1)
    rotation_error=angle.abs().reshape(-1)
    guard=tracking_guard_scale*(torch.clamp(position_error/.025-1,min=0.)**2
                +torch.clamp(rotation_error/np.deg2rad(15)-1,min=0.)**2) if airborne else torch.zeros_like(pen)
    value=-pd-config.vel_rew_scale*vd-pen-lost-guard
    tracking_bad=((position_error>.035) | (rotation_error>np.deg2rad(20))).float()
    penetration_bad=(pen>config.penetration_penalty_scale*(.005-config.penetration_margin)**2).float()
    return value,{'qpos_dist':pd,'qvel_dist':vd,'pen_penalty':pen,'contact_loss_penalty':lost,
                  'tracking_guard_penalty':guard,
                  'tracking_bad':tracking_bad,'penetration_bad':penetration_bad,
                  'contact_miss':1-hand if airborne else torch.zeros_like(hand),
                  'unexpected_floor':floor if airborne else torch.zeros_like(floor)}


def make_reference(raw, model, first_contact, anchor=100, dwell_frames=87):
    """Preserve source frames, inserting only a static validation hold at anchor.

    Fingers start open and smoothly recover the reference before optimization.
    Quaternion resampling uses SLERP; the 30Hz source times remain exact.
    """
    source = np.array(raw, copy=True)
    start_frame = max(0, first_contact - 9)
    for t in range(start_frame + 1):
        x = t / max(1, start_frame)
        alpha = x*x*(3-2*x)
        source[t, 6:28] *= alpha
        source[t, 34:56] *= alpha
    # A meshless second object is a structural placeholder, never a task target.
    source[:, -7:] = [0, 0, -2, 1, 0, 0, 0]
    rows = np.concatenate([source[:anchor+1], np.repeat(source[anchor:anchor+1],dwell_frames,axis=0), source[anchor+1:]])
    source_indices = np.r_[np.arange(anchor+1),np.full(dwell_frames,anchor),np.arange(anchor+1,len(source))]
    sim_dt = 1/150
    old_t = np.arange(len(rows))/30
    new_t = np.arange((len(rows)-1)*5+1)*sim_dt
    ref = np.column_stack([np.interp(new_t,old_t,rows[:,j]) for j in range(model.nq)])
    for a in [model.nq-11,model.nq-4]:
        r = Rotation.from_quat(rows[:,[a+1,a+2,a+3,a]])
        xyzw = Slerp(old_t,r)(new_t).as_quat()
        ref[:,a:a+4] = xyzw[:,[3,0,1,2]]
    velocity = np.zeros((len(ref),model.nv))
    for i in range(len(ref)-1):
        mujoco.mj_differentiatePos(model,velocity[i],sim_dt,ref[i],ref[i+1])
    velocity[-1] = velocity[-2]
    release_frame = anchor + 12
    schedule = {"first_contact_frame":first_contact,"optimizer_start_frame":start_frame,
                "anchor_source_frame":anchor,"release_frame":release_frame,
                "full_gravity_frame":release_frame+15,
                "gate_frame":anchor+dwell_frames,
                "source_frames":len(source),"simulation_dt":sim_dt,
                "inserted_hold_seconds":dwell_frames/30,
                "kinematic_approach_ends_s":start_frame/30,
                "release_s":release_frame/30,"full_gravity_s":(release_frame+15)/30,
                "gate_s":(anchor+dwell_frames)/30,
                "physical_hold_seconds":(dwell_frames-27)/30,
                "object_weld_tracks_source_before_release":True,
                "source_mapping":source_indices.tolist()}
    return ref,velocity,schedule


class PhaseSchedule:
    def __init__(self, config, env, reference, schedule):
        self.config,self.env,self.reference,self.schedule=config,env,reference,schedule
        self.real_eq=env.model_cpu.equality("right_object_weld").id
        self.dummy_eq=env.model_cpu.equality("left_object_weld").id
        self.real_body=env.model_cpu.body("right_object").id
        self.dummy_body=env.model_cpu.body("left_object").id
        self.eq_active=wp.to_torch(env.data_wp.eq_active)
        self.eq_data=wp.to_torch(env.model_wp.eq_data)
        self.gravity=wp.to_torch(env.model_wp.opt.gravity)
        self.constant_unassisted=False
        # Equality-to-world transform, for the fixed anchor used by this scene.
        self.targets={}
        for body,eq in [(self.real_body,self.real_eq),(self.dummy_body,self.dummy_eq)]:
            adr=env.model_cpu.jnt_qposadr[env.model_cpu.body_jntadr[body]]
            poses=reference[:,adr:adr+7]
            rot=Rotation.from_quat(poses[:,[4,5,6,3]])
            anchor=self.eq_data[0,eq,:3].cpu().numpy()
            relp=rot.inv().apply(anchor-poses[:,:3])
            quat=rot.inv().as_quat()[:,[3,0,1,2]]
            self.targets[eq]=torch.as_tensor(np.c_[relp,quat],dtype=torch.float32,device=config.device)

    def apply(self,index):
        if self.constant_unassisted:
            return False,1.
        index=max(0,min(int(index),len(self.reference)-1))
        sec=index*self.config.sim_dt
        locked,ramp=phase_at(sec,self.schedule)
        self.eq_active[:,self.real_eq]=locked
        self.eq_active[:,self.dummy_eq]=True
        self.eq_data[0,self.dummy_eq,3:10]=self.targets[self.dummy_eq][index]
        if locked:
            self.eq_data[0,self.real_eq,3:10]=self.targets[self.real_eq][index]
        self.gravity[0,2]=-9.81*ramp
        return locked,ramp

    def step(self, config, env, ctrl, perturb=True, sim_step=None):
        self.apply(sim_step if sim_step is not None else config.warmup_sim_step)
        physics.step_env(config,env,ctrl,perturb=perturb,sim_step=sim_step)


def run(args):
    if args.commit_seconds is not None:
        assert args.native_guard and 0<args.commit_seconds<=args.horizon
    ROOT.mkdir(exist_ok=True)
    trial=ROOT/f"trial_{args.name}"
    trial.mkdir(exist_ok=True)
    shutil.copy2(__file__,trial/"pipeline_source.py")
    torch.set_num_threads(4)
    # Torch reads Warp's live state arrays. Use one non-default CUDA stream
    # for simulation and scoring so removing host readbacks cannot race them.
    shared_cuda_stream=wp.stream_to_torch(wp.get_stream('cuda:0'))
    torch.cuda.set_stream(shared_cuda_stream)
    live.init_viser("BIHOI Steel Bowl — Contact Initialization",port=8080)
    server=live._get_server()
    status=server.gui.add_markdown("**准备物理场景 / Preparing physics scene**")
    pause=server.gui.add_checkbox("暂停计算 / Pause",initial_value=False)
    stop=server.gui.add_checkbox("停止本次试验 / Stop",initial_value=False)
    metrics_view=server.gui.add_markdown("尚未开始物理优化。")
    server.gui.add_markdown("灰色机械手：物理状态；半透明模型：参考。先观察接触，再检查无辅助承重。")
    @server.on_client_connect
    def camera(client):
        client.camera.position=(.65,-.65,.6)
        client.camera.look_at=(.16,-.12,.12)
        client.camera.up_direction=(0,0,1)
    directory=ROOT/"sharpa/bimanual"/TASK/"0"
    scene=directory/"scene.xml"
    if not scene.exists():
        generate_scene(output_root_dir=str(ROOT),dataset_name="bihoi",robot_type="sharpa",
                       embodiment_type="bimanual",task=TASK,force=True,show_viewer=False,
                       use_visual_mesh_as_collision=False,object_floor_collision=True,
                       hand_floor_collision=False,object_armature=0,object_damping=0,
                       object_frictionloss=0)
        shutil.copy2(directory/"scene_ik.xml",scene)
    # Saving a modified compiled MjModel does not serialize inferred body inertia.
    # Scale the source geom densities instead, then verify the recompiled mass.
    mass_model=mujoco.MjModel.from_xml_path(str(scene))
    mass_scale=args.mass/mass_model.body_mass[mass_model.body("right_object").id]
    if abs(mass_scale-1)>1e-10:
        tree=ET.parse(scene)
        body=tree.find(".//body[@name='right_object']")
        inertia=body.find("inertial")
        if inertia is not None:
            inertia.set("mass",str(args.mass))
            for attr in ["diaginertia","fullinertia"]:
                if attr in inertia.attrib:
                    inertia.set(attr," ".join(str(float(x)*mass_scale) for x in inertia.get(attr).split()))
        else:
            for geom in body.findall("geom"):
                geom.set("density",str(float(geom.get("density","1000"))*mass_scale))
        tree.write(scene,encoding="unicode")
    checked=mujoco.MjModel.from_xml_path(str(scene))
    assert abs(checked.body_mass[checked.body("right_object").id]-args.mass)<1e-8
    config=Config(output_root_dir=str(ROOT),dataset_name="bihoi",task=TASK,robot_type="sharpa",
        embodiment_type="bimanual",sim_dt=1/150,ctrl_dt=args.ctrl_dt,ref_dt=1/30,horizon=args.horizon,knot_dt=.2,
        num_samples=args.samples,max_num_iterations=args.iterations,temperature=.5,
        improvement_threshold=0.0002,improvement_check_steps=3,
        base_pos_rew_scale=.3,base_rot_rew_scale=.05,joint_rew_scale=.006,
        pos_rew_scale=3,rot_rew_scale=args.object_rotation_weight,terminal_rew_scale=10,
        penetration_penalty_scale=args.penetration_weight,penetration_margin=.003,
        pos_noise_scale=.01,rot_noise_scale=.03,joint_noise_scale=.12,
        first_ctrl_noise_scale=1,last_ctrl_noise_scale=2,final_noise_scale=.05,
        use_torch_compile=False,nconmax_per_env=400,njmax_per_env=1800,
        num_perturb_samples=1,warmup_duration=0,show_viewer=False,viewer="",save_video=False,
        save_info=True,force=True)
    config=process_config(config)
    config.output_dir=str(trial)
    model=physics.setup_mj_model(config)
    raw=np.load(directory/"trajectory_kinematic.npz")["qpos"]
    analysis=json.loads((ROOT/"contact_analysis.json").read_text())
    first=min(v["first_persistent_contact_frame"] for v in analysis["hands"].values() if v["first_persistent_contact_frame"] is not None)
    qref,vref,schedule=make_reference(raw,model,first)
    if args.max_seconds:
        limit=min(len(qref)-1,round(args.max_seconds/config.sim_dt))
    else:
        limit=len(qref)-1
    config.max_sim_steps=limit
    dump(trial/"schedule.json",schedule)
    dump(trial/"parameters.json",{"mass_kg":args.mass,"mass_is_assumed":True,"samples":args.samples,
         "iterations":args.iterations,"dt":config.sim_dt,"horizon_s":config.horizon,
         "ctrl_dt":config.ctrl_dt,"resume_from":args.resume_from,
         "resume_at":args.resume_at,"continuous_control_splice":True,
         "verify_final_controls":args.verify_controls,"contact_loss_penalty":args.contact_penalty,
         "shared_warp_torch_cuda_stream":True,
         "fast_reward":args.fast_reward,
         "object_rotation_weight":args.object_rotation_weight,
         "penetration_weight":args.penetration_weight,
         "seed_controls_from":args.seed_controls_from or None,
         "validate_warm_start_before_sampling":args.validate_warm_start,
         "tracking_guard_scale":args.tracking_guard,
         "native_execution_guard":args.native_guard,
         "native_guard_checks_full_forecast":args.native_guard,
         "verified_commit_seconds":args.commit_seconds,
         "execute_verified_terminal_plan":args.certified_tail,
         "feasible_prediction_early_stop":args.feasible_stop,
         "friction":"upstream hand-object default, sliding=1.0",
         "gate":{"hold_s":2,"max_position_error_m":.02,"max_rotation_error_deg":15,
                 "max_penetration_m":.003,"minimum_bottom_height_m":.005}})
    # Pad references for the MPC horizon without extending the executed trial.
    pad=config.horizon_steps+config.ctrl_steps+1
    qp=np.r_[qref,np.repeat(qref[-1:],pad,axis=0)]
    vp=np.r_[vref,np.zeros((pad,model.nv))]
    cp=qp[:,:model.nu].copy()
    if args.seed_controls_from:
        cp,seed_counts=seed_recorded_controls(cp,args.seed_controls_from,config.sim_dt)
        dump(trial/'control_seed.json',{'source_trial':args.seed_controls_from,
             'selected_native_controls':seed_counts,'future_states_copied':False})
    ref_data=tuple(torch.as_tensor(a,dtype=torch.float32,device=config.device) for a in
                   [qp,vp,cp,np.zeros((len(qp),10)),np.zeros((len(qp),10,3))])
    np.savez_compressed(trial/"reference.npz",qpos=qref,qvel=vref)
    _,_,handles=live.build_and_log_scene(scene)
    view_model=mujoco.MjModel.from_xml_path(str(scene));view_data=mujoco.MjData(view_model);ghost=mujoco.MjData(view_model)
    view_model.vis.global_.offwidth=720
    view_model.vis.global_.offheight=480
    renderer=mujoco.Renderer(view_model,height=480,width=720)
    video=imageio.get_writer(str(trial/"physics.mp4"),fps=30,codec="libx264",quality=7)
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=qref[min(len(qref)-1,500),-14:-11]+[0,0,.02]
    cam.distance=.65;cam.azimuth=135;cam.elevation=-25
    resume=None
    if args.resume_from:
        resume_dir=ROOT/args.resume_from
        prior_gate=json.loads((resume_dir/"grasp_gate.json").read_text())
        if not prior_gate["passed"]:
            raise ValueError("Cannot resume task motion from an unverified grasp")
        resume=dict(np.load(resume_dir/"trajectory_physics.npz"))
        if args.resume_at is not None:
            keep=resume["time"]<=args.resume_at+1e-8
            if not keep.any():
                raise ValueError("Resume time precedes saved physical states")
            resume={k:(v[keep] if getattr(v,'ndim',0)>0 and len(v)==len(keep) else v) for k,v in resume.items()}
        dump(trial/"grasp_gate.json",prior_gate|{"inherited_from":str(resume_dir)})
    env=None
    rows=[];qs=[];vs=[];us=[];times=[]
    gate_rows=[];gate_passed=resume is not None;failure=None
    stage="RESUME VERIFIED GRASP" if resume is not None else "KINEMATIC APPROACH"
    wall_start=time.monotonic()
    next_gate_index=round(schedule["gate_s"]/config.sim_dt)
    gate_start=round(schedule["full_gravity_s"]/config.sim_dt)
    real_bid=model.body("right_object").id
    object_geoms=set(np.flatnonzero(model.geom_bodyid==real_bid).tolist())
    hand_geoms={g for g in range(model.ngeom) if (model.geom(g).name or "").startswith("collision_hand_")}
    surface_vertices=np.asarray(__import__("trimesh").load(ROOT/"assets/objects/bowl/source_mesh.ply",process=False).convex_hull.vertices)
    # Map MANO contact hints onto the inserted-hold timeline, without changing poses.
    source_contact=list(csv.DictReader((ROOT/'contact_analysis.csv').open()))
    source_index=np.asarray(schedule['source_mapping'])[np.minimum(np.rint(np.arange(len(qp))*config.sim_dt*30).astype(int),len(schedule['source_mapping'])-1)]
    expected_held=np.array([min(float(source_contact[i]['right_min_distance_m']),float(source_contact[i]['left_min_distance_m']))<.015 for i in source_index])
    ref_bottom=np.array([float(source_contact[i]['object_bottom_m']) for i in source_index])
    expected_airborne=expected_held & (ref_bottom>.02)
    loss_steps=0
    native_rejections=0
    native_probe=mujoco.MjData(view_model)
    native_qs=[];native_vs=[];native_us=[];native_times=[]

    def record(index,q,v,u,locked,ramp):
        nonlocal stage
        sec=index*config.sim_dt
        view_data.qpos[:]=q;view_data.qvel[:]=v;view_data.ctrl[:]=u
        view_model.opt.gravity[2]=-9.81*ramp
        # Viewer/contact diagnostics use the same physical state, without weld assistance.
        view_data.eq_active[:]=False
        mujoco.mj_forward(view_model,view_data)
        ghost.qpos[:]=qp[index];mujoco.mj_forward(view_model,ghost)
        pos_error=float(np.linalg.norm(q[-14:-11]-qp[index,-14:-11]))
        aq=Rotation.from_quat(q[[-10,-9,-8,-11]])
        bq=Rotation.from_quat(qp[index,[model.nq-10,model.nq-9,model.nq-8,model.nq-11]])
        angle=float(np.degrees((aq.inv()*bq).magnitude()))
        bottom=float((surface_vertices@aq.as_matrix().T+q[-14:-11])[:,2].min())
        ncontact=0;penetration=0.;normal_force=0.
        for contact in view_data.contact:
            a,b=map(int,contact.geom)
            if (a in object_geoms and b in hand_geoms) or (b in object_geoms and a in hand_geoms):
                if contact.dist<=.0005:
                    ncontact+=1
                penetration=max(penetration,float(-contact.dist))
        row={"step":index,"time_s":sec,"phase":stage,"weld_active":locked,
             "gravity_fraction":ramp,"object_position_error_m":pos_error,
             "object_rotation_error_deg":angle,"object_bottom_m":bottom,
             "hand_object_contacts":ncontact,"hand_object_penetration_m":penetration}
        rows.append(row)
        if gate_start<=index<=next_gate_index:
            gate_rows.append(row)
        qs.append(q.copy());vs.append(v.copy());us.append(u.copy());times.append(sec)
        live.log_frame(view_data,sec,handles,data_ref=ghost,record=True,playback_fps=30)
        if len(qs)==1 and live._STATE.playback_checkbox is not None:
            live._STATE.playback_checkbox.value=True
        live.update_sim_progress(index,limit)
        metrics_view.content=(f"**{stage}** · t={sec:.2f}s\n\n"
           f"重力 {ramp*100:.0f}% · 物体锁定 {locked}\n\n"
           f"手—物体接触 {ncontact} · 穿透 {penetration*1000:.1f} mm\n\n"
           f"物体偏差 {pos_error*1000:.1f} mm / {angle:.1f}° · 离桌 {bottom*1000:.1f} mm")
        renderer.update_scene(view_data,camera=cam)
        frame=renderer.render()
        cv2.putText(frame,f"{stage}  t={sec:.2f}s",(12,26),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,f"gravity={ramp:.2f} locked={int(locked)} contacts={ncontact} error={pos_error*1000:.1f}mm",(12,51),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        video.append_data(frame)
        return row

    def checkpoint():
        np.savez_compressed(trial/"trajectory_physics.npz",qpos=np.asarray(qs),qvel=np.asarray(vs),ctrl=np.asarray(us),time=np.asarray(times),frequency=30.)
        with (trial/"diagnostics.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

    def save_native():
        if native_times:
            np.savez_compressed(trial/'native_trajectory.npz',qpos=np.asarray(native_qs,dtype=np.float32),
                qvel=np.asarray(native_vs,dtype=np.float32),ctrl=np.asarray(native_us,dtype=np.float32),
                time=np.asarray(native_times),frequency=1/config.sim_dt)

    try:
        status.content="**张手接近 / Kinematic approach**"
        start=round(schedule["kinematic_approach_ends_s"]/config.sim_dt)
        if resume is not None:
            start=round(float(resume["time"][-1])/config.sim_dt)
            record(start,resume["qpos"][-1],resume["qvel"][-1],resume["ctrl"][-1],False,1.)
        else:
            for i in range(0,min(start,limit)+1,5):
                record(i,qp[i],vp[i],cp[i],True,0.)
        status.content="**正在编译 GPU 物理内核；完成后自动开始接触优化。**"
        init=tuple(r[start:] for r in ref_data)
        env=physics.setup_env(config,init)
        # setup_env compiles by stepping: explicitly restore the exact pre-contact state.
        initial_q=ref_data[0][start] if resume is None else torch.as_tensor(resume["qpos"][-1],device=config.device,dtype=torch.float32)
        physics.set_qpos(config,env,initial_q)
        wp.to_torch(env.data_wp.qvel)[:]=0 if resume is None else torch.as_tensor(resume["qvel"][-1],device=config.device,dtype=torch.float32)
        wp.to_torch(env.data_wp.qacc_warmstart)[:]=0
        mjwarp.forward(env.model_wp,env.data_wp)
        physics.precompute_hand_object_geom_mask(config,env.model_cpu)
        # The meshless dummy is excluded from both object-pose and velocity costs.
        config._qpos_weight_cache=physics._weight_diff_qpos(config)
        config._qpos_weight_cache[-6:]=0
        scheduler=PhaseSchedule(config,env,qp,schedule)
        if start*config.sim_dt>=schedule['full_gravity_s']:
            # Every rollout starts after release. Avoid rewriting identical
            # gravity/equality arrays on every substep; restoration is identical.
            scheduler.apply(start)
            scheduler.constant_unassisted=True
        base_reward=physics.get_reward
        def reward(c,e,ref,sim_step=0):
            # Dummy velocities are zero because the placeholder stays welded.
            if args.fast_reward:
                return contact_aware_reward(c,e,ref,bool(expected_airborne[min(sim_step,len(expected_airborne)-1)]),args.contact_penalty,args.tracking_guard)
            value,details=base_reward(c,e,ref,sim_step=sim_step)
            if args.contact_penalty>0 and expected_airborne[min(sim_step,len(expected_airborne)-1)]:
                touching=physics._pair_contact_worlds(e,c._is_hand_geom,c._is_right_object_geom)
                on_floor=physics._pair_contact_worlds(e,c._is_floor_geom,c._is_right_object_geom)
                penalty=args.contact_penalty*((~touching).float()+on_floor.float())
                value=value-penalty
                details['contact_loss_penalty']=penalty
            else:
                details['contact_loss_penalty']=torch.zeros(e.num_worlds,device=c.device)
            return value,details
        base_rollout=make_rollout_fn(scheduler.step,physics.save_state,physics.load_state,reward,
                 lambda c,e,r,sim_step=0:(lambda x:(x[0]*c.terminal_rew_scale,x[1]))(reward(c,e,r,sim_step)),
                 physics.get_trace,physics.save_env_params,physics.load_env_params)
        last_candidates={}
        rejected_prefixes=[]
        def rollout(*call_args,**call_kwargs):
            answer=base_rollout(*call_args,**call_kwargs)
            scores=answer[1]
            for prefix in rejected_prefixes:
                same=(answer[0][:,:len(prefix)]-prefix).abs().flatten(1).amax(1)<1e-6
                scores=torch.where(same,-torch.inf,scores)
            answer=(answer[0],scores,answer[2])
            finite_scores=torch.where(torch.isfinite(answer[1]),answer[1],-torch.inf)
            last_candidates['best']=answer[0][torch.argmax(finite_scores)].clone()
            if args.feasible_stop:
                metrics=answer[2]
                valid=prediction_feasible_mask(metrics,answer[1])
                if bool(valid.any()):
                    scores=torch.where(valid,answer[1],-torch.inf)
                    idx=int(torch.argmax(scores));score=float(scores[idx])
                    if score>last_candidates.get('feasible_score',-float('inf')):
                        last_candidates['feasible']=answer[0][idx].clone()
                        last_candidates['feasible_score']=score
            return answer
        optimize_once=make_optimize_once_fn(rollout)
        optimize=make_optimize_fn(optimize_once)
        if args.feasible_stop:
            assert args.fast_reward, 'Feasibility checks require the equivalent fast reward'
            def optimize(c,e,controls,refs,callback=None):
                if args.validate_warm_start and not rejected_prefixes:
                    repeated=controls[None].repeat(c.num_samples,1,1)
                    _,scores,metrics=rollout(c,e,repeated,refs,{})
                    if bool(prediction_feasible_mask(metrics,scores).all()):
                        if callback:callback(1,1)
                        return controls,{'opt_steps':np.array([0]),'feasible':True,
                                         'validated_warm_start':True}
                for iteration in range(c.max_num_iterations):
                    controls,_=optimize_once(c,e,controls,refs,c.env_params_list[iteration],
                                             {'global_noise_scale':c.beta_traj**iteration})
                    if callback:callback(iteration+1,c.max_num_iterations)
                    if iteration>=2 and last_candidates.get('feasible') is not None:
                        return last_candidates['feasible'],{'opt_steps':np.array([iteration+1]),'feasible':True}
                return controls,{'opt_steps':np.array([c.max_num_iterations]),'feasible':False}
        config.env_params_list=[[{}] for _ in range(config.max_num_iterations)]
        ctrl=ref_data[2][start:start+config.horizon_steps].clone()
        if resume is not None and not args.seed_controls_from:
            residual=torch.as_tensor(resume['ctrl'][-1],device=config.device,dtype=torch.float32)-ctrl[0]
            decay=torch.linspace(1.,0.,len(ctrl),device=config.device)[:,None]
            ctrl+=decay*residual
        sim=start
        while sim<limit:
            while pause.value and not stop.value:
                time.sleep(.2)
            if stop.value:
                failure="Stopped by user";break
            config.warmup_sim_step=sim
            stage=("CONTACT OPTIMIZATION" if sim*config.sim_dt<schedule["release_s"] else
                   "GRAVITY RELEASE" if sim*config.sim_dt<schedule["full_gravity_s"] else
                   "UNSUPPORTED HOLD TEST" if sim<next_gate_index else "TASK MOTION")
            status.content=f"**{stage}** — GPU MPC is computing the next {config.ctrl_dt:g} seconds."
            ref_slice=tuple(r[sim+1:sim+config.horizon_steps+1] for r in ref_data)
            last_candidates.pop('feasible',None);last_candidates.pop('feasible_score',None)
            ctrl,info=optimize(config,env,ctrl,ref_slice,live.update_opt_progress)
            if args.verify_controls and not info.get('feasible',False):
                # A mean of successful contact modes can itself fail. Evaluate
                # the returned mean and the best final sample before execution.
                best=last_candidates['best']
                half=config.num_samples//2
                candidates=torch.cat([ctrl[None].repeat(half,1,1),best[None].repeat(config.num_samples-half,1,1)])
                _,scores,_=base_rollout(config,env,candidates,ref_slice,{})
                if scores[half:].mean()>scores[:half].mean():
                    ctrl=best
            n=min(config.ctrl_steps,limit-sim)
            if args.commit_seconds is not None and info.get('feasible',False):
                n=min(round(args.commit_seconds/config.sim_dt),limit-sim)
            if args.certified_tail and info.get('feasible',False) and limit-sim<=config.horizon_steps:
                n=limit-sim
            pending=None
            if args.native_guard and gate_passed:
                saved_params=physics.save_env_params(config,env)
                saved_state=physics.save_state(env)
                pending=[];rejected=None;committed_state=None
                probe_steps=min(config.horizon_steps,limit-sim)
                for k in range(probe_steps):
                    at=sim+k
                    scheduler.step(config,env,ctrl[k],perturb=False,sim_step=at+1)
                    q=physics.get_qpos(config,env)[0].detach().cpu().numpy().copy()
                    v=physics.get_qvel(config,env)[0].detach().cpu().numpy().copy()
                    u=ctrl[k].detach().cpu().numpy().copy()
                    rejected=native_state_failure(view_model,native_probe,q,qp[at+1],
                        object_geoms,hand_geoms,surface_vertices,bool(expected_airborne[at+1]))
                    if rejected is None and not np.isfinite(v).all():rejected='non-finite velocity'
                    if rejected:break
                    if k<n:pending.append((q,v,u))
                    if k==n-1 and probe_steps>n:committed_state=capture_dynamic_state(env)
                if rejected:
                    physics.load_state(env,saved_state)
                    physics.load_env_params(config,env,saved_params)
                    native_rejections+=1
                    rejected_prefixes.append(ctrl[:k+1].clone())
                    print(f'NATIVE_REJECT t={(at+1)*config.sim_dt:.6f} reason={rejected} attempt={native_rejections}',flush=True)
                    status.content=f'候选未提交，重新搜索：{rejected}'
                    if native_rejections>=16:
                        failure=f'No candidate passed native execution checks after 16 attempts: {rejected}'
                        break
                    continue
                if committed_state is not None:restore_dynamic_state(env,committed_state)
                native_rejections=0
                rejected_prefixes.clear()
            for k in range(n):
                at=sim+k
                if pending is None:
                    scheduler.step(config,env,ctrl[k],perturb=False,sim_step=at+1)
                    q=physics.get_qpos(config,env)[0].detach().cpu().numpy().copy()
                    v=physics.get_qvel(config,env)[0].detach().cpu().numpy().copy()
                    u=ctrl[k].detach().cpu().numpy().copy()
                else:q,v,u=pending[k]
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    failure="Non-finite physical state";break
                native_qs.append(q);native_vs.append(v);native_us.append(u);native_times.append((at+1)*config.sim_dt)
                error=float(np.linalg.norm(q[-14:-11]-qp[at+1,-14:-11]))
                loss_steps=loss_steps+1 if expected_airborne[at+1] and error>.06 else 0
                if loss_steps>=round(.2/config.sim_dt):
                    failure=f"Sustained object tracking loss (>6 cm for 0.2s) at {(at+1)*config.sim_dt:.3f}s"
                    if (at+1)%5:
                        locked,ramp=scheduler.apply(at+1);record(at+1,q,v,u,locked,ramp)
                    break
                if (at+1)%5==0:
                    locked,ramp=scheduler.apply(at+1)
                    record(at+1,q,v,ctrl[k].detach().cpu().numpy(),locked,ramp)
                if at+1>=next_gate_index and not gate_passed:
                    gate_passed,checks=evaluate_grasp_gate(gate_rows)
                    dump(trial/"grasp_gate.json",{"passed":gate_passed,"checks":checks,"samples":len(gate_rows)})
                    if not gate_passed:
                        failure="Initial grasp gate failed: "+", ".join(k for k,v in checks.items() if not v)
                        break
                if at+1>gate_start and q[-12]<-.15:
                    failure="Object fell below the tabletop";break
            sim+=n
            print(f"PROGRESS t={sim*config.sim_dt:.3f} phase={stage} gate={gate_passed} iterations={int(info['opt_steps'][0])} feasible={info.get('feasible',False)} wall={time.monotonic()-wall_start:.1f}",flush=True)
            if rows:
                checkpoint();dump(trial/"progress.json",{"time_s":sim*config.sim_dt,"phase":stage,"gate_passed":gate_passed,"last":rows[-1],
                    "prediction_feasible":bool(info.get('feasible',False)),"planning_iterations":int(info['opt_steps'][0])})
            if failure:break
            physics.sync_env(config,env,view_data)
            tail=ref_data[2][sim+config.horizon_steps-n:sim+config.horizon_steps]
            ctrl=append_continuous_controls(ctrl[n:],tail)
    except Exception as exc:
        failure=f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        video.close();renderer.close()
        if rows:checkpoint()
        save_native()
        completed=bool(not failure and gate_passed and times and times[-1]>=limit*config.sim_dt-.05)
        full_reference_completed=bool(completed and limit==len(qref)-1)
        result={"status":"completed" if full_reference_completed else "window_completed" if completed else "failed" if failure else "partial",
                "completed_full_reference":full_reference_completed,
                "initial_grasp_passed":gate_passed,"failure":failure,
                "simulated_seconds":times[-1] if times else 0,"wall_seconds":time.monotonic()-wall_start,
                "task":TASK,"trial":args.name,"video":str(trial/"physics.mp4"),
                "mass_kg":args.mass,"mass_is_assumed":True,
                "original_trajectory_untouched":True,"continued_from":args.resume_from or None}
        dump(trial/"result.json",result)
        print(json.dumps(result,indent=2),flush=True)
        status.content=f"**{'完成' if completed else '未通过 / Stopped'}**\n\n{failure or 'Simulation finished'}\n\n可使用时间轴回放。"
        while args.keep_viewer:
            time.sleep(1)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--name",default="01")
    p.add_argument("--samples",type=int,default=256);p.add_argument("--iterations",type=int,default=12)
    p.add_argument("--mass",type=float,default=.3);p.add_argument("--max-seconds",type=float,default=0.)
    p.add_argument("--resume-from",default="")
    p.add_argument("--resume-at",type=float)
    p.add_argument("--verify-controls",action="store_true")
    p.add_argument("--contact-penalty",type=float,default=0.)
    p.add_argument("--object-rotation-weight",type=float,default=.3)
    p.add_argument("--penetration-weight",type=float,default=3000.)
    p.add_argument("--seed-controls-from",default="")
    p.add_argument("--validate-warm-start",action="store_true")
    p.add_argument("--tracking-guard",type=float,default=0.)
    p.add_argument("--native-guard",action="store_true")
    p.add_argument("--certified-tail",action="store_true")
    p.add_argument("--commit-seconds",type=float)
    p.add_argument("--fast-reward",action="store_true")
    p.add_argument("--feasible-stop",action="store_true")
    p.add_argument("--horizon",type=float,default=2.0);p.add_argument("--ctrl-dt",type=float,default=.2)
    p.add_argument("--keep-viewer",action="store_true")
    run(p.parse_args())
