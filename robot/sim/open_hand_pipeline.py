"""Full-gravity open-hand initialization and evidence-based physical acceptance.

All poses are simulated from the first open-hand state. Historical physics may
propose controls, never supply object states after initialization.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation, Slerp
from build_arm_scene import ROOT, OLD, TASK, SIDES, build_scene
from projector import WristProjector

IDS=np.r_[0:6,28:34]
JOIN_LEAD=9            # the approach joins the capture 0.3 s (30 Hz frames) before the human's first contact
JOIN_BEFORE=15         # if that pose is infeasible: at most this many frames earlier ...
FINGER_CLOSE_FRAMES=30 # fingers close during the last second of the approach
FINGERS=np.r_[6:28,34:56]

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def prepare_open(args):
    plan=json.loads((ROOT/args.plan_result).read_text())
    out=ROOT/'physics'/args.name;out.mkdir(parents=True,exist_ok=True)
    scene=out/'model'/'sharpa'/'bimanual'/TASK/'0';scene.mkdir(parents=True,exist_ok=True)
    model,mount=build_scene(scene/'scene.xml',plan['center_xy'],plan['heading_rad'],base_height=plan['height_m'])
    # Free object from frame zero; the only pre-grasp support is the actual table.
    model.eq_active0[model.equality('right_object_weld').id]=False
    model.opt.gravity[:]=[0,0,-9.81]
    # Store the same defaults in XML so independent replay has the same policy.
    import xml.etree.ElementTree as ET
    xml=ET.parse(scene/'scene.xml')
    xml.find(".//weld[@name='right_object_weld']").set('active','false')
    xml.write(scene/'scene.xml',encoding='unicode')
    raw_path=OLD/'sharpa/bimanual'/TASK/'0'
    raw=np.load(raw_path/'trajectory_kinematic.npz')['qpos']
    info=json.loads((ROOT/'reference_info.json').read_text())
    stop=info['source_end_frame']+1;anchor=info['source_start_frame']
    raw=raw[:stop].copy();raw[:,-7:]=[0,0,-2,1,0,0,0]
    old_model=mujoco.MjModel.from_xml_path(str(raw_path/'scene.xml'))
    od=mujoco.MjData(old_model)
    pos=np.empty((stop,2,3));rot=np.empty((stop,2,3,3))
    for i,q in enumerate(raw):
        od.qpos[:]=q;mujoco.mj_kinematics(old_model,od)
        for s,side in enumerate(SIDES):
            b=old_model.body(side+'_hand_C_MC').id
            pos[i,s]=od.xpos[b];rot[i,s]=od.xmat[b].reshape(3,3)
    planner=WristProjector(model,1/150,acceleration_limit=args.acceleration,
        jerk_limit=args.arm_jerk if args.smooth_commands else None,
        tracking_rate=args.arm_tracking_rate)
    planned=np.load(ROOT/plan['trajectory_file'])['qpos']
    # The capture's hands are unreliable while they enter the view, so they are
    # not tracked: the open-hand approach replaces them up to JOIN_LEAD frames
    # (0.3 s) before the human's first contact, where it joins the retargeted hands
    # (or at the nearest feasible frame around there, never after first contact).
    old_schedule=json.loads((OLD/'trial_accepted_v2/schedule.json').read_text())
    first_contact=int(old_schedule['first_contact_frame'])
    join=max(1,first_contact-JOIN_LEAD);lowest=max(1,join-JOIN_BEFORE)
    # A continuous IK branch is extended backwards from the selected stable
    # grasp down to the join window, then the original selected full-clip solution is retained.
    arm30=raw.copy();arm30[anchor:,:56]=planned[:stop-anchor,:56]
    for s,k in enumerate(planner.kin):
        bp=planner.bp[s];br=planner.br[s]
        p=(torch.tensor(pos[:anchor,s],device='cuda',dtype=torch.float32)-bp)@br
        R=br.T@torch.tensor(rot[:anchor,s],device='cuda',dtype=torch.float32)
        seed=torch.tensor(planned[0,s*28:s*28+6],device='cuda',dtype=torch.float32)
        for i in range(anchor-1,lowest-1,-1):
            seed,_,_=k.ik(p[i:i+1],R[i:i+1],seed.reshape(1,6),iterations=24,step_cap=.15)
            seed=seed[0]
            arm30[i,s*28:s*28+6]=seed.cpu().numpy()
    arm30[:,FINGERS]=raw[:,FINGERS]
    raw_pos=pos.copy();raw_rot=rot.copy()
    # The placement is fitted to the stable-grasp action. Generate a safe
    # approach from an open hand that joins the retargeted hands 0.3 s before first contact.
    arms={g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith(('right_arm_','left_arm_')) and model.geom_contype[g]}
    d=mujoco.MjData(model)
    def safe(row):
        d.qpos[:]=row;mujoco.mj_forward(model,d)
        return max([-float(c.dist) for c in d.contact if int(c.geom[0]) in arms or int(c.geom[1]) in arms]+[0.])<.0008
    approach_anchor=None
    # 0.3 s before first contact, then earlier frames, then later ones up to first contact.
    order=[join]+list(range(join-1,lowest-1,-1))+list(range(join+1,min(anchor,first_contact)+1))
    for i in order:
        row=arm30[i].copy();row[FINGERS]=0
        if not safe(row):continue
        errors=[np.linalg.norm(d.xpos[model.body(side+'_hand_C_MC').id]-pos[i,j]) for j,side in enumerate(SIDES)]
        if max(errors)<.003:approach_anchor=i;break
    if approach_anchor is None:raise RuntimeError(f'No collision-free reachable open-hand pose within frames {lowest}-{min(anchor,first_contact)} (join target {join}, first contact {first_contact})')
    target=arm30[approach_anchor].copy();target[FINGERS]=0
    approach=None
    for rise,retreat in [(.06,.025),(.05,.02),(.04,.015),(.025,.01),(.01,0.),(0.,0.)]:
        initial=target.copy()
        for j,k in enumerate(planner.kin):
            away=pos[approach_anchor,j]-raw[0,-14:-11]
            away[2]=0;away=away/max(np.linalg.norm(away),1e-9)
            goal=pos[approach_anchor,j]+retreat*away+np.array([0.,0.,rise])
            tp=(torch.tensor(goal,device='cuda',dtype=torch.float32)-planner.bp[j])@planner.br[j]
            tr=planner.br[j].T@torch.tensor(rot[approach_anchor,j],device='cuda',dtype=torch.float32)
            sol,e,_=k.ik(tp[None],tr[None],torch.tensor(initial[j*28:j*28+6][None],device='cuda',dtype=torch.float32),iterations=24)
            initial[j*28:j*28+6]=sol[0].cpu().numpy()
        u=np.linspace(0,1,approach_anchor+1);u=u*u*u*(10-15*u+6*u*u)
        candidate=initial[None]+u[:,None]*(target-initial)[None]
        candidate[:,-14:]=raw[:approach_anchor+1,-14:]
        if all(safe(row) for row in candidate):
            approach=candidate;break
    if approach is None:raise RuntimeError('Open-hand approach has no collision-free joint path')
    arm30[:approach_anchor+1,:56]=approach[:,:56]
    for i in range(approach_anchor+1):
        d.qpos[:]=arm30[i];mujoco.mj_kinematics(model,d)
        for j,side in enumerate(SIDES):
            b=model.body(side+'_hand_C_MC').id
            pos[i,j]=d.xpos[b];rot[i,j]=d.xmat[b].reshape(3,3)
    arm30[:,FINGERS]=raw[:,FINGERS]
    write_json(out/'approach_plan.json',{
        'kind':'collision-checked open-hand joint approach to the capture 0.3 s before first contact, with wrists derived from FK',
        'first_contact_source_frame':first_contact,'join_target_source_frame':join,'joins_raw_hand_source_frame':approach_anchor,
        'join_before_first_contact_s':(first_contact-approach_anchor)/30,
        'approach_duration_s':approach_anchor/30,'initial_rise_m':rise,'initial_radial_retreat_m':retreat,
        'finger_close_s':min(FINGER_CLOSE_FRAMES,approach_anchor)/30,
        'raw_action_reference_unchanged_from_frame':approach_anchor+1,
        'reason':'capture hands entering the view are unreliable; they are replaced by this approach until 0.3 s before first contact'})
    # Keep the collision-free approach as a control proposal. Once reference
    # contact begins, the tracking objective is the original retargeted pose,
    # including the few contact frames where raw IK alone is infeasible.
    control_pos=pos.copy();control_rot=rot.copy()
    pos[first_contact:]=raw_pos[first_contact:]
    rot[first_contact:]=raw_rot[first_contact:]
    write_json(out/'first_contact_policy.json',{
        'retarget_first_contact_source_frame':first_contact,
        'tracking_reference_raw_from_source_frame':first_contact,
        'contact_event':'actual hand-object collision in the physical simulator',
        'physics_active_before_contact':True,'gravity_ramp_seconds':0.})
    precontact=max(1,old_schedule['first_contact_frame']-9)
    # Fingers stay open, then close to the first-contact grasp over the last part of the approach.
    closing=min(FINGER_CLOSE_FRAMES,approach_anchor)
    close=np.clip((np.arange(stop)-(approach_anchor-closing))/closing,0,1)
    close=close*close*(3-2*close)
    controls30=arm30[:,:56].copy()
    # Exactly open at frame zero, then smoothly close toward retargeted fingers.
    controls30[:,FINGERS]=raw[:,FINGERS]*close[:,None]
    controls30[:first_contact+1,FINGERS]=raw[first_contact,FINGERS][None]*close[:first_contact+1,None]
    for s in range(2):
        controls30[:,s*28:s*28+3]=control_pos[:,s]
        controls30[:,s*28+3:s*28+6]=np.unwrap(Rotation.from_matrix(control_rot[:,s]).as_euler('XYZ'),axis=0)
    prior30=controls30.copy()
    if args.contact_prior:
        accepted=np.load(OLD/'trial_accepted_v2/trajectory_physics.npz')
        mapping=np.asarray(old_schedule['source_mapping'])
        indices=np.searchsorted(mapping,np.arange(stop),side='right')-1
        prior30[:,FINGERS]=accepted['ctrl'][indices][:,FINGERS]*close[:,None]
        for i,index in enumerate(indices):
            od.qpos[:]=accepted['qpos'][index];mujoco.mj_kinematics(old_model,od)
            for s,side in enumerate(SIDES):
                b=old_model.body(side+'_hand_C_MC').id
                prior30[i,s*28:s*28+3]=od.xpos[b]
                prior30[i,s*28+3:s*28+6]=Rotation.from_matrix(od.xmat[b].reshape(3,3)).as_euler('XYZ')
        prior30[:approach_anchor+1,:]=controls30[:approach_anchor+1,:]
        for s in range(2):prior30[:,s*28+3:s*28+6]=np.unwrap(prior30[:,s*28+3:s*28+6],axis=0)
    dwell=0  # user requested no stationary grasp validation stage
    mapping=np.r_[np.arange(anchor+1),np.full(dwell,anchor),np.arange(anchor+1,stop)]
    q30=arm30[mapping];p30=pos[mapping];r30=rot[mapping]
    u30=controls30[mapping];w30=prior30[mapping]
    ts30=np.arange(len(mapping))/30;ts=np.arange((len(mapping)-1)*5+1)/150
    interp=lambda x:np.stack([np.interp(ts,ts30,x[:,j]) for j in range(x.shape[1])],axis=1)
    qr=interp(q30);cp=interp(u30);cw=interp(w30)
    pp=np.empty((len(ts),2,3));rr=np.empty((len(ts),2,3,3))
    for s in range(2):
        pp[:,s]=interp(p30[:,s])
        rr[:,s]=Slerp(ts30,Rotation.from_matrix(r30[:,s]))(ts).as_matrix()
    for a in [59,66]:
        qr[:,a:a+4]=Slerp(ts30,Rotation.from_quat(q30[:,[a+1,a+2,a+3,a]]))(ts).as_quat(scalar_first=True)
    qv=np.zeros((len(ts),model.nv))
    for i in range(len(ts)-1):mujoco.mj_differentiatePos(model,qv[i],1/150,qr[i],qr[i+1])
    qv[-1]=qv[-2]
    initial=qr[0].copy();initial[FINGERS]=0
    np.savez_compressed(scene/'trajectory_kinematic.npz',qpos=qr[::5],qvel=qv[::5],frequency=30.)
    np.savez_compressed(out/'reference.npz',qpos=qr,qvel=qv,latent=cp,warm_latent=cw,wrist_pos=pp,wrist_rot=rr,
        control_wrist_pos=pp,control_wrist_rot=rr,time=ts,source_frame=np.interp(ts,ts30,mapping),raw_wrist_pos_30hz=raw_pos,raw_wrist_rot_30hz=raw_rot)
    args.open_hold_start=anchor/30
    args.contact_required_after=anchor/30+.3
    args.hold_seconds=anchor/30+dwell/30
    write_json(out/'placement.json',plan);write_json(out/'mount.json',mount)
    write_json(out/'reference_policy.json',{
        'objective_hand_reference':'unoptimized retargeted Sharpa after generated collision-free pregrasp approach',
        'objective_object_reference':'unoptimized original object poses',
        'sampling_space':'floating hand wrist SE(3) and finger controls',
        'execution_space':'UR3e joints and fingers; wrist pose comes from FK',
        'previous_physics_usage':'control proposals only; every state resimulated from open hand',
        'excluded_tail_seconds':info['excluded_tail_seconds'],
        'reference_time_scale':1.0,'source_start_frame':0,'source_end_frame':stop-1,
        'inserted_grasp_validation_hold_seconds':dwell/30,'separate_grasp_hold_test_enabled':False})
    write_json(out/'physics_policy.json',{
        'gravity_m_s2':[0,0,-9.81],'gravity_ramp_seconds':0.,
        'real_object_weld_from_first_frame':False,'external_object_force':False,
        'initial_support':'existing physical floor; no weld and no artificial support',
        'initial_fingers_exactly_open':bool(np.all(initial[FINGERS]==0)),
        'open_hold_start_s':args.open_hold_start,'motion_start_s':args.hold_seconds,
        'initial_grasp_reused_as':'control proposal only; historical state not spliced',
        'generated_approach_ends_at_source_frame':approach_anchor,
        'raw_contact_reference_starts_at_source_frame':first_contact})
    return out,model,planner,qr,qv,cp,pp,rr,initial,cw

class FullGravity:
    """Irreversibly free real object, at full gravity in every sampled world."""
    def __init__(self,config,env):
        self.config=config;self.env=env
        self.real_eq=env.model_cpu.equality('right_object_weld').id
        self.dummy_eq=env.model_cpu.equality('left_object_weld').id
        self.real_body=env.model_cpu.body('right_object').id
        self.constant_unassisted=True
    def apply(self,index):
        import warp as wp
        wp.to_torch(self.env.model_wp.opt.gravity)[:]=torch.tensor([0,0,-9.81],device=self.config.device)
        active=wp.to_torch(self.env.data_wp.eq_active)
        active[:,self.real_eq]=False;active[:,self.dummy_eq]=True
        return False,1.
    def step(self,config,env,ctrl,perturb=False,sim_step=0):
        from retargeting.utils import mjwp as physics
        self.apply(sim_step)
        physics.step_env(config,env,ctrl,perturb=False,sim_step=sim_step)

def telemetry(env,model):
    import warp as wp
    j=model.jnt_dofadr[model.body_jntadr[model.body('right_object').id]]
    b=model.body('right_object').id
    return dict(gravity=wp.to_torch(env.model_wp.opt.gravity)[0].detach().cpu().numpy().copy(),
        object_weld_active=bool(wp.to_torch(env.data_wp.eq_active)[0,model.equality('right_object_weld').id]),
        object_external_force=np.r_[wp.to_torch(env.data_wp.qfrc_applied)[0,j:j+6].detach().cpu().numpy(),
                                   wp.to_torch(env.data_wp.xfrc_applied)[0,b].detach().cpu().numpy()],
        actuator_force=wp.to_torch(env.data_wp.actuator_force)[0].detach().cpu().numpy().copy())
