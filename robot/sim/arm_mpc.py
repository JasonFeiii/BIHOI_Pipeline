"""Sample wrist/finger targets, project wrist targets through arm IK, then simulate arms."""
from pathlib import Path
import argparse
import json
import shutil
import time
import traceback
import signal
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import mujoco_warp as mjwarp
import warp as wp
from scipy.spatial.transform import Rotation,Slerp
from build_arm_scene import ROOT,OLD,TASK,SIDES,build_scene
from projector import WristProjector
from arm_servo import ArmTrajectoryServo
from smooth_targets import smooth_targets
from contact_guard import ContactGuide
from compute_progress import ComputeProgress
from angular_budget import AngularBudget,AngularBudgetTorch
from drop_guard import allowed as drop_allowed, bottom_height, bottom_height_tensor, min_safe_bottom, preview_drop_cost, speed_allowed, speed_allowed_tensor, speed_preview_cost
from retargeting.config import Config,process_config
from retargeting.utils import mjwp as physics
from retargeting.utils import viser_viewer as live
from retargeting.utils.sampling import make_rollout_fn,make_optimize_once_fn
from retargeting.pipeline.optimize_contact_init import (PhaseSchedule,native_state_failure,
    append_continuous_controls,capture_dynamic_state,restore_dynamic_state)

def dump(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    temp.replace(path)

def save_npz(path,**arrays):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(path)

def prepare(args):
    if args.from_open_hand:
        from open_hand_pipeline import prepare_open
        return prepare_open(args)
    plan=json.loads((ROOT/args.plan_result).read_text())
    run=ROOT/'physics'/args.name;run.mkdir(parents=True,exist_ok=True)
    scene_dir=run/'model'/'sharpa'/'bimanual'/TASK/'0';scene_dir.mkdir(parents=True,exist_ok=True)
    model,mount=build_scene(scene_dir/'scene.xml',plan['center_xy'],plan['heading_rad'],base_height=plan['height_m'])
    ref=dict(np.load(ROOT/'reference_30hz.npz'))
    planned=np.load(ROOT/plan['trajectory_file'])['qpos']
    if len(planned)!=len(ref['qpos']):raise ValueError('Arm planning must cover the complete clip before physical optimization')
    old=np.load(OLD/'trial_01/trajectory_physics.npz');old_q=old['qpos'][-1];old_u=old['ctrl'][-1]
    old_model=mujoco.MjModel.from_xml_path(str(OLD/'sharpa/bimanual'/TASK/'0/scene.xml'))
    od=mujoco.MjData(old_model);od.qpos[:]=old_q;mujoco.mj_forward(old_model,od)
    projector=WristProjector(model,1/150,acceleration_limit=args.acceleration,
                             jerk_limit=args.arm_jerk if args.smooth_commands else None,
                             tracking_rate=args.arm_tracking_rate)
    qinit=planned[0].copy();qinit[6:28]=old_q[6:28];qinit[34:56]=old_q[34:56];qinit[-14:]=old_q[-14:]
    target_initial_p=[];target_initial_R=[]
    for s,side in enumerate(SIDES):
        bid=old_model.body(side+'_hand_C_MC').id
        p=od.xpos[bid].copy();R=od.xmat[bid].reshape(3,3).copy()
        target_initial_p.append(p);target_initial_R.append(R)
        k=projector.kin[s]
        localp=(torch.tensor(p,device='cuda',dtype=torch.float32)-projector.bp[s])@projector.br[s]
        localR=projector.br[s].T@torch.tensor(R,device='cuda',dtype=torch.float32)
        sol,_,_=k.ik(localp[None],localR[None],torch.tensor(qinit[s*28:s*28+6][None],device='cuda',dtype=torch.float32),iterations=25)
        qinit[s*28:s*28+6]=sol[0].cpu().numpy()
    hold=round(args.hold_seconds*30)
    q30=np.r_[np.repeat(qinit[None],hold,axis=0),planned]
    p30=np.r_[np.repeat(np.array(target_initial_p)[None],hold,axis=0),ref['wrist_pos']]
    R30=np.r_[np.repeat(np.array(target_initial_R)[None],hold,axis=0),ref['wrist_rot']]
    original_p30=p30.copy();original_R30=R30.copy()
    decay=np.exp(-np.arange(len(planned))/30/3)
    grasp_delta_p=np.array(target_initial_p)-ref['wrist_pos'][0]
    grasp_delta_R=Rotation.from_matrix(np.array(target_initial_R)@ref['wrist_rot'][0].transpose(0,2,1)).as_rotvec()
    for s in range(2):
        p30[hold:,s]+=decay[:,None]*grasp_delta_p[s]
        R30[hold:,s]=Rotation.from_rotvec(decay[:,None]*grasp_delta_R[s]).as_matrix()@R30[hold:,s]
    q30[hold:,6:28]=ref['qpos'][:,6:28];q30[hold:,34:56]=ref['qpos'][:,34:56]
    q30[hold:,-14:]=ref['qpos'][:,-14:];q30[:,-7:]=[0,0,-2,1,0,0,0]
    latent30=q30[:,:56].copy()
    gravity_data=mujoco.MjData(model)
    qgravity=q30.copy()
    for i,row in enumerate(q30):
        gravity_data.qpos[:]=row;gravity_data.qvel[:]=0;mujoco.mj_forward(model,gravity_data)
        for s in range(2):
            ids=np.arange(s*28,s*28+6)
            if not args.trajectory_servo:
                qgravity[i,ids]+=gravity_data.qfrc_bias[ids]/model.actuator_gainprm[ids,0]
    pp,rr=projector.wrists(torch.tensor(qgravity,device='cuda',dtype=torch.float32))
    pp=pp.cpu().numpy().astype(float);rr=rr.cpu().numpy().astype(float)
    for s in range(2):
        pp[hold:,s]+=decay[:,None]*grasp_delta_p[s]
        rr[hold:,s]=Rotation.from_rotvec(decay[:,None]*grasp_delta_R[s]).as_matrix()@rr[hold:,s]
        latent30[:,s*28:s*28+3]=pp[:,s]
        latent30[:,s*28+3:s*28+6]=Rotation.from_matrix(rr[:,s]).as_euler('XYZ')
    # Euler coordinates are periodic. Interpolating +pi to -pi would create
    # a fictitious full turn although the physical wrist moved only slightly.
    for s in range(2):latent30[:,s*28+3:s*28+6]=np.unwrap(latent30[:,s*28+3:s*28+6],axis=0)
    for s in range(2):
        f=s*28+6
        latent30[:hold,f:f+22]=old_u[f:f+22]
        decay=np.exp(-np.arange(len(planned))/30/3)[:,None]
        grip_bias=old_u[f:f+22]-old_q[f:f+22]
        latent30[hold:,f:f+22]=ref['qpos'][:,f:f+22]+decay*(old_q[f:f+22]-ref['qpos'][0,f:f+22])+grip_bias
    warm30=latent30.copy()
    if args.accepted_finger_seed or args.accepted_hand_seed or args.contact_prior:
        accepted=np.load(OLD/'trial_accepted_v2/trajectory_physics.npz')
        mapping=np.asarray(json.loads((OLD/'trial_accepted_v2/schedule.json').read_text())['source_mapping'])
        indices=np.searchsorted(mapping,ref['source_frame'],side='right')-1
        for s in range(2):
            f=s*28+6;warm30[hold:,f:f+22]=accepted['ctrl'][indices,f:f+22]
        dump(run/'finger_seed_provenance.json',{'source':str(OLD/'trial_accepted_v2/trajectory_physics.npz'),'source_frame_mapping':'last occurrence of each source frame after the initial hold','copied_fields':'finger control seeds only; all resulting physical states are re-simulated'})
        if args.accepted_hand_seed or args.contact_prior:
            accepted_p=np.empty_like(p30[hold:]);accepted_R=np.empty_like(R30[hold:])
            for i,index in enumerate(indices):
                od.qpos[:]=accepted['qpos'][index];mujoco.mj_kinematics(old_model,od)
                for s,side in enumerate(SIDES):
                    b=old_model.body(side+'_hand_C_MC').id
                    accepted_p[i,s]=od.xpos[b];accepted_R[i,s]=od.xmat[b].reshape(3,3)
            for s in range(2):
                gravity_delta_p=pp[hold:,s]-p30[hold:,s]
                gravity_delta_R=rr[hold:,s]@R30[hold:,s].transpose(0,2,1)
                warm30[hold:,s*28:s*28+3]=accepted_p[:,s]+gravity_delta_p
                warm30[hold:,s*28+3:s*28+6]=Rotation.from_matrix(gravity_delta_R@accepted_R[:,s]).as_euler('XYZ')
            for s in range(2):warm30[:,s*28+3:s*28+6]=np.unwrap(warm30[:,s*28+3:s*28+6],axis=0)
            dump(run/'hand_seed_provenance.json',{'source':str(OLD/'trial_accepted_v2/trajectory_physics.npz'),'used_as':('extra control candidate only; always reprojected and resimulated' if args.contact_prior else 'initial MPC window control initialization only; never a tracking reference or appended nominal trajectory'),'all_wrist_targets_pass_through_arm_ik':True,'all_physical_states_re_simulated':True,'original_wrist_reference_preserved_for_audit':True})
    times30=np.arange(len(q30))/30;times=np.arange((len(q30)-1)*5+1)/150
    qr=np.column_stack([np.interp(times,times30,q30[:,j]) for j in range(model.nq)])
    cp=np.column_stack([np.interp(times,times30,latent30[:,j]) for j in range(56)])
    cw=np.column_stack([np.interp(times,times30,warm30[:,j]) for j in range(56)])
    pr=np.empty((len(times),2,3));Rr=np.empty((len(times),2,3,3));po=pr.copy();Ro=Rr.copy()
    for s in range(2):
        pr[:,s]=np.column_stack([np.interp(times,times30,p30[:,s,j]) for j in range(3)])
        Rr[:,s]=Slerp(times30,Rotation.from_matrix(R30[:,s]))(times).as_matrix()
        po[:,s]=np.column_stack([np.interp(times,times30,original_p30[:,s,j]) for j in range(3)])
        Ro[:,s]=Slerp(times30,Rotation.from_matrix(original_R30[:,s]))(times).as_matrix()
    for offset in [-14,-7]:
        sl=slice(offset+3,offset+7 if offset+7 else None)
        xyzw=q30[:,sl][:,[1,2,3,0]]
        qr[:,sl]=Slerp(times30,Rotation.from_quat(xyzw))(times).as_quat(scalar_first=True)
    qv=np.zeros((len(qr),model.nv))
    for i in range(len(qr)-1):mujoco.mj_differentiatePos(model,qv[i],1/150,qr[i],qr[i+1])
    np.savez_compressed(scene_dir/'trajectory_kinematic.npz',qpos=qr[::5],qvel=qv[::5],frequency=30.)
    np.savez_compressed(run/'reference.npz',qpos=qr,qvel=qv,latent=cp,warm_latent=cw,wrist_pos=po,wrist_rot=Ro,control_wrist_pos=pr,control_wrist_rot=Rr,time=times)
    dump(run/'grasp_blend.json',{'translation_offset_m':grasp_delta_p.tolist(),'rotation_offset_rotvec':grasp_delta_R.tolist(),'exponential_decay_seconds':3.,'original_wrist_reference_preserved_for_audit':True})
    dump(run/'placement.json',plan);dump(run/'mount.json',mount)
    # Warm-start controls may come from a verified grasp, but the objective
    # always uses the original, unoptimized retargeted hand/object reference.
    dump(run/'reference_policy.json',{'objective_hand_reference':'unoptimized retargeted Sharpa','objective_object_reference':'unoptimized original object poses','previous_physics_usage':('initial grasp plus optional resimulated control candidate; never the objective' if args.contact_prior else 'initial grasp and optional control warm-start only'),'sampling_space':'floating-hand wrist SE(3) plus finger joints','execution_space':'UR3e arm joints plus finger joints; wrist from arm FK'})
    return run,model,projector,qr,qv,cp,po,Ro,qinit,cw

def run(args):
    progress=ComputeProgress(ROOT/'compute_progress.json',args.name,candidate_count=args.samples,iteration_total=args.iterations,horizon_s=args.horizon,commit_s=args.ctrl_dt,search_scale=1.,resident_retry=0)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    wp.set_device('cuda:0');torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream('cuda:0')))
    out,model,projector,qref,vref,latent_ref,wrist_p,wrist_R,qinit,warm_ref=prepare(args)
    from numerical_solver import install as install_numerical_solver, residual as numerical_residual
    from cached_broadphase import install as install_broadphase_cache
    install_broadphase_cache()
    numerical_policy=install_numerical_solver(model)
    dump(out/'numerical_solver_policy.json',numerical_policy)
    shutil.copy2(__file__,out/'pipeline_source.py')
    config=Config(output_root_dir=str(out/'model'),dataset_name='bihoi',task=TASK,robot_type='sharpa',
        embodiment_type='bimanual',sim_dt=1/150,ref_dt=1/30,ctrl_dt=args.ctrl_dt,horizon=args.horizon,knot_dt=.2,
        num_samples=args.samples,max_num_iterations=args.iterations,temperature=.5,
        pos_noise_scale=.01,rot_noise_scale=.03,joint_noise_scale=.10,
        first_ctrl_noise_scale=.5,last_ctrl_noise_scale=2,final_noise_scale=.05,
        zero_first_knot_noise=False,
        use_torch_compile=False,nconmax_per_env=700,njmax_per_env=2500,
        num_perturb_samples=1,warmup_duration=0,show_viewer=False,save_video=False,save_info=True,force=True)
    config=process_config(config);config.output_dir=str(out)
    limit=min(len(qref)-1,round(args.max_seconds*150)) if args.max_seconds is not None else len(qref)-1
    checkpoint_stride=max(1,round(args.checkpoint_interval/config.sim_dt))
    config.max_sim_steps=limit
    pad=config.horizon_steps+2
    extend=lambda x:np.r_[x,np.repeat(x[-1:],pad,axis=0)]
    qp=extend(qref);vp=extend(vref);lp=extend(latent_ref);wpref=extend(wrist_p);wRref=extend(wrist_R)
    tensors=[torch.tensor(x,device='cuda',dtype=torch.float32) for x in [qp,vp,qp[:,:56],np.zeros((len(qp),10)),np.zeros((len(qp),10,3))]]
    tlatent=torch.tensor(lp,device='cuda',dtype=torch.float32)
    tprior=torch.tensor(extend(warm_ref),device='cuda',dtype=torch.float32)
    tp=torch.tensor(wpref,device='cuda',dtype=torch.float32);tR=torch.tensor(wRref,device='cuda',dtype=torch.float32)
    if args.headless_worker:
        server=None;status=SimpleNamespace(content='');stop=SimpleNamespace(value=False);follow=SimpleNamespace(value=False)
    else:
        live.init_viser('双 UR3e + Sharpa · IK 后的真实机械臂物理优化',port=args.port)
        server=live._get_server();status=server.gui.add_markdown('准备 GPU 物理模型…')
        stop=server.gui.add_checkbox('停止本次仿真',initial_value=False)
        follow=server.gui.add_checkbox('实时跟随最新物理状态',initial_value=True)
    requested_stop={'value':False}
    signal.signal(signal.SIGTERM,lambda *_:requested_stop.update(value=True))
    if args.headless_worker:
        view_model=model;handles=[]
    else:
        spec=mujoco.MjSpec.from_file(str(config.model_path));view_model=spec.compile()
        for body in spec.bodies:
            for geom in body.geoms:
                if geom.name:
                    gid=view_model.geom(geom.name).id
                    geom.type=mujoco.mjtGeom(int(view_model.geom_type[gid]));geom.group=int(view_model.geom_group[gid])
        handles=live.build_and_log_scene_from_spec(spec,view_model,xml_path=Path(config.model_path),build_ref=False)
    view=mujoco.MjData(view_model);probe=mujoco.MjData(view_model)
    def camera(client):
        client.camera.position=(-.65,-.65,.55);client.camera.look_at=(.15,-.1,.18);client.camera.up_direction=(0,0,1)
    if server is not None:server.on_client_connect(camera)
    env=physics.setup_env(config,tuple(tensors))
    assert not env.model_wp.opt.ls_parallel, 'Validated solver requires iterative line search'
    assert model.ngeom==env.model_cpu.ngeom and all(model.geom(i).name==env.model_cpu.geom(i).name for i in range(model.ngeom)), 'runtime geometry mapping differs from validation model'
    dump(out/'runtime_physics_model.json',dict(cpu_probe_dt=model.opt.timestep,runtime_dt=env.model_cpu.opt.timestep,cpu_integrator=int(model.opt.integrator),runtime_integrator=int(env.model_cpu.opt.integrator),object_mass=float(env.model_cpu.body_mass[env.model_cpu.body('right_object').id]),ngeom=model.ngeom))
    physics.set_qpos(config,env,torch.tensor(qinit,device='cuda',dtype=torch.float32))
    wp.to_torch(env.data_wp.qvel)[:]=0;wp.to_torch(env.data_wp.qacc_warmstart)[:]=0
    if args.from_open_hand:
        from open_hand_pipeline import FullGravity
        scheduler=FullGravity(config,env)
    else:
        scheduler=PhaseSchedule(config,env,qp,{'release_s':-1.,'full_gravity_s':-1.})
    scheduler.apply(0);scheduler.constant_unassisted=True;mjwarp.forward(env.model_wp,env.data_wp)
    objects={g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('right_object').id}
    hands={g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('collision_hand_')}
    arms={g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith(('right_arm_','left_arm_')) and model.geom_contype[g]}
    mask=lambda ids:torch.tensor([g in ids for g in range(model.ngeom)],device='cuda')
    om,hm,am=mask(objects),mask(hands),mask(arms)
    side_masks=[mask({g for g in hands if (model.geom(g).name or '').startswith('collision_hand_'+s+'_')}) for s in SIDES]
    wrist_collision_geoms=[[g for g in arms if (model.body(model.geom_bodyid[g]).name or '')==side+'_arm_wrist_2_link'] for side in SIDES]
    guide=ContactGuide(model)
    predicted_contact={'gaps':None}
    wrench_search=None
    backend_probe={'previous_q':None,'examples':[]}
    shortlist_steps=[]
    vertices=__import__('trimesh').load(OLD/'assets/objects/bowl/source_mesh.ply',process=False).convex_hull.vertices
    vertices_gpu=torch.as_tensor(vertices,device='cuda',dtype=torch.float32)
    drop_context={'start_bottom':None,'reference_start_bottom':None,'reference_bottoms':None,'handover':None}
    from handover_control import HandoverState,observe as observe_handover
    handover_fields=('handover_support','handover_control_valid','handover_support_contact','handover_normal_force','handover_relative_drift','handover_table_contact','handover_table_frames','handover_anchor_local','handover_elapsed_frames','actual_hand_normal_force','actual_hand_force_world','actual_contact_counts','actual_force_available','actual_object_contact_force','actual_object_constraint_force','actual_object_qacc')
    handover_defaults=dict(handover_support=-1,handover_control_valid=False,handover_support_contact=False,handover_normal_force=0.,handover_relative_drift=0.,handover_table_contact=False,handover_table_frames=0,handover_anchor_local=np.zeros(3),handover_elapsed_frames=0,actual_hand_normal_force=np.full(2,np.nan),actual_hand_force_world=np.full((2,3),np.nan),actual_contact_counts=np.zeros(2,int),actual_force_available=False,actual_object_contact_force=np.full(3,np.nan),actual_object_constraint_force=np.full(6,np.nan),actual_object_qacc=np.full(6,np.nan))
    handover=None
    from solver_contacts import SolverContacts
    solver_contacts=SolverContacts(model)
    from collision_anticipation import CollisionAnticipation
    collision_warning=CollisionAnticipation(model)
    finger_indices=torch.tensor(list(range(6,28))+list(range(34,56)),device='cuda')
    collision_probe=mujoco.MjData(model)
    from clearance_guidance import CapsuleClearance,clearance_risk_cost
    wrist_clearance_geometry=CapsuleClearance(model)
    def reward(c,e,ref,sim_step=0):
        if args.contact_guard:
            # A physics step leaves collision data at the pre-integration
            # configuration. Score contacts and anchors at the same qpos as
            # the wrist/object tracking terms, without changing physical state.
            mjwarp.kinematics(e.model_wp,e.data_wp)
            mjwarp.collision(e.model_wp,e.data_wp)
        q=wp.to_torch(e.data_wp.qpos);v=wp.to_torch(e.data_wp.qvel)
        p,R=projector.wrists(q)
        wrist_error=(p-tp[sim_step]).norm(dim=-1).sum(-1)
        cos=((R.transpose(-1,-2)@tR[sim_step]).diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
        wrist_rot=torch.acos(cos.clamp(-1+1e-6,1-1e-6)).sum(-1)
        free_goal_error=None;free_goal_angle=None
        # One common object path and reference-local grasp targets for both hands.
        ti=max(0,min(sim_step-config.warmup_sim_step-1,config.horizon_steps-1))
        gp=wrench_goal_p[ti];gr=wrench_goal_r[ti]
        wrist_error=(p-gp).norm(dim=-1).sum(-1)
        cc=((R.transpose(-1,-2)@gr).diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
        wrist_rot=torch.acos(cc.clamp(-1+1e-6,1-1e-6)).sum(-1)

        # Score directly against the retarget objective; the actual state is only
        # the physical initial condition, never a replacement reference.
        relative_step=sim_step-config.warmup_sim_step
        wrench_search.capture(q,v,relative_step)
        reference_paths=drop_context['reference_qpos']
        index=max(0,min(relative_step,reference_paths.shape[1]-1))
        targets=reference_paths[:,index,:]
        p_errors=(q[:,None,-14:-11]-targets[None,:,-14:-11]).norm(dim=-1)
        cosines=(q[:,None,-11:-7]*targets[None,:,-11:-7]).sum(-1).abs().clamp(max=1)
        angles=2*torch.acos(cosines.clamp(max=1-1e-7))
        # All slots contain exactly the same raw reference.
        choices=torch.ones(len(q),device=q.device,dtype=torch.long)  # identical raw reference in all compatibility slots
        pe=p_errors.gather(1,choices[:,None]).squeeze(1)
        angle=angles.gather(1,choices[:,None]).squeeze(1)
        raw_pe=(q[:,-14:-11]-ref[0][-14:-11]).norm(dim=-1)
        raw_cosine=(q[:,-11:-7]*ref[0][-11:-7]).sum(-1).abs().clamp(max=1)
        raw_angle=2*torch.acos(raw_cosine.clamp(max=1-1e-7))
        finger=(q[:,finger_indices]-ref[0][finger_indices]).norm(dim=-1)
        object_velocity=(v[:,-12:-9]-ref[1][-12:-9]).norm(dim=-1)
        con=e.data_wp.contact;dist=wp.to_torch(con.dist).reshape(-1)
        ids=wp.to_torch(con.geom).reshape(-1,2).long().clamp(0,model.ngeom-1)
        worlds=wp.to_torch(con.worldid).reshape(-1).long().clamp(0,e.num_worlds-1)
        valid=torch.arange(len(dist),device='cuda')<wp.to_torch(e.data_wp.nacon).reshape(-1)[0]
        ho=valid&((hm[ids[:,0]]&om[ids[:,1]])|(hm[ids[:,1]]&om[ids[:,0]]))
        ac=valid&(am[ids[:,0]]|am[ids[:,1]])
        pen=torch.zeros(e.num_worlds,device='cuda');apen=torch.zeros_like(pen);touch=torch.zeros_like(pen)
        pen.scatter_reduce_(0,worlds,torch.where(ho,torch.clamp(-dist-.004,min=0.)**2,0.),reduce='amax',include_self=True)
        apen.scatter_reduce_(0,worlds,torch.where(ac,torch.clamp(-dist,min=0.)**2,0.),reduce='amax',include_self=True)
        touch.scatter_reduce_(0,worlds,(ho&(dist<=.0005)).float(),reduce='amax',include_self=True)
        side_touch=torch.zeros((e.num_worlds,2),device='cuda')
        if args.contact_guard:
            for s,sm in enumerate(side_masks):
                belongs=ho&(sm[ids[:,0]]|sm[ids[:,1]])&(dist<=.0008)
                side_touch[:,s].scatter_reduce_(0,worlds,belongs.float(),reduce='amax',include_self=True)
        guard=.5*(torch.clamp(pe/.025-1,min=0)**2+torch.clamp(angle/np.deg2rad(15)-1,min=0)**2)
        holding=False  # no extra stationary grasp test
        contact_required=not args.from_open_hand or sim_step>=round(args.contact_required_after*150)
        if holding:
            score=-15*pe-1.5*angle-2*wrist_error-.15*wrist_rot-.015*finger-30000*pen-100000*apen-.2*(1-touch)-guard
            position_bound=.019;rotation_bound=np.deg2rad(14)
        else:
            score=-3*pe-.3*angle-.3*wrist_error-.04*wrist_rot-.006*finger-30000*pen-100000*apen-.12*(1-touch)-guard
            score-=.5*object_velocity
            position_bound=.035;rotation_bound=np.deg2rad(20)
        clearance_now=wrist_clearance_geometry(wp.to_torch(e.data_wp.geom_xpos),wp.to_torch(e.data_wp.geom_xmat))
        previous=predicted_contact['wrist_previous']
        risk_cost,closing=clearance_risk_cost(clearance_now,previous,config.sim_dt)
        predicted_contact['wrist_previous']=clearance_now.detach()
        predicted_contact['wrist_min']=torch.minimum(predicted_contact['wrist_min'],clearance_now.detach())
        if contact_required:
            score-=risk_cost
            predicted_contact['wrist_cost']+=risk_cost.detach()
        if free_goal_error is not None:
            # Candidate generation and scoring must agree on the new grasp.
            # This soft term provides progress before a distant hand touches.
            score-=1.5*free_goal_error+.1*free_goal_angle
        # Preserve the parent score weights, now all against the raw reference.
        score-=.10*raw_pe+.02*raw_angle
        # Momentum target replaces the old per-step selectable angular target.
        score-=wrench_search.momentum_cost(q,v,index)
        if args.first_pass:position_bound=.10;rotation_bound=np.deg2rad(90)
        if args.contact_guard:
            from regrasp_guidance import no_hand_cost
            # Object-local target plus response-null exploration can change grasp;
            # do not simultaneously pull the hands back to a frozen old anchor.
            score-=no_hand_cost(side_touch,contact_required)
            position_bound=min(position_bound,.075);rotation_bound=min(rotation_bound,np.deg2rad(50))
        if args.strict_physics:
            position_bound=.05 if contact_required else .10
            rotation_bound=np.deg2rad(30 if contact_required else 60)
            force=wp.to_torch(e.data_wp.actuator_force)[:,trajectory_servo.ids]
            limits=actuator_force_limits
            score-=.15*torch.clamp(force.abs()/limits-.85,min=0).square().mean(-1)
        penetration_allowance=.001 if args.strict_physics else (.0035 if args.contact_guard else (.008 if args.first_pass else .0009))
        if args.continue_on_task_failure:
            # Task progress is scored, but a temporarily grounded/dropped
            # object is a valid physical state and may recover.
            position_bound=float('inf');rotation_bound=float('inf')
        arm_allowance=.001 if args.strict_physics else (.003 if args.contact_guard else .0001)
        feasible=((pe<position_bound)&(angle<rotation_bound)&(pen<penetration_allowance**2)&(apen<arm_allowance**2)&((touch>0)|args.contact_guard)).float()
        reasons={}
        if args.contact_guard:
            reasons={'bad_position':(pe>=position_bound).float(),'bad_rotation':(angle>=rotation_bound).float(),
                     'bad_warp_hand':(pen>=penetration_allowance**2).float(),'bad_warp_arm':(apen>=arm_allowance**2).float(),
                     'bad_native_hand':torch.zeros_like(pe),'bad_native_arm':torch.zeros_like(pe),'bad_clearance':torch.zeros_like(pe),'bad_drop_clearance':torch.zeros_like(pe),'bad_object_speed':torch.zeros_like(pe),'bad_speed_linear':torch.zeros_like(pe),'bad_speed_angular':torch.zeros_like(pe),'bad_handover_support':torch.zeros_like(pe)}
        if args.contact_guard:score-=70000*pen
        # The GPU and CPU convex-contact backends can disagree near a bowl rim.
        # Score native collision distances before selection, not only execution.
        relative_step=sim_step-config.warmup_sim_step
        if contact_required and 0<relative_step<=config.horizon_steps and drop_context['start_bottom'] is not None:
            ref_bottom=drop_context['reference_bottoms'][relative_step-1]
            bottom=bottom_height_tensor(q,vertices_gpu)
            score-=args.drop_guidance_weight*preview_drop_cost(bottom,drop_context['start_bottom'],drop_context['reference_start_bottom'],ref_bottom)
            score-=args.speed_guidance_weight*speed_preview_cost(v,ref[1])
            if relative_step<=config.ctrl_steps:
                safe_bottom=min_safe_bottom(drop_context['start_bottom'],drop_context['reference_start_bottom'],ref_bottom)
                bottom_ok=bottom>=safe_bottom
                if drop_context['handover'] is not None:
                    bottom_ok=(bottom>=-.003)&((bottom>.005)|(v[:,-10]>=-.35))
                linear=torch.linalg.vector_norm(v[:,-12:-9],dim=-1)
                angular=torch.linalg.vector_norm(v[:,-9:-6],dim=-1)
                linear_limit=torch.clamp(2*torch.linalg.vector_norm(ref[1][-12:-9])+.25,min=.6)
                angular_limit=torch.clamp(2*torch.linalg.vector_norm(ref[1][-9:-6])+2.,min=4.)
                if 0<relative_step<=config.ctrl_steps:
                    angular_ok=predicted_contact['angle'].step(angular,angular_limit,side_touch.any(dim=1),is_end=(relative_step==config.ctrl_steps))
                    speed_ok=torch.isfinite(linear)&(linear<=linear_limit)&angular_ok
                else:
                    angular_ok=angular<=angular_limit;speed_ok=speed_allowed_tensor(v,ref[1])
                feasible*=bottom_ok.float()*speed_ok.float()
                reasons['bad_drop_clearance']=(~bottom_ok).float()
                reasons['bad_object_speed']=(~speed_ok).float()
                linear=torch.linalg.vector_norm(v[:,-12:-9],dim=-1)
                angular=torch.linalg.vector_norm(v[:,-9:-6],dim=-1)
                linear_limit=torch.clamp(2*torch.linalg.vector_norm(ref[1][-12:-9])+.25,min=.6)
                angular_limit=torch.clamp(2*torch.linalg.vector_norm(ref[1][-9:-6])+2.,min=4.)
                reasons['bad_speed_linear']=(linear>linear_limit).float()
                reasons['bad_speed_angular']=(~angular_ok).float()
        progress.update('rollout',rollout_step=int(relative_step),rollout_total=int(config.horizon_steps))
        native_horizon=max(config.ctrl_steps,round(.6/config.sim_dt)) if args.contact_guard else config.ctrl_steps
        screen_now=relative_step%3==0 or relative_step==native_horizon
        if (args.contact_guard or args.diagnostic_contact_backend) and relative_step<=config.ctrl_steps:screen_now=True
        if args.native_shortlist and 0<relative_step<=config.ctrl_steps:
            shortlist_steps.append((q.detach().clone(),v.detach().clone()))
        if args.native_screen and not args.native_shortlist and relative_step<=native_horizon and screen_now:
            native_pen=[];native_arm=[];native_touch=[];native_clearance=[]
            for row in q.detach().cpu().numpy():
                collision_probe.qpos[:]=row
                mujoco.mj_kinematics(model,collision_probe);mujoco.mj_collision(model,collision_probe)
                hp=0.;ap=0.;nt=[False,False]
                for contact in collision_probe.contact:
                    x,y=map(int,contact.geom)
                    if (x in objects and y in hands) or (x in hands and y in objects):hp=max(hp,-float(contact.dist))
                    if x in arms or y in arms:ap=max(ap,-float(contact.dist))
                    if args.contact_guard and contact.dist<=.0005 and ((x in objects and y in hands) or (y in objects and x in hands)):
                        name=model.geom(y if x in objects else x).name or ''
                        for s,side in enumerate(SIDES):nt[s]|=name.startswith('collision_hand_'+side+'_')
                native_pen.append(hp);native_arm.append(ap)
                native_touch.append(nt)
                clearance=.02
                if args.contact_guard:
                    for x in wrist_collision_geoms[0]:
                        for y in wrist_collision_geoms[1]:
                            clearance=min(clearance,float(mujoco.mj_geomDistance(model,collision_probe,x,y,.02,None)))
                native_clearance.append(clearance)
            hp=torch.tensor(native_pen,device='cuda',dtype=q.dtype);ap=torch.tensor(native_arm,device='cuda',dtype=q.dtype)
            if args.diagnostic_contact_backend and len(backend_probe['examples'])<6:
                raw_depth=pen.sqrt()+.004
                mismatch=torch.nonzero((raw_depth>=.0075)&(hp<.0075),as_tuple=False).flatten()
                if len(mismatch):
                    world=int(mismatch[0]);previous=backend_probe['previous_q']
                    old_depth=None
                    if previous is not None:
                        collision_probe.qpos[:]=previous[world]
                        mujoco.mj_kinematics(model,collision_probe);mujoco.mj_collision(model,collision_probe)
                        old_depth=max([-float(con.dist) for con in collision_probe.contact if (int(con.geom[0]) in objects and int(con.geom[1]) in hands) or (int(con.geom[1]) in objects and int(con.geom[0]) in hands)]+[0.])
                    backend_probe['examples'].append({'step':int(relative_step),'world':world,'warp_mm':float(raw_depth[world])*1000,'native_current_mm':float(hp[world])*1000,'native_previous_mm':None if old_depth is None else old_depth*1000})
            # Retain the small compliant compression needed by the verified
            # grasp. The independent execution guard still rejects >=5 mm.
            score-=30000*torch.clamp(hp-.004,min=0)**2+100000*ap**2
            feasible*=((hp<(.005 if args.strict_physics else (.0075 if args.contact_guard else (.015 if args.first_pass else .0048))))&(ap<(.001 if args.contact_guard else (.005 if args.first_pass else .0008)))).float()
            if args.contact_guard:
                nt=torch.tensor(native_touch,device='cuda',dtype=q.dtype)
                from regrasp_guidance import no_hand_cost
                score-=no_hand_cost(nt,contact_required)
                side_touch=nt
                clearance=torch.tensor(native_clearance,device='cuda',dtype=q.dtype)
                score-=30000*torch.clamp(.01-clearance,min=0)**2
                feasible*=(clearance>.001).float()
                reasons['bad_native_hand']=(hp>=(.005 if args.strict_physics else .0075)).float()
                reasons['bad_native_arm']=(ap>=.001).float()
                reasons['bad_clearance']=(clearance<=.001).float()
        if args.contact_guard:
            predicted_contact['gaps']=torch.where(side_touch>0,0,predicted_contact['gaps']+1)
            if drop_context['handover'] is not None:
                support=drop_context['handover'].support
                if 0<relative_step<=config.ctrl_steps:
                    support_ok=(predicted_contact['gaps']<=3).any(dim=1)  # native shortlist verifies controlled transfer
                    feasible*=support_ok.float()
                    reasons['bad_handover_support']=(~support_ok).float()
                elif relative_step>config.ctrl_steps:
                    score+=.3*side_touch[:,1-support]
            if relative_step<=config.ctrl_steps and contact_required and not args.continue_on_task_failure:
                feasible*=torch.all(predicted_contact['gaps']<=round(args.contact_gap*150),dim=-1).float()
            elif relative_step>config.ctrl_steps:
                # Future risks remain in the score. Hard acceptance protects
                # the committed action, which is followed by a fresh replan.
                feasible=torch.ones_like(feasible)
        solver_ok,solver_f,solver_t=numerical_residual(e.data_wp)
        feasible*=solver_ok.float()
        reasons['bad_solver']=(~solver_ok).float()
        metrics={'infeasible':1-feasible,'object_error':pe,'arm_penetration':apen.sqrt()}
        metrics.update(solver_force_residual=solver_f,solver_torque_residual=solver_t)
        if args.contact_guard:metrics.update(missing_right=1-side_touch[:,0],missing_left=1-side_touch[:,1])
        metrics.update(reasons)
        if args.contact_guard:
            for key,value in reasons.items():metrics['commit_'+key]=value if relative_step<=config.ctrl_steps else torch.zeros_like(value)
        if args.diagnostic_contact_backend:backend_probe['previous_q']=q.detach().cpu().numpy().copy()
        return score,metrics
    trajectory_servo=ArmTrajectoryServo(model,config.sim_dt) if args.trajectory_servo else None
    if trajectory_servo is not None:
        trajectory_servo.reset(wp.to_torch(env.data_wp.qpos)[0],wp.to_torch(env.data_wp.qvel)[0])
    actuator_force_limits=(torch.tensor(model.actuator_forcerange[trajectory_servo.ids.cpu().numpy(),1],device='cuda') if trajectory_servo is not None else None)
    def actuator_control(e,action):
        if trajectory_servo is None:return action
        # Saved-state restoration does not restore derived bias forces. Compute
        # them for the current state before every feedback-control evaluation.
        mjwarp.kinematics(e.model_wp,e.data_wp)
        mjwarp.com_pos(e.model_wp,e.data_wp)
        mjwarp.com_vel(e.model_wp,e.data_wp)
        mjwarp.rne(e.model_wp,e.data_wp)
        return trajectory_servo.control(action,wp.to_torch(e.data_wp.qfrc_bias))
    def controlled_step(c,e,action,perturb=False,sim_step=0):
        return scheduler.step(c,e,actuator_control(e,action),perturb=perturb,sim_step=sim_step)
    base=make_rollout_fn(controlled_step,physics.save_state,physics.load_state,reward,
        lambda c,e,r,sim_step=0:(lambda x:(x[0]*10,x[1]))(reward(c,e,r,sim_step)),
        physics.get_trace,physics.save_env_params,physics.load_env_params)
    latest={};sim=0;transport_candidates=None;transport_origin=None
    last_command_velocity=torch.zeros(model.nu,device='cuda',dtype=torch.float32)
    def project(latent):
        q0=wp.to_torch(env.data_wp.qpos)[0];v0=wp.to_torch(env.data_wp.qvel)[0]
        if trajectory_servo is not None:
            boundary_q,boundary_v=trajectory_servo.boundary(q0,v0)
        elif args.continuous_arm_commands:
            boundary_q=q0.clone();boundary_v=v0.clone()
            for side in range(2):
                sl=slice(side*28,side*28+6)
                boundary_q[sl]=wp.to_torch(env.data_wp.ctrl)[0,sl]
                boundary_v[sl]=last_command_velocity[sl]
        else:boundary_q=q0;boundary_v=v0
        mapped=projector.project(latent,tensors[0][sim:sim+config.horizon_steps],boundary_q,boundary_v,
            iterations=args.ik_iterations,astart=trajectory_servo.a_des if args.smooth_commands else None,ik_start=q0)[0]
        dump(out/'ik_seed_audit.json',dict(time_s=sim/150,sides=projector.last_seed_stats))
        if args.smooth_commands:
            mapped[...,finger_indices]=smooth_targets(mapped[...,finger_indices],
                wp.to_torch(env.data_wp.ctrl)[0,finger_indices],last_command_velocity[finger_indices],
                config.sim_dt,args.finger_speed,args.finger_acceleration,tracking_rate=40.)
        if trajectory_servo is not None:return trajectory_servo.pack(mapped)
        if args.velocity_feedforward:
            # UR assets use tau = kp*(ctrl-q) - kv*qdot. Adding kv/kp*qdot_des
            # gives a tracking PD controller instead of a heavily lagged setpoint.
            for side in range(2):
                sl=slice(side*28,side*28+6)
                trajectory=mapped[...,sl].clone()
                velocity=torch.diff(trajectory,dim=-2,prepend=trajectory[...,:1,:])/config.sim_dt
                velocity[...,0,:]=v0[sl]
                ratio=torch.tensor(-model.actuator_biasprm[sl,2]/model.actuator_gainprm[sl,0],device='cuda',dtype=mapped.dtype)
                mapped[...,sl]+=ratio*velocity
        return mapped
    def native_shortlist_mask(scores,gpu_valid):
        # CPU collision verification is required for every step of every
        # candidate eligible for execution, not every discarded GPU sample.
        mask=torch.zeros_like(gpu_valid)
        ids=torch.argsort(torch.where(gpu_valid,scores,-torch.inf),descending=True)
        ids=ids[gpu_valid[ids]][:args.native_shortlist]
        if transport_candidates is not None or latest.get('regrasp_candidate_side') is not None:
            special=[i for i in range(4,min(8,len(gpu_valid))) if bool(gpu_valid[i])]
            order=list(dict.fromkeys(special+ids.cpu().tolist()))[:args.native_shortlist]
            ids=torch.tensor(order,device='cuda',dtype=torch.long)
        latest['native_shortlist_ids']=ids.cpu().tolist()
        if not len(ids):return mask,0
        states=torch.stack([row[0] for row in shortlist_steps],dim=0)[:,ids].cpu().numpy()
        speeds=torch.stack([row[1] for row in shortlist_steps],dim=0)[:,ids].cpu().numpy()
        for column,index in enumerate(ids.tolist()):
            gaps=list(side_gaps);good=True;why=None;angular_screen=AngularBudget(config.sim_dt)
            check_handover=handover.copy() if handover is not None else None
            for k,q in enumerate(states[:,column]):
                collision_probe.qpos[:]=q
                mujoco.mj_kinematics(model,collision_probe);mujoco.mj_collision(model,collision_probe)
                touch=[False,False];hp=ap=0.
                for con in collision_probe.contact:
                    x,y=map(int,con.geom)
                    if x in arms or y in arms:ap=max(ap,-float(con.dist))
                    if (x in objects and y in hands) or (y in objects and x in hands):
                        hp=max(hp,-float(con.dist))
                        if con.dist<=.0005:
                            name=model.geom(y if x in objects else x).name or ''
                            for j,side in enumerate(SIDES):touch[j]|=name.startswith('collision_hand_'+side+'_')
                if hp>=(.005 if args.strict_physics else .0075) or ap>=.001:
                    good=False;why=f'penetration hand={hp} arm={ap}';break
                from wrist_clearance import allowed as wrist_clearance_allowed
                if not wrist_clearance_allowed(model,collision_probe,wrist_collision_geoms):
                    good=False;why='cross wrist clearance <=1mm';break
                gaps=[0 if hit else gaps[j]+1 for j,hit in enumerate(touch)]
                required=not args.from_open_hand or (sim+k+1)/150>=args.contact_required_after
                if required and drop_context['start_bottom'] is not None:
                    if check_handover is None:
                        ok,why=drop_allowed(q,qp[sim+k+1],vertices,drop_context['start_bottom'],drop_context['reference_start_bottom'])
                    else:
                        ob=observe_handover(model,collision_probe,q,objects,hands,check_handover.support,measure_force=False)
                        ok,why=check_handover.observe_step(ob['hits'],ob['normal'],ob['relative_both'],bottom_height(q,vertices),speeds[k,column][-10],ob['table'],check_force=False)
                    speed_ok,speed_why=angular_screen.velocity_step(speeds[k,column],vp[sim+k+1],bool(any(touch)),is_end=(k==len(states)-1))
                    if not (ok and speed_ok):good=False;why=why or speed_why;break
                if required and not args.continue_on_task_failure and max(gaps)>round(args.contact_gap*150):good=False;break
            mask[index]=good
            with (out/'shortlist_audit.jsonl').open('a') as f:f.write(json.dumps(dict(time_s=sim/150,candidate=index,ok=bool(good),step=k+1,reason=why))+'\n')
        return mask,int(len(ids))

    def rollout(c,e,latent,refs,params):
        shortlist_steps.clear()
        # Keep the current and nominal trajectories among the sampled choices.
        # This prevents random exploration from discarding a stable grasp.
        latent[0]=controls
        if len(latent)>1:latent[1]=tlatent[sim:sim+config.horizon_steps]
        if args.contact_prior and len(latent)>3:latent[3]=tprior[sim:sim+config.horizon_steps]
        if args.contact_guard and len(latent)>2:
            # Include a candidate that keeps current grip while moving the wrist.
            latent[2]=controls
            latent[2][:,finger_indices]=wp.to_torch(env.data_wp.ctrl)[0,finger_indices]
        nominal=tlatent[sim:sim+config.horizon_steps]
        if transport_candidates is not None and len(latent)>=8:
            latent[4:8]=transport_candidates
        if regrasp_origin is not None:
            from regrasp_candidates import add_regrasp_candidates
            free=add_regrasp_candidates(latent,regrasp_origin,wp.to_torch(env.data_wp.ctrl)[0],side_gaps,config.sim_dt,hold_support=(handover is not None or transport_origin is None),elapsed_s=(handover.role_elapsed_frames/150 if handover is not None else 0.),goals=regrasp_goals,free_side=(1-handover.support if handover is not None else None),support_goals=support_turn_goals,moving_goals=moving_regrasp_goals)
            if free is not None:
                latest['regrasp_candidate_side']=SIDES[free]
        latent=wrench_search.propose(latent)
        for side in range(2):
            j=side*28
            radius=args.wrist_radius
            lo=nominal[:,j:j+3]-radius;hi=nominal[:,j:j+3]+radius
            alo=nominal[:,j+3:j+6]-.10;ahi=nominal[:,j+3:j+6]+.10
            if transport_origin is not None:
                # Retain the physically attained grasp as an admissible
                # starting point while tracking the unchanged raw reference.
                lo=torch.minimum(lo,transport_origin[j:j+3]-.005)
                hi=torch.maximum(hi,transport_origin[j:j+3]+.005)
                alo=torch.minimum(alo,transport_origin[j+3:j+6]-.025)
                ahi=torch.maximum(ahi,transport_origin[j+3:j+6]+.025)
            if regrasp_origin is not None and latest.get('regrasp_candidate_side') is not None:
                rp=latent[4:8,:,j:j+3].clone();rr=latent[4:8,:,j+3:j+6].clone()
            latent[...,j:j+3]=latent[...,j:j+3].clamp(lo[None],hi[None])
            latent[...,j+3:j+6]=latent[...,j+3:j+6].clamp(alo[None],ahi[None])
            if regrasp_origin is not None and latest.get('regrasp_candidate_side') is not None:
                latent[4:8,:,j:j+3]=rp.clamp(torch.minimum(lo,regrasp_origin[j:j+3]-.05)[None],torch.maximum(hi,regrasp_origin[j:j+3]+.05)[None])
                latent[4:8,:,j+3:j+6]=rr.clamp(torch.minimum(alo,regrasp_origin[j+3:j+6]-.025)[None],torch.maximum(ahi,regrasp_origin[j+3:j+6]+.025)[None])
        if latest.get('regrasp_candidate_side') is not None:
            # Keep the displayed hand reference_paths identical to the actual bounded
            # free-hand candidates that will be projected through IK.
            save_npz(out/'candidate_targets.npz',
                time=sim/150+np.arange(config.horizon_steps+1)*config.sim_dt,
                qpos=reference_paths,latent=latent[4:7].detach().cpu().numpy())
        mapped=project(latent)
        if args.diagnostic_contact_backend:
            backend_probe['previous_q']=wp.to_torch(e.data_wp.qpos).detach().cpu().numpy().copy()
            backend_probe['examples']=[]
        predicted_contact['angle']=AngularBudgetTorch(e.num_worlds,'cuda',config.sim_dt)
        predicted_contact['wrist_previous']=torch.full((e.num_worlds,),drop_context['clearance_start'],device='cuda')
        predicted_contact['wrist_cost']=torch.zeros(e.num_worlds,device='cuda')
        predicted_contact['wrist_min']=torch.full((e.num_worlds,),float('inf'),device='cuda')
        if args.contact_guard:predicted_contact['gaps']=torch.tensor(side_gaps,device='cuda',dtype=torch.float32)[None].expand(e.num_worlds,-1).clone()
        _,score,metrics=base(c,e,mapped,refs,params)
        if args.diagnostic_contact_backend:dump(out/'backend_probe.json',backend_probe['examples'])
        valid=(metrics['infeasible']<1e-6)&torch.isfinite(score)
        latest['grasp_transport_gpu_feasible']=(valid[4:8].cpu().tolist() if transport_candidates is not None else None)
        if args.native_shortlist:
            native_valid,checked=native_shortlist_mask(score,valid)
            valid&=native_valid
            # Do not steer the next iteration toward an unverified candidate
            # when at least one verified candidate is available.
            if bool(valid.any()):score=torch.where(valid,score,score-1000.)
            latest['native_shortlist_checked']=checked
        idx=int(torch.argmax(torch.where(torch.isfinite(score),score,-torch.inf)))
        latest['best']=latent[idx].clone()
        dump(out/'clearance_guidance_metrics.json',dict(time_s=sim/150,candidate_count=len(latent),
            horizon_s=args.horizon,hard_filter_added=False,preview_s=.15,margin_m=.01,
            min_clearance_m=predicted_contact['wrist_min'].cpu().tolist(),
            mean_soft_cost=(predicted_contact['wrist_cost']/config.horizon_steps).cpu().tolist(),
            best_scored_candidate=idx,native_feasible_count=int(valid.sum())))
        wrench_search.observe(score,metrics,valid)
        latest['feasible_count']=int(valid.sum())
        if not bool(valid.any()):
            latest['diagnostic_score']=score.detach().cpu().numpy().copy()
            latest['diagnostic_metrics']={k:v.detach().cpu().numpy().copy() for k,v in metrics.items() if k.startswith('commit_bad_')}
        latest['grasp_transport_feasible']=(valid[4:8].cpu().tolist() if transport_candidates is not None else None)
        latest['best_candidate_index']=idx
        if args.contact_prior:latest['prior_feasible']=bool(valid[3])
        if args.contact_guard:
            latest['constraint_diagnostics']={key:{'min_fraction':float(value.min()),'zero_count':int((value<1e-6).sum()),'best_score_fraction':float(value[idx])} for key,value in metrics.items() if key.startswith(('bad_','missing_','commit_'))}
        if latest.get('regrasp_candidate_side') is not None:
            with (out/'handover_candidates.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time_s=sim/150,elapsed_s=None if handover is None else handover.elapsed_frames/150,scores=score[4:8].cpu().tolist(),valid=valid[4:8].cpu().tolist(),diagnostics={key:value[4:8].cpu().tolist() for key,value in metrics.items() if key.startswith('commit_bad_')}))+'\n')
        if bool(valid.any()):
            selected=int(torch.argmax(torch.where(valid,score,-torch.inf)))
            latest['feasible']=latent[selected].clone()
            latest['selected_candidate_index']=selected
        return latent,score,metrics
    once=make_optimize_once_fn(rollout);config.env_params_list=[[{}] for _ in range(args.iterations)]
    controls=tlatent[:config.horizon_steps].clone()
    records=[];native=[];physical_records={};task_warnings=[];grasp_verified_time=None;failure=None;started=time.monotonic();hold_passed=False;native_rejects=0;contact_gap_steps=0;observed_failures=[];side_gaps=[0,0]
    if args.known_failure_time is not None:observed_failures.append({'time_s':args.known_failure_time,'reason':'sustained loss of grasp observed in the preceding portion of this trial'})
    def record(i,q,v,u,latent,publish=True):
        view.qpos[:]=q;mujoco.mj_forward(view_model,view)
        if publish and not args.headless_worker:live.log_frame(view,i/150,handles,record=True,playback_fps=30)
        if publish and not args.headless_worker and follow.value:
            live._STATE.playback_checkbox.value=False
            live._STATE.playback_slider.value=len(live._STATE.frame_history)-1
            live.render_current_frame(handles)
        p=projector.wrists(torch.tensor(q,device='cuda',dtype=torch.float32))[0].cpu().numpy()
        qa=Rotation.from_quat(q[[-10,-9,-8,-11]]);qb=Rotation.from_quat(qp[i,[-10,-9,-8,-11]])
        contacts=0;side_contacts=[0,0]
        for con in view.contact:
            a,b=map(int,con.geom)
            if ((a in objects and b in hands) or (b in objects and a in hands)) and con.dist<=.0005:
                contacts+=1
                name=model.geom(b if a in objects else a).name or ''
                for s,side in enumerate(SIDES):side_contacts[s]+=int(name.startswith('collision_hand_'+side+'_'))
        phase=('open_hand_approach' if args.from_open_hand and i/150<args.open_hold_start
            else 'full_gravity_grasp_hold' if i/150<args.hold_seconds else 'arm_hand_manipulation')
        row={'phase':phase,'time_s':i/150,'object_error_m':float(np.linalg.norm(q[-14:-11]-qp[i,-14:-11])),
             'wrist_error_m':np.linalg.norm(p-wpref[i],axis=-1).tolist(),'object_weld':False,'gravity':9.81,'strict_full_gravity_policy':args.strict_physics,
             'object_rotation_error_deg':float(np.degrees((qa.inv()*qb).magnitude())),
             'object_bottom_m':float((vertices@qa.as_matrix().T+q[-14:-11])[:,2].min()),
             'hand_object_contacts':contacts,'side_contacts':side_contacts}
        records.append((q.copy(),v.copy(),u.copy(),latent.copy(),i/150,row))
        if publish and not args.no_publish_live:
            save_npz(ROOT/'live_state.npz',qpos=q,time=np.array(i/150),run=np.array(args.name))
            dump(ROOT/'live_status.json',dict(run=args.name,state='running',**row))
        if publish:status.content=f"t={i/150:.2f}s · {'无辅助抓取保持' if i<args.hold_seconds*150 else 'IK 后关节驱动物理动作'}\n\n物体误差 {row['object_error_m']*1000:.1f} mm · 腕部误差 {np.array(row['wrist_error_m'])*1000} mm"
    def save_trajectories():
        if native:
            extra={}
            if args.strict_physics:
                from open_hand_pipeline import telemetry
                keys=['gravity','object_weld_active','object_external_force','actuator_force']
                if any(round(x[4]*150) not in physical_records for x in native):
                    raise RuntimeError('Strict run cannot certify a prefix without physical telemetry')
                extra={key:np.array([physical_records[round(x[4]*150)][key] for x in native]) for key in keys}
                extra.update({key:np.array([physical_records[round(x[4]*150)].get(key,handover_defaults[key]) for x in native]) for key in handover_fields})
            save_npz(out/'native_trajectory.npz',**extra,qpos=np.array([x[0] for x in native]),qvel=np.array([x[1] for x in native]),ctrl=np.array([x[2] for x in native]),latent=np.array([x[3] for x in native]),time=np.array([x[4] for x in native]),arm_q_des=np.array([x[5] for x in native]),arm_v_des=np.array([x[6] for x in native]))
        save_npz(out/'trajectory.npz',qpos=np.array([x[0] for x in records]),qvel=np.array([x[1] for x in records]),ctrl=np.array([x[2] for x in records]),latent=np.array([x[3] for x in records]),time=np.array([x[4] for x in records]),side_contacts=np.array([x[5]['side_contacts'] for x in records]),object_error=np.array([x[5]['object_error_m'] for x in records]),object_rotation_error=np.array([x[5]['object_rotation_error_deg'] for x in records]))
    if args.resume_from:
        previous=ROOT/'physics'/args.resume_from
        checkpoint=np.load(previous/'checkpoint.npz')
        for key in checkpoint.files:
            if not key.startswith('state_'):continue
            dst=wp.to_torch(getattr(env.data_wp,key[6:]))
            dst.copy_(torch.as_tensor(checkpoint[key],device='cuda',dtype=dst.dtype).expand_as(dst))
        sim=int(checkpoint['sim']);hold_passed=bool(checkpoint['hold_passed'])
        controls=torch.tensor(checkpoint['controls'],device='cuda',dtype=torch.float32)
        # A recovery branch may use a longer preview than its parent. Retain
        # the optimized prefix and append a continuous raw-reference tail.
        if len(controls)<config.horizon_steps:
            controls=append_continuous_controls(controls,tlatent[sim+len(controls):sim+config.horizon_steps])
        else:
            controls=controls[:config.horizon_steps]
        hist=np.load(previous/'native_trajectory.npz')
        qd=hist['arm_q_des'] if 'arm_q_des' in hist.files else np.full((len(hist['time']),12),np.nan)
        vd=hist['arm_v_des'] if 'arm_v_des' in hist.files else np.full((len(hist['time']),12),np.nan)
        native=[(q,v,u,l,float(t),d,dv) for q,v,u,l,t,d,dv in zip(hist['qpos'],hist['qvel'],hist['ctrl'],hist['latent'],hist['time'],qd,vd) if t<=sim/150+1e-7]
        if args.strict_physics:
            fields=['gravity','object_weld_active','object_external_force','actuator_force']
            if any(key not in hist.files for key in fields):
                raise ValueError('Strict run only resumes prefixes recorded under full-gravity policy')
            for i,t in enumerate(hist['time']):
                if t<=sim/150+1e-7:
                    physical_records[round(float(t)*150)]={key:hist[key][i] for key in fields}
                    physical_records[round(float(t)*150)].update({key:hist[key][i] for key in handover_fields if key in hist.files})
        if 'handover_support_state' in checkpoint.files and int(checkpoint['handover_support_state'])>=0:
            handover=HandoverState(int(checkpoint['handover_support_state']),checkpoint['handover_anchor'].copy(),int(checkpoint['handover_contact_gap']),int(checkpoint['handover_weak_gap']),int(checkpoint['handover_table_frames_state']),int(checkpoint['handover_elapsed']),int(checkpoint['handover_other_frames']),bool(checkpoint['handover_released']) if 'handover_released' in checkpoint.files else True,int(checkpoint['handover_release_gap']) if 'handover_release_gap' in checkpoint.files else 0)
        if handover is not None:
            if 'handover_recent_local' in checkpoint.files:
                for side in range(2):
                    handover.recent_local[side]=[x.copy() for x in checkpoint['handover_recent_local'][side] if np.isfinite(x).all()]
            handover.transfers=int(checkpoint['handover_transfers']) if 'handover_transfers' in checkpoint.files else 0
            handover.role_elapsed_frames=int(checkpoint['handover_role_elapsed']) if 'handover_role_elapsed' in checkpoint.files else handover.elapsed_frames
        if len(native)>=2:
            last_command_velocity.copy_(torch.tensor((native[-1][2]-native[-2][2])/config.sim_dt,device='cuda',dtype=torch.float32))
        hist=np.load(previous/'trajectory.npz')
        for q,v,u,l,t in zip(hist['qpos'],hist['qvel'],hist['ctrl'],hist['latent'],hist['time']):
            if t<=sim/150+1e-7:record(round(t*150),q,v,u,l,publish=False)
        scheduler.apply(sim);mjwarp.forward(env.model_wp,env.data_wp)
        if trajectory_servo is not None:
            trajectory_servo.reset(wp.to_torch(env.data_wp.qpos)[0],wp.to_torch(env.data_wp.qvel)[0])
            if 'servo_q_des' in checkpoint.files:
                trajectory_servo.q_des=torch.tensor(checkpoint['servo_q_des'],device='cuda')
                trajectory_servo.v_des=torch.tensor(checkpoint['servo_v_des'],device='cuda')
                if 'servo_a_des' in checkpoint.files:
                    trajectory_servo.a_des=torch.tensor(checkpoint['servo_a_des'],device='cuda')
                elif len(vd)>1 and np.isfinite(vd[-2:]).all():
                    trajectory_servo.a_des=torch.tensor((vd[-1]-vd[-2])/config.sim_dt,device='cuda',dtype=torch.float32).clamp(-args.acceleration,args.acceleration)
        if 'last_command_velocity' in checkpoint.files:
            last_command_velocity.copy_(torch.tensor(checkpoint['last_command_velocity'],device='cuda'))
        # Resume the contact-gap counter as well as the mechanical state.
        for q in reversed([x[0] for x in native[-31:]]):
            probe.qpos[:]=q;mujoco.mj_forward(model,probe)
            if any(con.dist<=.0005 and ((int(con.geom[0]) in objects and int(con.geom[1]) in hands) or (int(con.geom[1]) in objects and int(con.geom[0]) in hands)) for con in probe.contact):break
            contact_gap_steps+=1
        for s,side in enumerate(SIDES):
            for q in reversed([x[0] for x in native[-31:]]):
                probe.qpos[:]=q;mujoco.mj_forward(model,probe)
                if any(con.dist<=.0005 and ((int(con.geom[0]) in objects and (model.geom(int(con.geom[1])).name or '').startswith('collision_hand_'+side+'_')) or (int(con.geom[1]) in objects and (model.geom(int(con.geom[0])).name or '').startswith('collision_hand_'+side+'_'))) for con in probe.contact):break
                side_gaps[s]+=1
        print(f'RESUME {args.resume_from} at {sim/150:.3f}s',flush=True)
    else:
        record(0,qinit,np.zeros(model.nv),qinit[:56],controls[0].cpu().numpy())
        if args.strict_physics:
            from open_hand_pipeline import telemetry
            native.append((qinit.copy(),np.zeros(model.nv),qinit[:56].copy(),controls[0].cpu().numpy(),0.,qinit[np.r_[0:6,28:34]].copy(),np.zeros(12)))
            physical_records[0]=telemetry(env,model)
    from precontact_phase import ContactPhase
    contact_phase=ContactPhase.from_history(physical_records,
        bool(checkpoint.get('precontact_complete',False)) if args.resume_from else False)
    contact_event=False
    dump(out/'precontact_policy.json',dict(candidate_optimization_before_contact=False,
        continuous_physical_execution=True,gravity_ramp_seconds=0.,
        switch='first actual solver hand contact or native hand-object collision within 0.5mm',
        checkpoint_interval_s=args.checkpoint_interval,starts_after_interaction=contact_phase.interacted))
    dump(out/'handover_policy.json',dict(enabled=True,max_cumulative_table_s=.5,max_relative_slip_m=.04,max_touchdown_speed_m_s=.35,max_contact_gap_s=.02,normal_force_estimate_min_N=.02,force_source='actual Warp explicit constraint-row forces checked against generalized force; native reconstruction fallback is labeled',scope='active handover committed steps only',unassisted=True))
    dump(out/'drop_guard_policy.json',dict(max_window_drop_m=.030,min_clearance_m=.010,reference_support_m=.010,scope='committed 0.2s; reference descent accommodated',soft_preview_horizon_s=args.horizon,soft_speed_guidance=True,object_linear_limit='max(0.6 m/s, 2 x reference + 0.25 m/s)',object_angular_limit='max(4 rad/s, 2 x reference + 2 rad/s)',hard_reference_error_limit=False,hard_contact_requirement=False))
    dump(out/'angular_impulse_policy.json',dict(base_limit='max(4 rad/s, 2*reference+2)',peak_margin_rad_s=2,max_high_frames=2,max_excess_angle_deg=.75,hand_control_required_during_excess=True,end_must_be_below_base_limit=True,linear_speed_guard_unchanged=True,scope='executed 0.2s only'))
    if args.accepted_hand_seed or args.accepted_finger_seed:
        # Warm initialization of only this first optimization window. All
        # subsequent appended targets continue to use raw-retarget-derived lp.
        controls=torch.tensor(warm_ref[sim:sim+config.horizon_steps],device='cuda',dtype=torch.float32)
    if args.reset_initial_controls:
        controls=tlatent[sim:sim+config.horizon_steps].clone()
    if args.contact_guard:
        probe.qpos[:]=wp.to_torch(env.data_wp.qpos)[0].cpu().numpy();mujoco.mj_forward(model,probe)
        try:guide.initialize(probe)
        except ValueError:
            found=False
            for row in reversed(native[-23:]):
                probe.qpos[:]=row[0];mujoco.mj_forward(model,probe)
                try:
                    guide.initialize(probe);guide.description['source_time_s']=row[4];found=True;break
                except ValueError:pass
            if not found and not args.from_open_hand:raise ValueError('No bilateral contact geometry within 150 ms of recovery checkpoint')
        if guide.ready:dump(out/'contact_anchors.json',guide.description)
    if not args.no_publish_live:
        save_npz(ROOT/'live_batch.npz',qpos=np.array([records[-1][0]]),time=np.array([sim/150]),run=np.array(args.name))
    def save_checkpoint():
        state=capture_dynamic_state(env)
        # All worlds were synchronized after accepted execution. Save one copy.
        arrays={'state_'+key:value[:1].cpu().numpy() for key,value in state.items()}
        arrays.update(sim=np.array(sim),controls=controls.cpu().numpy(),hold_passed=np.array(bool(hold_passed)))
        arrays['last_command_velocity']=last_command_velocity.cpu().numpy()
        if trajectory_servo is not None:
            arrays['servo_q_des']=trajectory_servo.q_des.cpu().numpy()
            arrays['servo_v_des']=trajectory_servo.v_des.cpu().numpy()
            arrays['servo_a_des']=trajectory_servo.a_des.cpu().numpy()
        arrays['contact_gap_steps']=np.array(contact_gap_steps)
        arrays['precontact_complete']=np.array(contact_phase.interacted)
        arrays['side_contact_gap_steps']=np.array(side_gaps)
        arrays['handover_support_state']=np.array(handover.support if handover is not None else -1)
        if handover is not None:
            arrays.update(handover_anchor=handover.anchor_local.copy(),handover_contact_gap=np.array(handover.contact_gap),handover_weak_gap=np.array(handover.weak_force_gap),handover_table_frames_state=np.array(handover.table_frames),handover_elapsed=np.array(handover.elapsed_frames),handover_other_frames=np.array(handover.other_contact_frames),handover_released=np.array(handover.released),handover_release_gap=np.array(handover.release_gap))
        if handover is not None:
            recent=np.full((2,15,3),np.nan)
            for side in range(2):
                if handover.recent_local[side]:recent[side,-len(handover.recent_local[side]):]=handover.recent_local[side]
            arrays.update(handover_recent_local=recent,handover_transfers=np.array(handover.transfers),handover_role_elapsed=np.array(handover.role_elapsed_frames))
        # Never overwrite a verified checkpoint with an uncommitted state.
        if native and abs(float(native[-1][4])-sim/150)<1e-7:
            expected_q=np.asarray(native[-1][0]).reshape(-1)
            expected_v=np.asarray(native[-1][1]).reshape(-1)
            actual_q=np.asarray(arrays['state_qpos']).reshape(-1)
            actual_v=np.asarray(arrays['state_qvel']).reshape(-1)
            if not np.allclose(actual_q,expected_q,atol=1e-6,rtol=0) or not np.allclose(actual_v,expected_v,atol=1e-6,rtol=0):
                dump(out/'checkpoint_integrity_error.json',dict(time_s=sim/150,reason='uncommitted state differs from accepted trajectory'))
                print('CHECKPOINT_INTEGRITY_ERROR: preserved last verified checkpoint',flush=True)
                return
        save_npz(out/'checkpoint.npz',**arrays)
        if sim%checkpoint_stride==0:save_npz(out/f'checkpoint_{sim:06d}.npz',**arrays)
        save_trajectories()
    save_checkpoint()
    if not args.no_publish_live:dump(ROOT/'live_status.json',dict(run=args.name,state='running',**records[-1][-1]))
    import cProfile,pstats
    profile_output=Path('/tmp')/('bihoi_v56_'+args.name+'_first_window.pstats')
    first_profile=cProfile.Profile();first_profile.enable()
    profile_begin_s=sim/150;profile_wall_start=time.perf_counter();profile_saved=False
    def finish_profile():
        nonlocal profile_saved
        if profile_saved:return
        first_profile.disable();profile_saved=True
        first_profile.dump_stats(str(profile_output))
        stats=pstats.Stats(first_profile)
        rows=[dict(file=f,line=line,function=func,calls=values[1],self_s=values[2],cumulative_s=values[3])
              for (f,line,func),values in stats.stats.items()]
        rows.sort(key=lambda row:row['cumulative_s'],reverse=True)
        dump(out/'first_window_profile.json',dict(start_s=profile_begin_s,end_s=sim/150,
            wall_s=time.perf_counter()-profile_wall_start,profile_file=str(profile_output),
            total_profile_self_s=stats.total_tt,rows=rows))
        print('PROFILE_READY '+str(profile_output),flush=True)
    try:
        while sim<limit and not stop.value and not requested_stop['value']:
            config.warmup_sim_step=sim;latest.clear()
            approach_fast=bool(args.from_open_hand and not contact_phase.interacted)
            contact_event=False
            if approach_fast:
                # Deterministic collision-checked approach: no candidate search,
                # no 1.2s predictive rollout, no learned-response fitting.
                controls=tlatent[sim:sim+config.horizon_steps].clone()
                drop_context['start_bottom']=bottom_height(wp.to_torch(env.data_wp.qpos)[0].cpu().numpy(),vertices)
                drop_context['reference_start_bottom']=bottom_height(qp[sim],vertices)
                progress.update('precontact_approach',accepted_time_s=sim/150,
                    window_start_s=sim/150,window_end_s=sim/150+args.ctrl_dt,
                    iteration=0,iteration_completed=0,rollout_step=0,force=True)
            else:
                if handover is None and ((side_gaps[0]>=3 and side_gaps[1]<=1) or (side_gaps[1]>=3 and side_gaps[0]<=1)):
                    support=0 if side_gaps[0]<=1 else 1
                    probe.qpos[:]=wp.to_torch(env.data_wp.qpos)[0].cpu().numpy()
                    probe.qvel[:]=wp.to_torch(env.data_wp.qvel)[0].cpu().numpy()
                    probe.ctrl[:]=wp.to_torch(env.data_wp.ctrl)[0].cpu().numpy()
                    mujoco.mj_forward(model,probe)
                    ob=observe_handover(model,probe,probe.qpos,objects,hands,support)
                    handover=HandoverState(support,ob['relative'].copy())
                if handover is None and sim/150>=args.contact_required_after and max(side_gaps)<=1:
                    probe.qpos[:]=wp.to_torch(env.data_wp.qpos)[0].cpu().numpy();probe.qvel[:]=wp.to_torch(env.data_wp.qvel)[0].cpu().numpy();probe.ctrl[:]=wp.to_torch(env.data_wp.ctrl)[0].cpu().numpy()
                    mujoco.mj_forward(model,probe)
                    warning=collision_warning.evaluate(probe.qpos,probe.qvel)
                    dump(out/'collision_warning.json',dict(time_s=sim/150,**warning))
                    if warning['risk']:
                        proposed=args.seed%2
                        ob=observe_handover(model,probe,probe.qpos,objects,hands,proposed)
                        if ob['hits'][proposed] and ob['normal'][proposed]>=.02:
                            handover=HandoverState(proposed,ob['relative'].copy(),released=False)
                            dump(out/'handover_trigger.json',dict(time_s=sim/150,support=SIDES[proposed],reason='impending_cross_arm_collision',**warning))
                drop_context['handover']=handover
                drop_context['start_bottom']=bottom_height(wp.to_torch(env.data_wp.qpos)[0].cpu().numpy(),vertices)
                drop_context['reference_start_bottom']=bottom_height(qp[sim],vertices)
                drop_context['reference_bottoms']=[bottom_height(qp[min(sim+k+1,len(qp)-1)],vertices) for k in range(config.horizon_steps)]
                current=wp.to_torch(env.data_wp.qpos)[0].cpu().numpy()
                probe.qpos[:]=current;mujoco.mj_kinematics(model,probe)
                drop_context['clearance_start']=float(wrist_clearance_geometry(
                    torch.tensor(probe.geom_xpos.copy()),torch.tensor(probe.geom_xmat.copy())))
                # Exact retarget objective. Current physics initializes the rollout,
                # but never shifts, rotates, slows or lifts the reference path.
                reference_paths=np.repeat(qp[None,sim:sim+config.horizon_steps+1],3,axis=0).copy()
                drop_context['reference_qpos']=torch.as_tensor(reference_paths,device='cuda',dtype=torch.float32)
                save_npz(out/'reference_targets.npz',time=sim/150+np.arange(config.horizon_steps+1)*config.sim_dt,
                    qpos=reference_paths,latent=np.empty((0,0,56)))
                dump(out/'reference_target_status.json',dict(time_s=sim/150,mode='raw_retarget',
                    exact_object_reference=True,synthetic_target=False,reference_time_scale=1.))
                progress.update('sampling',accepted_time_s=sim/150,window_start_s=sim/150,window_end_s=sim/150+args.horizon,iteration=0,iteration_completed=0,rollout_step=0,force=True)
                transport_candidates=None;transport_origin=None;regrasp_origin=None;regrasp_goals=None;support_turn_goals=None;moving_regrasp_goals=None
                drop_context['paired_wrist_pos']=None;drop_context['paired_wrist_rot']=None
                drop_context['support_turn_pos']=None
                drop_context['support_turn_rot']=None
                drop_context['regrasp_goal_pos']=None;drop_context['regrasp_goal_rot']=None
                if args.grasp_transport and guide.ready:
                    from grasp_transport import proposals
                    cq=wp.to_torch(env.data_wp.qpos)[0]
                    cp_now,cr_now=projector.wrists(cq)
                    nominal_now=tlatent[sim:sim+config.horizon_steps].cpu().numpy()
                    # Generate grasp-transport proposals from the raw object path.
                    # Wrist/finger sampling, IK and physical guards are unchanged.
                    raw_refs=drop_context['reference_qpos'][:,:config.horizon_steps].cpu().numpy()
                    variants_by_reference=[proposals(g,nominal_now,cq.cpu().numpy(),
                        wp.to_torch(env.data_wp.ctrl)[0].cpu().numpy(),
                        cp_now.cpu().numpy(),cr_now.cpu().numpy(),config.sim_dt) for g in raw_refs]
                    variants=np.stack((variants_by_reference[0][0],variants_by_reference[1][1],
                                       variants_by_reference[2][0],variants_by_reference[2][3]))
                    # Floating-hand targets are visualization/optimization
                    # proposals; execution still passes through arm IK.
                    save_npz(out/'candidate_targets.npz',
                        time=sim/150+np.arange(config.horizon_steps+1)*config.sim_dt,
                        qpos=reference_paths,latent=variants[:3])
                    transport_candidates=torch.tensor(variants,device='cuda',dtype=torch.float32)
                    origin=nominal_now[0].copy()
                    for side in range(2):
                        j=side*28;origin[j:j+3]=cp_now[side].cpu().numpy()
                        angle=Rotation.from_matrix(cr_now[side].cpu().numpy()).as_euler('XYZ')
                        origin[j+3:j+6]=nominal_now[0,j+3:j+6]+np.arctan2(
                            np.sin(angle-nominal_now[0,j+3:j+6]),np.cos(angle-nominal_now[0,j+3:j+6]))
                    transport_origin=torch.tensor(origin,device='cuda',dtype=torch.float32)
                    regrasp_origin=transport_origin
                elif handover is not None or (side_gaps[0]>=3 and side_gaps[1]<=1) or (side_gaps[1]>=3 and side_gaps[0]<=1):
                    # Recover directly from the actual single-hand state even if
                    # bilateral contact anchors cannot be initialized.
                    cq=wp.to_torch(env.data_wp.qpos)[0]
                    cp_now,cr_now=projector.wrists(cq)
                    nominal_now=tlatent[sim:sim+config.horizon_steps].cpu().numpy()
                    origin=nominal_now[0].copy()
                    for side in range(2):
                        j=side*28;origin[j:j+3]=cp_now[side].cpu().numpy()
                        angle=Rotation.from_matrix(cr_now[side].cpu().numpy()).as_euler('XYZ')
                        origin[j+3:j+6]=nominal_now[0,j+3:j+6]+np.arctan2(
                            np.sin(angle-nominal_now[0,j+3:j+6]),np.cos(angle-nominal_now[0,j+3:j+6]))
                    regrasp_origin=torch.tensor(origin,device='cuda',dtype=torch.float32)
                if regrasp_origin is not None and handover is not None:
                    from regrasp_candidates import object_relative_goals
                    desired=object_relative_goals(reference_paths,qp[sim:sim+config.horizon_steps],tp[sim:sim+config.horizon_steps].cpu().numpy(),tR[sim:sim+config.horizon_steps].cpu().numpy(),tlatent[sim:sim+config.horizon_steps].cpu().numpy(),1-handover.support)
                    regrasp_goals=torch.tensor(desired,device='cuda',dtype=torch.float32)
                    drop_context['regrasp_goal_pos']=regrasp_goals[:,:,:3]
                    goalrot=Rotation.from_euler('XYZ',desired[:,:,3:6].reshape(-1,3)).as_matrix().reshape(2,config.horizon_steps,3,3)
                    drop_context['regrasp_goal_rot']=torch.tensor(goalrot,device='cuda',dtype=torch.float32)
                    from regrasp_candidates import supporting_turn_goals
                    hs=handover.support
                    turns=supporting_turn_goals(current,regrasp_origin.cpu().numpy()[hs*28:hs*28+6],reference_paths[:,1:],config.sim_dt)
                    support_turn_goals=torch.tensor(turns,device='cuda',dtype=torch.float32)
                    drop_context['support_turn_pos']=support_turn_goals[:,:,:3]
                    tr=Rotation.from_euler('XYZ',turns[:,:,3:6].reshape(-1,3)).as_matrix().reshape(2,config.horizon_steps,3,3)
                    drop_context['support_turn_rot']=torch.tensor(tr,device='cuda',dtype=torch.float32)
                    # Invert the bounded support-wrist trajectory to a shared
                    # effective object path, then transport the future free grasp.
                    from coherent_regrasp import paired_free_goals
                    moving,effective_objects=paired_free_goals(current,
                        regrasp_origin.cpu().numpy()[hs*28:hs*28+6],turns,
                        qp[sim+1:sim+config.horizon_steps+1],
                        tp[sim+1:sim+config.horizon_steps+1].cpu().numpy(),
                        tR[sim+1:sim+config.horizon_steps+1].cpu().numpy(),
                        tlatent[sim+1:sim+config.horizon_steps+1].cpu().numpy(),1-hs)
                    moving_regrasp_goals=torch.as_tensor(moving,device='cuda',dtype=torch.float32)
                    pairs=np.repeat(np.repeat(regrasp_origin.cpu().numpy().reshape(2,28)[None,None,:,:6],4,axis=0),config.horizon_steps,axis=1)
                    pairs[:2,:,1-hs]=desired[:,:,:6]
                    pairs[2:,:,hs]=turns
                    pairs[2:,:,1-hs]=moving[:,:,:6]
                    drop_context['paired_wrist_pos']=torch.as_tensor(pairs[:,:,:,:3],device='cuda',dtype=torch.float32)
                    pairR=Rotation.from_euler('XYZ',pairs[:,:,:,3:6].reshape(-1,3)).as_matrix().reshape(4,config.horizon_steps,2,3,3)
                    drop_context['paired_wrist_rot']=torch.as_tensor(pairR,device='cuda',dtype=torch.float32)
                    save_npz(out/'paired_regrasp_targets.npz',time=sim/150+np.arange(1,config.horizon_steps+1)*config.sim_dt,
                        support=np.array(hs),support_targets=turns,free_targets=moving,effective_objects=effective_objects,
                        paired_wrist_targets=pairs,current_qpos=current,current_wrist=regrasp_origin.cpu().numpy())
    
    
                from wrench_search import WrenchSearch
                wrench_search=WrenchSearch(model,current,wp.to_torch(env.data_wp.qvel)[0].cpu().numpy(),
                    controls.detach().cpu().numpy(),reference_paths[1],qp[sim:sim+config.horizon_steps+1],
                    tp[sim:sim+config.horizon_steps+1].cpu().numpy(),tR[sim:sim+config.horizon_steps+1].cpu().numpy(),
                    tlatent[sim:sim+config.horizon_steps].cpu().numpy(),config.sim_dt,config.ctrl_steps,
                    args.seed+sim,config.beta_traj,out/'wrench_search.jsonl',start_s=sim/150)
                wrench_goal_p=torch.tensor(np.stack((wrench_search.goals[:,:3],wrench_search.goals[:,28:31]),axis=1),device='cuda',dtype=torch.float32)
                wrench_goal_r=torch.tensor(wrench_search.goal_r,device='cuda',dtype=torch.float32)
                refs=tuple(x[sim+1:sim+config.horizon_steps+1] for x in tensors)
                if args.replay_checkpoint_controls:latest['feasible']=controls.clone()
                for it in range(0 if args.replay_checkpoint_controls else args.iterations):
                    progress.update('sampling',iteration=it+1,iteration_completed=it,rollout_step=0,force=True)
                    exploration=.8 if args.from_open_hand and sim<args.open_hold_start*150 else (.2 if sim<args.hold_seconds*150 else 1.)
                    controls,_=once(config,env,controls,refs,[{}],{'global_noise_scale':exploration*config.beta_traj**it})
                    progress.update('scoring',iteration_completed=it+1,feasible_count=latest.get('feasible_count',0),force=True)
                    if not args.headless_worker:live.update_opt_progress(it+1,args.iterations)
                    if it>=2 and 'feasible' in latest:break
                if args.contact_guard:dump(out/'constraint_diagnostics.json',dict(time_s=sim/150,feasible_count=latest.get('feasible_count',0),constraints=latest.get('constraint_diagnostics',{}),prior_feasible=latest.get('prior_feasible'),native_shortlist_checked=latest.get('native_shortlist_checked'),grasp_transport_feasible=latest.get('grasp_transport_feasible'),best_candidate_index=latest.get('best_candidate_index'),grasp_transport_gpu_feasible=latest.get('grasp_transport_gpu_feasible'),native_shortlist_ids=latest.get('native_shortlist_ids'),regrasp_candidate_side=latest.get('regrasp_candidate_side'),selected_candidate_index=latest.get('selected_candidate_index'),handover_support=None if handover is None else SIDES[handover.support],handover_elapsed_s=None if handover is None else handover.elapsed_frames/150))
                if args.diagnostic_only:
                    failure='Diagnostic only: no new physics executed';break
                if args.contact_guard and 'feasible' not in latest:
                    # Save the already-computed committed-window predictions for
                    # independent threshold/collision analysis; no search changes.
                    if shortlist_steps:
                        try:
                            save_npz(out/'diagnostic_predictions.npz',sim=np.array(sim),
                                qpos=np.stack([x[0].cpu().numpy() for x in shortlist_steps]),
                                qvel=np.stack([x[1].cpu().numpy() for x in shortlist_steps]),
                                score=latest.get('diagnostic_score',np.empty(0)),**latest.get('diagnostic_metrics',{}))
                        except Exception as exc:print('DIAGNOSTIC_SAVE_ERROR',str(exc),flush=True)
                    print(f'CONTACT_CANDIDATES {sim/150:.3f}s feasible={latest.get("feasible_count",0)}',flush=True)
                    failure=f'No feasible contact-safe candidate from {sim/150:.3f}s; checkpoint rollback required'
                    break
                if 'feasible' in latest:controls=latest['feasible']
                else:controls=latest.get('best',controls)
            mapped=project(controls)
            n=min(config.ctrl_steps,limit-sim,checkpoint_stride-(sim%checkpoint_stride))
            saved=physics.save_state(env);window_start_state=capture_dynamic_state(env)
            params=physics.save_env_params(config,env);pending=[];reject=None
            trial_gap=contact_gap_steps;last_good_gap=contact_gap_steps
            trial_side_gaps=side_gaps.copy();last_good_side_gaps=side_gaps.copy()
            trial_handover=handover.copy() if handover is not None else None
            angular_execution=AngularBudget(config.sim_dt)
            last_good_handover=handover.copy() if handover is not None else None
            for k in range(n):
                physical={}
                motor_control=actuator_control(env,mapped[k])
                scheduler.step(config,env,motor_control,perturb=False,sim_step=sim+k+1)
                wp.synchronize_device(env.data_wp.qpos.device)
                q=physics.get_qpos(config,env)[0].cpu().numpy().copy();v=physics.get_qvel(config,env)[0].cpu().numpy().copy()
                if args.strict_physics:
                    from open_hand_pipeline import telemetry
                    physical=telemetry(env,model)
                    actual_contact=solver_contacts.read(env.model_wp,env.data_wp)
                    physical.update(actual_hand_normal_force=actual_contact['normal'],actual_hand_force_world=actual_contact['world_force'],actual_contact_counts=actual_contact['counts'],actual_force_available=True,actual_object_contact_force=actual_contact['object_force_world'],actual_object_constraint_force=wp.to_torch(env.data_wp.qfrc_constraint)[0,-12:-6].cpu().numpy().copy(),actual_object_qacc=wp.to_torch(env.data_wp.qacc)[0,-12:-6].cpu().numpy().copy())
                    if args.replay_checkpoint_controls and k==n-1:
                        raw={}
                        for key in ('nacon','nefc','qfrc_constraint','qfrc_smooth','qacc','qacc_smooth','qvel'):
                            if hasattr(env.data_wp,key):raw[key]=wp.to_torch(getattr(env.data_wp,key)).cpu().numpy()
                        for key in ('geom','frame','dist','dim','efc_address','worldid','friction'):
                            raw['contact_'+key]=wp.to_torch(getattr(env.data_wp.contact,key)).cpu().numpy()
                        for key in ('type','id','J','force'):
                            raw['efc_'+key]=wp.to_torch(getattr(env.data_wp.efc,key))[0].cpu().numpy()
                        save_npz(out/'raw_solver_last.npz',**raw)
                    physical['actual_force_available']=bool(np.linalg.norm(actual_contact['object_force_world']-physical['actual_object_constraint_force'][:3])<.01)
                    solver_execution_ok,solver_execution_f,solver_execution_t=numerical_residual(env.data_wp)
                    physical['actual_force_available'] &= bool(solver_execution_ok[0])
                    probe.qpos[:]=q;probe.qvel[:]=v;probe.ctrl[:]=motor_control[0].cpu().numpy()
                    mujoco.mj_forward(model,probe)
                    required=(sim+k+1)/150>=args.contact_required_after
                    reject=native_state_failure(model,probe,q,qp[sim+k+1],objects,hands,vertices,required)
                    if args.continue_on_task_failure and reject and reject.startswith(('object bottom','position error','rotation error')):
                        if len(task_warnings)<100:task_warnings.append({'time_s':(sim+k+1)/150,'warning':reject})
                        reject=None
                    if args.continue_on_task_failure:
                        rr=Rotation.from_quat(q[[-10,-9,-8,-11]])
                        if (vertices@rr.as_matrix().T+q[-14:-11])[:,2].min()<-.01:
                            reject='object penetrates support plane by more than 10 mm'
                    if required:
                        if trial_handover is None:
                            drop_ok,drop_reason=drop_allowed(q,qp[sim+k+1],vertices,drop_context['start_bottom'],drop_context['reference_start_bottom'])
                        else:
                            hs=trial_handover.support
                            ob=observe_handover(model,probe,q,objects,hands,hs)
                            if physical['actual_force_available']:ob['normal']=actual_contact['normal']
                            drop_ok,drop_reason=trial_handover.observe_step(ob['hits'],ob['normal'],ob['relative_both'],bottom_height(q,vertices),v[-10],ob['table'])
                            hs=trial_handover.support;ob['relative']=ob['relative_both'][hs]
                            physical.update(handover_support=hs,handover_control_valid=drop_ok,handover_support_contact=ob['hits'][hs],handover_normal_force=ob['normal'][hs],handover_relative_drift=float(np.linalg.norm(ob['relative']-trial_handover.anchor_local)),handover_table_contact=ob['table'],handover_table_frames=trial_handover.table_frames,handover_anchor_local=trial_handover.anchor_local.copy(),handover_elapsed_frames=trial_handover.elapsed_frames)
                            if not drop_ok:drop_reason='controlled handover: '+drop_reason
                        speed_ok,speed_reason=angular_execution.velocity_step(v,vp[sim+k+1],bool(physical['actual_force_available'] and np.any(actual_contact['normal']>.02)),is_end=(k==n-1))
                        if not drop_ok:reject='DROP_GUARD: '+drop_reason
                        elif not speed_ok:reject='DROP_GUARD: '+speed_reason
                    hits=[False,False]
                    for con in probe.contact:
                        x,y=map(int,con.geom)
                        if con.dist<=.0005 and ((x in objects and y in hands) or (y in objects and x in hands)):
                            name=model.geom(y if x in objects else x).name or ''
                            for side,side_name in enumerate(SIDES):hits[side]|=name.startswith('collision_hand_'+side_name+'_')
                    trial_gap=0 if any(hits) else trial_gap+1
                    trial_side_gaps=[0 if hit else trial_side_gaps[side]+1 for side,hit in enumerate(hits)]
                    if required and not args.continue_on_task_failure and max(trial_side_gaps)>round(args.contact_gap*150):reject='loss of bilateral load-bearing grasp'
                    if physical['object_weld_active'] or not np.allclose(physical['gravity'],[0,0,-9.81],atol=1e-5) or np.any(physical['object_external_force']!=0):
                        reject='full-gravity no-assistance invariant violated'
                elif args.first_pass:
                    probe.qpos[:]=q;mujoco.mj_forward(model,probe)
                    hand_contacts=[c for c in probe.contact if ((int(c.geom[0]) in objects and int(c.geom[1]) in hands) or (int(c.geom[1]) in objects and int(c.geom[0]) in hands))]
                    trial_gap=0 if any(c.dist<=.0005 for c in hand_contacts) else trial_gap+1
                    depth=max([-float(c.dist) for c in hand_contacts]+[0.])
                    distance=float(np.linalg.norm(q[-14:-11]-qp[sim+k+1,-14:-11]))
                    rot=Rotation.from_quat(q[[-10,-9,-8,-11]])
                    bottom=float((vertices@rot.as_matrix().T+q[-14:-11])[:,2].min())
                    if not np.isfinite(q).all():reject='non-finite state'
                    elif depth>.015:reject=f'gross hand penetration {depth*1000:.1f} mm'
                    elif bottom<-.01:reject=f'object below table {bottom*1000:.1f} mm'
                    elif distance>.20:reject=f'gross object displacement {distance*1000:.1f} mm'
                    elif trial_gap>30 and distance>.05:reject='sustained loss of grasp'
                    if args.contact_guard:
                        touched=[False,False]
                        for con in hand_contacts:
                            if con.dist>.0005:continue
                            x,y=map(int,con.geom);name=model.geom(y if x in objects else x).name or ''
                            for s,side in enumerate(SIDES):touched[s]|=name.startswith('collision_hand_'+side+'_')
                        trial_side_gaps=[0 if hit else trial_side_gaps[s]+1 for s,hit in enumerate(touched)]
                        if reject is None and max(trial_side_gaps)>round(args.contact_gap*150):reject='loss of '+SIDES[int(np.argmax(trial_side_gaps))]+' hand contact'
                        if reject is None and distance>.08:reject='contact guard: object drift exceeds 80 mm'
                        if reject is None and depth>.008:reject='contact guard: excessive hand penetration'
                else:reject=native_state_failure(model,probe,q,qp[sim+k+1],objects,hands,vertices,True)
                if reject is None:
                    for con in probe.contact:
                        if (int(con.geom[0]) in arms or int(con.geom[1]) in arms) and con.dist<(-.001 if args.strict_physics else (-.005 if args.first_pass else -.001)):
                            reject=f'arm collision {-con.dist*1000:.2f} mm';break
                if reject is None and args.strict_physics:
                    from wrist_clearance import allowed as wrist_clearance_allowed
                    if not wrist_clearance_allowed(model,probe,wrist_collision_geoms):
                        reject='EXECUTION_GUARD: cross wrist clearance <=1mm'
                if args.strict_physics and not bool(solver_execution_ok[0]):
                    reject=f'EXECUTION_GUARD: solver residual F={float(solver_execution_f[0]):.6g} N, T={float(solver_execution_t[0]):.6g} Nm'
                if reject and args.record_through_failure and reject!='non-finite state':
                    if len(observed_failures)<50:
                        observed_failures.append({'time_s':(sim+k+1)/150,'reason':reject})
                    reject=None
                if reject:break
                u=motor_control[0] if motor_control.ndim==2 else motor_control
                pending.append((q,v,u.cpu().numpy().copy(),controls[k].cpu().numpy().copy()))
                if args.strict_physics:physical_records[sim+k+1]=physical
                last_good_gap=trial_gap
                last_good_side_gaps=trial_side_gaps.copy()
                last_good_handover=trial_handover.copy() if trial_handover is not None else None
                # Keep an exact rollback point at every accepted native step.
                physics.save_state(env)
                if approach_fast and contact_phase.observe(hits,actual_contact['counts'],actual_contact['normal']):
                    contact_event=True
                    n=len(pending)
                    dump(out/'first_physical_contact.json',dict(time_s=(sim+k+1)/150,
                        geometric_hits=hits,solver_counts=actual_contact['counts'].tolist(),
                        normal_force_N=actual_contact['normal'].tolist(),
                        next_action='start MPC from this accepted physical state'))
                    print(f'PRECONTACT_END {(sim+k+1)/150:.6f}s; start contact MPC',flush=True)
                    break
            if reject:
                # Preserve attempted physics before restoring any checkpoint.
                save_npz(out/'rejected_execution.npz',time=np.arange(len(pending)+1)/150+(sim+1)/150,qpos=np.array([x[0] for x in pending]+[q]),qvel=np.array([x[1] for x in pending]+[v]),ctrl=np.array([x[2] for x in pending]+[(motor_control[0] if motor_control.ndim==2 else motor_control).cpu().numpy()]),actual_hand_normal_force=np.array([physical_records.get(sim+i+1,{}).get('actual_hand_normal_force',np.full(2,np.nan)) for i in range(len(pending))]+[physical.get('actual_hand_normal_force',np.full(2,np.nan))]),actual_hand_force_world=np.array([physical_records.get(sim+i+1,{}).get('actual_hand_force_world',np.full((2,3),np.nan)) for i in range(len(pending))]+[physical.get('actual_hand_force_world',np.full((2,3),np.nan))]))
                extra_force={key:np.array([physical_records.get(sim+i+1,{}).get(key,np.full(3 if key=='actual_object_contact_force' else 6,np.nan)) for i in range(len(pending))]+[physical[key]]) for key in ('actual_object_contact_force','actual_object_constraint_force','actual_object_qacc')}
                save_npz(out/'rejected_force_balance.npz',time=np.arange(len(pending)+1)/150+(sim+1)/150,**extra_force)
                if args.replay_checkpoint_controls:
                    raw={}
                    for key in ('nacon','nefc','qfrc_constraint','qfrc_smooth','qacc','qacc_smooth','qvel'):
                        if hasattr(env.data_wp,key):raw[key]=wp.to_torch(getattr(env.data_wp,key)).cpu().numpy()
                    for key in ('geom','frame','dist','dim','efc_address','worldid','friction'):
                        raw['contact_'+key]=wp.to_torch(getattr(env.data_wp.contact,key)).cpu().numpy()
                    for key in ('type','id','J','force'):
                        raw['efc_'+key]=wp.to_torch(getattr(env.data_wp.efc,key))[0].cpu().numpy()
                    save_npz(out/'raw_solver_rejected.npz',**raw)
                # save_state uses one shared buffer that is updated for each
                # successful native substep. On a guard rejection the entire
                # uncommitted window must return to its cloned start state.
                if reject.startswith(('DROP_GUARD:','EXECUTION_GUARD:')):
                    restore_dynamic_state(env,window_start_state)
                else:
                    physics.load_state(env,saved)
                physics.load_env_params(config,env,params)
                native_rejects+=1
                print(f'NATIVE_REJECT {sim/150:.3f}s at {(sim+k+1)/150:.3f}s attempt={native_rejects}: {reject}',flush=True)
                if reject.startswith(('DROP_GUARD:','EXECUTION_GUARD:')):
                    failure=f'{reject}; predicted or attempted time {(sim+k+1)/150:.3f}s; rewind required'
                    dump(out/('execution_guard_rejection.json' if reject.startswith('EXECUTION_GUARD:') else 'drop_guard_rejection.json'),dict(time_s=(sim+k+1)/150,accepted_time_s=sim/150,reason=reject))
                    break
                if pending:
                    # Use the already validated prefix and replan before the
                    # invalid contact, shortening the execution window safely.
                    n=len(pending)
                    print(f'ADAPTIVE_PREFIX {n} steps; replan at {(sim+n)/150:.3f}s',flush=True)
                else:
                    if native_rejects>=args.native_retries:
                        failure=f'Native execution rejected after {args.native_retries} attempts at {(sim+k+1)/150:.3f}s: {reject}'
                        break
                    continue
            native_rejects=0
            contact_gap_steps=last_good_gap
            side_gaps=last_good_side_gaps.copy()
            handover=last_good_handover
            if handover is not None and pending and handover.finished(bottom_height(pending[-1][0],vertices)):
                handover=None
            for k,(q,v,u,l) in enumerate(pending):
                if trajectory_servo is not None:
                    d=mapped[k,trajectory_servo.ids].cpu().numpy().copy();dv=mapped[k,56:68].cpu().numpy().copy()
                else:d=np.full(12,np.nan);dv=np.full(12,np.nan)
                at=sim+k+1;native.append((q,v,u,l,at/150,d,dv))
                if at%5==0:record(at,q,v,u,l)
            if not args.no_publish_live:
                display=[(q,(sim+k+1)/150) for k,(q,v,u,l) in enumerate(pending) if (sim+k+1)%5==0]
                if display:save_npz(ROOT/'live_batch.npz',qpos=np.array([x[0] for x in display]),time=np.array([x[1] for x in display]),run=np.array(args.name))
            if len(native)>=2:
                last_command_velocity.copy_(torch.tensor((native[-1][2]-native[-2][2])/config.sim_dt,device='cuda',dtype=torch.float32))
            if trajectory_servo is not None:
                old_v=trajectory_servo.v_des.clone() if n==1 else mapped[n-2,56:68]
                trajectory_servo.accept(mapped[n-1])
                trajectory_servo.a_des=(mapped[n-1,56:68]-old_v)/config.sim_dt
            sim+=n
            if args.contact_guard and not guide.ready:
                probe.qpos[:]=native[-1][0];probe.qvel[:]=native[-1][1]
                mujoco.mj_forward(model,probe)
                try:
                    guide.initialize(probe)
                    guide.description['source_time_s']=sim/150
                    dump(out/'contact_anchors.json',guide.description)
                except ValueError:pass
            physics.sync_env(config,env,view)
            controls=append_continuous_controls(controls[n:],tlatent[sim+config.horizon_steps-n:sim+config.horizon_steps])
            if sim%checkpoint_stride==0 or contact_event:save_checkpoint()
            progress.update('first_contact' if contact_event else ('precontact_approach' if approach_fast else 'accepted'),accepted_time_s=sim/150,saved_time_s=sim/150,force=True)
            dump(out/'progress.json',{'time_s':sim/150,'hold_passed':hold_passed,'object_error_m':records[-1][-1]['object_error_m']})
            print(f'PROGRESS {sim/150:.3f}s hold={hold_passed} object_error={records[-1][-1]["object_error_m"]:.5f}',flush=True)
            finish_profile()
    except Exception as ex:
        failure=str(ex);traceback.print_exc()
    finally:
        finish_profile()
        save_checkpoint()
        result={'completed':sim>=limit and failure is None,'hold_passed':None,'failure':failure,
                'time_s':sim/150,'wall_s':time.monotonic()-started,'floating_wrist_actuators':False,'grasp_hold_test_enabled':False,
                'validation_mode':'exploratory_first_pass' if args.first_pass else 'strict',
                'observed_failures':observed_failures,'task_warnings':task_warnings,'grasp_verified_time_s':grasp_verified_time,'grasp_success':None,
                'from_open_hand':args.from_open_hand,'strict_full_gravity_policy':args.strict_physics,
                'separate_ik_trajectory_and_servo_input':args.trajectory_servo,
                'wrist_samples_projected_through_arm_ik':True,'object_assistance':False,'arguments':vars(args)}
        dump(out/'result.json',result);print(json.dumps(result,default=lambda x:x.item()),flush=True)
        if not args.no_publish_live:dump(ROOT/'live_status.json',dict(run=args.name,state='completed' if result['completed'] else 'stopped',**result))
        status.content=json.dumps(result,ensure_ascii=False,indent=2)
    while args.keep_viewer:time.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='trial_01')
    p.add_argument('--plan-result',default='mink_result_004.json')
    p.add_argument('--samples',type=int,default=64);p.add_argument('--iterations',type=int,default=6)
    p.add_argument('--horizon',type=float,default=1.2);p.add_argument('--ctrl-dt',type=float,default=.2)
    p.add_argument('--hold-seconds',type=float,default=2.);p.add_argument('--max-seconds',type=float,default=0.)
    p.add_argument('--acceleration',type=float,default=20.);p.add_argument('--ik-iterations',type=int,default=8)
    p.add_argument('--port',type=int,default=8083);p.add_argument('--keep-viewer',action='store_true')
    p.add_argument('--headless-worker',action='store_true')
    p.add_argument('--resume-from')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--native-screen',action='store_true')
    p.add_argument('--native-shortlist',type=int,default=0)
    p.add_argument('--accepted-finger-seed',action='store_true')
    p.add_argument('--accepted-hand-seed',action='store_true')
    p.add_argument('--velocity-feedforward',action='store_true')
    p.add_argument('--no-publish-live',action='store_true')
    p.add_argument('--first-pass',action='store_true')
    p.add_argument('--record-through-failure',action='store_true')
    p.add_argument('--continue-on-task-failure',action='store_true')
    p.add_argument('--known-failure-time',type=float)
    p.add_argument('--continuous-arm-commands',action='store_true')
    p.add_argument('--trajectory-servo',action='store_true')
    p.add_argument('--reset-initial-controls',action='store_true')
    p.add_argument('--smooth-commands',action='store_true')
    p.add_argument('--arm-jerk',type=float,default=400.)
    p.add_argument('--arm-tracking-rate',type=float,default=25.)
    p.add_argument('--finger-speed',type=float,default=6.)
    p.add_argument('--finger-acceleration',type=float,default=120.)
    p.add_argument('--contact-guard',action='store_true')
    p.add_argument('--contact-prior',action='store_true')
    p.add_argument('--grasp-transport',action='store_true')
    p.add_argument('--from-open-hand',action='store_true')
    p.add_argument('--strict-physics',action='store_true')
    p.add_argument('--coupled-wrist-sampling',action='store_true')
    p.add_argument('--contact-weight',type=float,default=15.)
    p.add_argument('--wrist-radius',type=float,default=None)
    p.add_argument('--contact-gap',type=float,default=.10)
    p.add_argument('--native-retries',type=int,default=8)
    p.add_argument('--replay-checkpoint-controls',action='store_true');p.add_argument('--diagnostic-only',action='store_true')
    p.add_argument('--diagnostic-contact-backend',action='store_true')
    p.add_argument('--checkpoint-interval',type=float,default=1.)
    p.add_argument('--speed-guidance-weight',type=float,default=1.)
    p.add_argument('--drop-guidance-weight',type=float,default=1.)
    args=p.parse_args()
    if args.from_open_hand:
        args.strict_physics=True
        args.coupled_wrist_sampling=True
        args.trajectory_servo=True
        args.smooth_commands=True
        args.contact_guard=True
    if args.native_shortlist and not args.contact_guard:
        p.error('native-shortlist requires contact-guard')
    if args.strict_physics and not args.from_open_hand:
        p.error('strict-physics requires from-open-hand provenance')
    if args.wrist_radius is None:args.wrist_radius=.05 if args.contact_guard else .015
    if args.continuous_arm_commands and args.velocity_feedforward:
        p.error('The continuity experiment requires velocity feedforward to remain disabled so command and target coordinates match.')
    if args.trajectory_servo and (args.continuous_arm_commands or args.velocity_feedforward):
        p.error('trajectory-servo replaces the older experimental control modes')
    if args.smooth_commands and not args.trajectory_servo:
        p.error('smooth-commands requires trajectory-servo')
    run(args)
