"""SPIDER-style sampling MPC for the dual-UR3e + Sharpa bowl demo.

Robot, scene, open-hand initialization, wrist->arm IK projection and the arm
trajectory servo are the same as arm_mpc.py. The optimization follows SPIDER
(facebookresearch/spider, spider/optimizers/sampling.py + simulators/mjwp.py):

  * 2048 samples, up to 16 annealed iterations; stop as soon as an iteration's
    best sample improves on the unperturbed sample by less than 0.01
  * 1.6 s horizon, execute 0.4 s, sim_dt 0.01
  * reward = -||weighted tracking error||_2 - 1e-4 ||qvel error||_2
    (wrist pos/rot, finger joints, object pose); soft only, nothing is rejected.
    The object position weight is --w-obj-pos, or --obj-weight-after-contact
    from the human's first contact (schedule.json) on
  * execute the softmax-weighted mean (--execute mean, default) or the best
    rollout seen in the window (--execute best)
  * --fallback stable (not in SPIDER): sampling and execution as above; when
    the object drops more than --fall-drop below the reference during an
    executed ctrl_dt, go back to the latest moment (any sim step, not only a
    window start) where the grasp had been stable for --stable-hold seconds,
    restore that state and continue optimizing from there. Stable: at least
    --stable-hands hands (default one) touch the object within +-0.05 s and
    the object is no more than --sink-tol below the reference. Each stable point is retried at most
    --fallback-retries times, then an earlier one is used.
  * --fallback window (not in SPIDER): after each executed ctrl_dt, if
    the object dropped more than --fall-drop below the reference, the window
    is rejected, the state is restored and the window is re-optimized with new
    samples, with the sampling noise scaled by --retry-noise per retry.
    Rotation error is only logged: the captured object rotation is not exact
    enough to gate on. During sampling, rollouts in which the object drops
    more than --fall-drop below the reference anywhere in the horizon are
    scored below every rollout that keeps it, so they get no weight. After
    --fallback-retries failed retries the previous window is redone (at most
    --fallback-depth windows back). In both modes the run fails when no
    restart point is left or after --fallback-max fallbacks.

The wrist terms use the wrist pose from arm FK, since qpos[0:6] are UR3e joint
angles here rather than SPIDER's floating-wrist coordinates. Before the first
hand-object contact the planned open-hand approach is executed without
optimization, as in arm_mpc.py. There is no checkpoint rollback.
"""
import argparse,json,signal,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import mujoco_warp as mjwarp
import warp as wp
from scipy.spatial.transform import Rotation,Slerp
from build_arm_scene import ROOT,OLD,TASK
from projector import WristProjector
from arm_servo import ArmTrajectoryServo
from smooth_targets import smooth_targets
from compute_progress import ComputeProgress
from open_hand_pipeline import prepare_open,FullGravity,FINGERS
from retargeting.config import Config,process_config
from retargeting.utils import mjwp as physics
from retargeting.utils.sampling import make_rollout_fn,make_optimize_once_fn
from retargeting.pipeline.optimize_contact_init import append_continuous_controls

# SPIDER reward weights (spider/config.py defaults).
W_WRIST_POS=1.    # base_pos_rew_scale
W_WRIST_ROT=.3    # base_rot_rew_scale
W_FINGER=.003     # joint_rew_scale
W_OBJ_POS=1.      # pos_rew_scale (default of --w-obj-pos)
W_OBJ_ROT=.3      # rot_rew_scale (default of --w-obj-rot)
W_VEL=1e-4        # vel_rew_scale
W_TERMINAL=1.     # terminal_rew_scale
# Arm/finger command shaping, unchanged from the arm_mpc.py demo launch.
ARM_ACCELERATION=20.;ARM_JERK=1200.;ARM_TRACKING_RATE=40.
FINGER_SPEED=6.;FINGER_ACCELERATION=120.;IK_ITERATIONS=8
# Saved for opt_viewer.py: predicted wrist/object paths of the best and of
# evenly spaced samples, every TRACE_STRIDE rollout steps, per iteration.
TRACE_STRIDE=4;TRACE_TOP=24;TRACE_UNIFORM=24

def dump(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def save_npz(path,**arrays):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(path)

def prepare(args):
    """Open-hand scene and reference (built at 150 Hz), resampled to sim_dt."""
    legacy=SimpleNamespace(name=args.name,plan_result=args.plan_result,acceleration=ARM_ACCELERATION,
        arm_jerk=ARM_JERK,smooth_commands=True,arm_tracking_rate=ARM_TRACKING_RATE,contact_prior=False)
    out,model,_,qr,qv,latent,wpos,wrot,qinit,_=prepare_open(legacy)
    t150=np.arange(len(qr))/150;t=np.arange(0,t150[-1]+1e-9,args.sim_dt)
    lin=lambda x:np.stack([np.interp(t,t150,c) for c in x.reshape(len(x),-1).T],-1).reshape((len(t),)+x.shape[1:])
    q=lin(qr)
    for a in (59,66):  # object quaternions (w,x,y,z)
        q[:,a:a+4]=Slerp(t150,Rotation.from_quat(qr[:,[a+1,a+2,a+3,a]]))(t).as_quat(scalar_first=True)
    R=np.stack([Slerp(t150,Rotation.from_matrix(wrot[:,s]))(t).as_matrix() for s in range(2)],1)
    return out,model,q,lin(qv),lin(latent),lin(wpos),R,qinit

DROP_PENALTY=10.  # far below any tracking score (scores are O(0.1-1))
# Data fields that define the simulation state (retargeting.utils.mjwp._copy_state).
STATE_FIELDS=('qpos','qvel','qacc','time','ctrl','act','act_dot','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat')

def window_violation(qpos,qref):
    """Max object rotation error (deg) and max drop below the reference (m) over executed states."""
    dot=np.abs((qpos[:,-11:-7]*qref[:,-11:-7]).sum(1)).clip(max=1)
    return float(np.degrees(2*np.arccos(dot)).max()),float((qref[:,-12]-qpos[:,-12]).max())

def run(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    wp.set_device('cuda:0');torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream('cuda:0')))
    out,model,qref,vref,lref,wpos,wrot,qinit=prepare(args)
    from numerical_solver import install as install_numerical_solver
    from cached_broadphase import install as install_broadphase_cache
    install_broadphase_cache();install_numerical_solver(model)
    config=Config(output_root_dir=str(out/'model'),dataset_name='bihoi',task=TASK,robot_type='sharpa',
        embodiment_type='bimanual',sim_dt=args.sim_dt,ref_dt=1/30,ctrl_dt=args.ctrl_dt,horizon=args.horizon,
        knot_dt=args.knot_dt,num_samples=args.samples,max_num_iterations=args.iterations,
        temperature=args.temperature,improvement_threshold=args.improvement_threshold,
        pos_noise_scale=.03,rot_noise_scale=.03,joint_noise_scale=.15,
        first_ctrl_noise_scale=.5,last_ctrl_noise_scale=1.,final_noise_scale=.1,zero_first_knot_noise=False,
        use_torch_compile=False,nconmax_per_env=700,njmax_per_env=2500,
        num_perturb_samples=1,warmup_duration=0,show_viewer=False,save_video=False,save_info=True,force=True)
    config=process_config(config);config.output_dir=str(out)
    dt=config.sim_dt;H=config.horizon_steps
    limit=min(len(qref)-1,round(args.end/dt));config.max_sim_steps=limit
    # Object position weight per sim step: --obj-weight-after-contact from the human's first contact on (schedule.json, 30 Hz simulation frames).
    first_contact_s=json.loads((OLD/'trial_accepted_v2/schedule.json').read_text())['first_contact_frame']/30
    obj_after=args.w_obj_pos if args.obj_weight_after_contact is None else args.obj_weight_after_contact
    dump(out/'spider_config.json',dict(samples=config.num_samples,max_iterations=config.max_num_iterations,
        improvement_threshold=config.improvement_threshold,horizon_s=config.horizon,ctrl_dt_s=config.ctrl_dt,
        knot_dt_s=config.knot_dt,sim_dt_s=dt,temperature=config.temperature,beta_traj=config.beta_traj,
        reward_weights=dict(wrist_pos=W_WRIST_POS,wrist_rot=W_WRIST_ROT,finger=W_FINGER,object_pos=args.w_obj_pos,
            object_pos_after_contact=obj_after,human_first_contact_s=first_contact_s,
            object_rot=args.w_obj_rot,qvel=W_VEL,terminal=W_TERMINAL),executed='best rollout' if args.execute=='best' else 'weighted mean',hard_constraints=False))
    progress=ComputeProgress(ROOT/'compute_progress.json',args.name,candidate_count=config.num_samples,
        iteration_total=config.max_num_iterations,horizon_s=config.horizon,commit_s=config.ctrl_dt)

    # Reference tensors, padded so that a full horizon is always available.
    pad=lambda x:np.r_[x,np.repeat(x[-1:],H+2,axis=0)]
    cuda=lambda x:torch.tensor(pad(x),device='cuda',dtype=torch.float32)
    tq,tv,tlatent,tp,tR=map(cuda,(qref,vref,lref,wpos,wrot))
    tensors=(tq,tv,tq[:,:56],torch.zeros((len(tq),10),device='cuda'),torch.zeros((len(tq),10,3),device='cuda'))
    w_obj=torch.where(torch.arange(len(tq),device='cuda')>=round(first_contact_s/dt),obj_after,args.w_obj_pos)

    resume=start=None
    if args.resume_from:
        # Continue a saved run: its state at --start becomes the initial state (same sim_dt grid).
        resume=np.load(args.resume_from);start=int(round(args.start/dt))
        if not np.isclose(resume['time'][start],start*dt) or not np.allclose(np.diff(resume['time'][:start+1]),dt):
            raise SystemExit(f'{args.resume_from} is not on the {dt}s grid up to {args.start}s')
        qinit=resume['qpos'][start]
    env=physics.setup_env(config,tensors)
    physics.set_qpos(config,env,torch.tensor(qinit,device='cuda',dtype=torch.float32))
    wp.to_torch(env.data_wp.qvel)[:]=0 if resume is None else torch.tensor(resume['qvel'][start],device='cuda',dtype=torch.float32)
    wp.to_torch(env.data_wp.qacc_warmstart)[:]=0
    if resume is not None:wp.to_torch(env.data_wp.ctrl)[:]=torch.tensor(resume['ctrl'][start],device='cuda',dtype=torch.float32)
    scheduler=FullGravity(config,env);scheduler.apply(0);mjwarp.forward(env.model_wp,env.data_wp)
    projector=WristProjector(model,dt,acceleration_limit=ARM_ACCELERATION,jerk_limit=ARM_JERK,tracking_rate=ARM_TRACKING_RATE)
    servo=ArmTrajectoryServo(model,dt)
    servo.reset(wp.to_torch(env.data_wp.qpos)[0],wp.to_torch(env.data_wp.qvel)[0])
    fingers=torch.tensor(FINGERS,device='cuda')
    last_command_velocity=torch.zeros(model.nu,device='cuda')
    if resume is not None:
        last_command_velocity[:]=torch.tensor((resume['ctrl'][start]-resume['ctrl'][start-1])/dt,device='cuda')
    objects={g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('right_object').id}
    hands={g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('collision_hand_')}
    probe=mujoco.MjData(model)

    def actuator_control(e,action):
        # Restored states do not carry bias forces; recompute before the servo law.
        mjwarp.kinematics(e.model_wp,e.data_wp);mjwarp.com_pos(e.model_wp,e.data_wp)
        mjwarp.com_vel(e.model_wp,e.data_wp);mjwarp.rne(e.model_wp,e.data_wp)
        return servo.control(action,wp.to_torch(e.data_wp.qfrc_bias))
    def controlled_step(c,e,action,perturb=False,sim_step=0):
        return scheduler.step(c,e,actuator_control(e,action),perturb=False,sim_step=sim_step)
    def project(latent):
        """Wrist SE(3) + finger samples -> bounded UR3e joint and finger commands."""
        q0=wp.to_torch(env.data_wp.qpos)[0];v0=wp.to_torch(env.data_wp.qvel)[0]
        bq,bv=servo.boundary(q0,v0)
        mapped=projector.project(latent,tq[sim:sim+H],bq,bv,iterations=IK_ITERATIONS,astart=servo.a_des,ik_start=q0)[0]
        mapped[...,fingers]=smooth_targets(mapped[...,fingers],wp.to_torch(env.data_wp.ctrl)[0,fingers],
            last_command_velocity[fingers],dt,FINGER_SPEED,FINGER_ACCELERATION,tracking_rate=40.)
        return servo.pack(mapped)

    def reward(c,e,ref,sim_step=0):
        """SPIDER get_reward: -|weighted tracking error| - w_vel |qvel - qvel_ref|."""
        q=wp.to_torch(e.data_wp.qpos);v=wp.to_torch(e.data_wp.qvel)
        p,R=projector.wrists(q)
        cos=((tR[sim_step].transpose(-1,-2)@R).diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
        wrist_angle=torch.acos(cos.clamp(-1+1e-6,1-1e-6))
        object_angle=2*torch.acos((q[:,-11:-7]*ref[0][-11:-7]).sum(-1).abs().clamp(max=1-1e-7))
        # Rotation errors enter as angles: the L2 norm only needs |rotvec| = angle.
        diff=torch.cat([W_WRIST_POS*(p-tp[sim_step]).flatten(1),W_WRIST_ROT*wrist_angle,
            W_FINGER*(q[:,fingers]-ref[0][fingers]),w_obj[sim_step]*(q[:,-14:-11]-ref[0][-14:-11]),
            (args.w_obj_rot*object_angle)[:,None]],dim=1)
        qpos_dist=diff.norm(dim=1);qvel_dist=(v-ref[1]).norm(dim=1)
        if args.fallback=='window':
            # Largest drop below the reference so far in this rollout.
            drop=ref[0][-12]-q[:,-12]
            fall[0]=drop if sim_step-config.warmup_sim_step==1 else torch.maximum(fall[0],drop)
        if (sim_step-config.warmup_sim_step)%TRACE_STRIDE==0:
            trace_buffer.append(torch.cat([p,q[:,None,-14:-11]],dim=1).detach().clone())  # right wrist, left wrist, object
        progress.update('rollout',rollout_step=int(sim_step-config.warmup_sim_step),rollout_total=H)
        return -qpos_dist-W_VEL*qvel_dist,{'qpos_dist':qpos_dist,'qvel_dist':qvel_dist,
            'object_pos_error':(q[:,-14:-11]-ref[0][-14:-11]).norm(dim=1)}
    base=make_rollout_fn(controlled_step,physics.save_state,physics.load_state,reward,
        lambda c,e,r,sim_step=0:(lambda x:(W_TERMINAL*x[0],x[1]))(reward(c,e,r,sim_step)),
        physics.get_trace,physics.save_env_params,physics.load_env_params)
    trace_buffer=[];window_samples=[];best={};fall=[None]
    def rollout(c,e,latent,refs,params):
        trace_buffer.clear()
        _,score,info=base(c,e,project(latent),refs,params)
        if args.fallback=='window':
            # Rollouts that drop the object rank below all that keep it (ordered among themselves by score).
            dropped=fall[0]>args.fall_drop;info['dropped']=dropped.float()
            score=torch.where(dropped,score-DROP_PENALTY,score)
        finite=torch.where(torch.isfinite(score),score,-torch.inf);k=int(finite.argmax())
        if not best or float(finite[k])>best['score']:
            best.update(score=float(finite[k]),latent=latent[k].clone(),iteration=len(window_samples),sample=k)
        paths=torch.stack(trace_buffer,dim=1)  # (N, T, 3 points, xyz)
        top=torch.topk(torch.where(torch.isfinite(score),score,-torch.inf),min(TRACE_TOP,len(score))).indices
        ids=torch.cat([top,torch.linspace(0,len(score)-1,TRACE_UNIFORM,device=score.device).long()])
        window_samples.append(dict(paths=paths[ids].cpu().numpy(),ids=ids.cpu().numpy(),score=score[ids].cpu().numpy(),
            all_scores=score.cpu().numpy(),qpos_dist=info['qpos_dist'].cpu().numpy()))
        return latent,score,info
    def save_samples(window):
        """One file per optimized window: every iteration's sampled predictions."""
        steps=np.arange(TRACE_STRIDE,H+1,TRACE_STRIDE)+sim
        index=torch.as_tensor(steps,device='cuda')
        reference=torch.cat([tp[index],tq[index][:,None,-14:-11]],dim=1).cpu().numpy()
        (out/'samples').mkdir(exist_ok=True)
        save_npz(out/'samples'/f'{window:04d}.npz',time_s=np.array(sim*dt),path_time=steps*dt,reference=reference,top=np.array(TRACE_TOP),
            **{k:np.stack([s[k] for s in window_samples]) for k in window_samples[0]})
    optimize_once=make_optimize_once_fn(rollout)

    def touching(q):
        probe.qpos[:]=q;mujoco.mj_forward(model,probe)
        return any(con.dist<=.0005 and ((int(con.geom[0]) in objects and int(con.geom[1]) in hands) or
            (int(con.geom[1]) in objects and int(con.geom[0]) in hands)) for con in probe.contact)
    traj={k:[] for k in ('time','qpos','qvel','ctrl','latent')}
    def record(t,q,v,u,l):
        for key,value in zip(traj,(t,q,v,u,l)):traj[key].append(value)
    def save():
        save_npz(out/'trajectory.npz',**{k:np.array(v) for k,v in traj.items()},
            object_error=np.linalg.norm(np.array(traj['qpos'])[:,-14:-11]-qref[:len(traj['qpos']),-14:-11],axis=1))
    def status(state):
        dump(ROOT/'continuous_status.json',dict(run=args.name,state=state,requested_end_s=args.end,best_time_s=sim*dt,pid=None))
        dump(ROOT/'live_status.json',dict(run=args.name,state=state,time_s=sim*dt))

    def checkpoint():
        data={f:wp.to_torch(getattr(env.data_wp,f)).clone() for f in STATE_FIELDS if getattr(env.data_wp,f).size}
        return dict(sim=sim,length=len(traj['time']),controls=controls.clone(),data=data,
            servo=(servo.q_des.clone(),servo.v_des.clone(),servo.a_des.clone()),command_velocity=last_command_velocity.clone())
    def restore(saved):
        for f,x in saved['data'].items():wp.to_torch(getattr(env.data_wp,f)).copy_(x)
        servo.q_des,servo.v_des,servo.a_des=(x.clone() for x in saved['servo'])
        last_command_velocity.copy_(saved['command_velocity'])
        for values in traj.values():del values[saved['length']:]
        return saved['sim'],saved['controls'].clone()
    checkpoints=[];attempts={};fallbacks=0  # recent window-start checkpoints (oldest first); executions per window start

    # --fallback stable: per-step restart points of the recent past.
    side_hands=[{g for g in hands if model.geom(g).name.startswith(f'collision_hand_{side}_')} for side in ('right','left')]
    history=[];near=round(.05/dt);hold=round(args.stable_hold/dt)
    def hand_contacts(q):
        probe.qpos[:]=q;mujoco.mj_forward(model,probe);found=[False,False]
        for con in probe.contact:
            a,b=int(con.geom[0]),int(con.geom[1])
            if con.dist>.001 or not (a in objects or b in objects):continue
            for h in range(2):found[h]|=a in side_hands[h] or b in side_hands[h]
        return found
    def remember(step,servo_state,plan,executed):
        """Restart point after `step`: world-0 state, servo, and the plan whose first `executed` controls are done."""
        q=traj['qpos'][-1];right,left=hand_contacts(q)
        history.append(dict(sim=step,length=len(traj['time']),plan=plan,executed=executed,servo=servo_state,
            data={f:wp.to_torch(getattr(env.data_wp,f))[0].clone() for f in STATE_FIELDS if getattr(env.data_wp,f).size},
            right=right,left=left,sunk=bool(qref[step,-12]-q[-12]>args.sink_tol)))
    def stable_point(upper):
        """Index of the latest step <= upper that ends --stable-hold seconds of stable grasp."""
        def holding(i):
            around=history[max(0,i-near):i+near+1]
            return any(x['right'] for x in around)+any(x['left'] for x in around)>=args.stable_hands
        ok=[holding(i) and not h['sunk'] for i,h in enumerate(history)]
        found=None;length=hold if ok and ok[0] else 0  # the first restart point has no recorded past
        for i,h in enumerate(history):
            length=length+1 if ok[i] else 0
            if i and length>=hold and h['sim']<=upper:found=i
            elif not i and ok[0] and h['sim']<=upper:found=0
        return found
    def restore_step(i):
        h=history[i]
        for f,x in h['data'].items():wp.to_torch(getattr(env.data_wp,f))[:]=x
        servo.q_des,servo.v_des,servo.a_des=(x.clone() for x in h['servo'])
        for values in traj.values():del values[h['length']:]
        last_command_velocity.copy_(torch.tensor((traj['ctrl'][-1]-traj['ctrl'][-2])/dt,device='cuda'))
        del history[i+1:]
        k=h['executed'];step=h['sim']
        return step,(append_continuous_controls(h['plan'][k:],tlatent[step+H-k:step+H]) if k else h['plan'].clone())

    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    sim=0;window=0;contact=False;controls=tlatent[:H].clone();failure=None;started=time.monotonic()
    if resume is None:record(0.,qinit.copy(),np.zeros(model.nv),qinit[:56].copy(),controls[0].cpu().numpy())
    else:
        # Keep the saved prefix so the trajectory still starts at 0 s; warm-start from its next executed controls.
        for k in range(start+1):record(*(resume[key][k].copy() for key in traj))
        sim=start;contact=True
        saved=torch.tensor(resume['latent'][start+1:start+1+H],device='cuda',dtype=torch.float32)
        controls=torch.cat([saved,tlatent[sim+len(saved):sim+H]])
        dump(out/'resume.json',dict(source=str(Path(args.resume_from).resolve()),start_s=start*dt,
            prefix='copied from the source trajectory; physics continues from its saved qpos/qvel/ctrl'))
        if args.fallback=='stable':remember(sim,(servo.q_des.clone(),servo.v_des.clone(),servo.a_des.clone()),controls.clone(),0)
    save();status('running')
    try:
        while sim<limit and not stop[0]:
            window+=1;window_start=time.monotonic();config.warmup_sim_step=sim;log=dict(window=window,time_s=sim*dt,contact=contact)
            if not contact:
                # Planned open-hand approach, executed until the first real contact.
                controls=tlatent[sim:sim+H].clone()
                progress.update('precontact_approach',accepted_time_s=sim*dt,window_start_s=sim*dt,
                    window_end_s=sim*dt+config.ctrl_dt,iteration=0,iteration_completed=0,force=True)
            else:
                if args.fallback=='window' and not (checkpoints and checkpoints[-1]['sim']==sim):
                    checkpoints.append(checkpoint())
                    if len(checkpoints)>args.fallback_depth+1:attempts.pop(checkpoints.pop(0)['sim'],None)
                refs=tuple(x[sim+1:sim+H+1] for x in tensors);iterations=[];window_samples.clear();best.clear()
                retry=attempts.get(sim,0) if args.fallback=='window' else 0;noise=args.retry_noise**retry
                if retry:log.update(retry=retry,noise_scale=noise)
                for it in range(config.max_num_iterations):
                    progress.update('sampling',accepted_time_s=sim*dt,window_start_s=sim*dt,window_end_s=sim*dt+config.horizon,
                        iteration=it+1,iteration_completed=it,force=True)
                    controls,info=optimize_once(config,env,controls,refs,[{}],{'global_noise_scale':noise*config.beta_traj**it})
                    iterations.append(dict(improvement=float(info['improvement']),rew_max=float(info['rew_max']),
                        rew_mean=float(info['rew_mean']),object_pos_error_min=float(info['object_pos_error_min']),
                        **({'dropped_fraction':float(info['dropped_mean'])} if 'dropped_mean' in info else {})))
                    # SPIDER early stopping (improvement_check_steps = 1).
                    if info['improvement']<config.improvement_threshold or stop[0]:break
                    if args.fallback=='window' and it>=2 and all(x.get('dropped_fraction',0)>=1 for x in iterations[-3:]):break  # every rollout drops: no use iterating
                log['iterations']=iterations;save_samples(window)
                if args.execute=='best':
                    controls=best['latent']
                    log['executed']=dict(score=best['score'],iteration=best['iteration'],sample=best['sample'],
                        last_mean_score=iterations[-1]['rew_max']-iterations[-1]['improvement'])  # sample 0 = previous mean
            # Execute the chosen control for ctrl_dt on the real state.
            progress.update('executing',force=True)
            mapped=project(controls);n=min(config.ctrl_steps,limit-sim)
            for k in range(n):
                u=actuator_control(env,mapped[k])
                scheduler.step(config,env,u,perturb=False,sim_step=sim+k+1)
                q=physics.get_qpos(config,env)[0].cpu().numpy().copy();v=physics.get_qvel(config,env)[0].cpu().numpy().copy()
                record((sim+k+1)*dt,q,v,u[0].cpu().numpy().copy(),controls[k].cpu().numpy().copy())
                if args.fallback=='stable' and 'iterations' in log:
                    # Servo state as if the window had ended after this step (cf. accept below).
                    old=servo.v_des if k==0 else mapped[k-1,56:68]
                    remember(sim+k+1,(mapped[k][servo.ids].clone(),mapped[k,56:68].clone(),(mapped[k,56:68]-old)/dt),controls,k+1)
                if not contact and touching(q):
                    contact=True;n=k+1;log['first_contact_s']=(sim+n)*dt
                    print(f'PRECONTACT_END {(sim+n)*dt:.3f}s; start SPIDER MPC',flush=True);break
            old_v=servo.v_des.clone() if n==1 else mapped[n-2,56:68]
            servo.accept(mapped[n-1]);servo.a_des=(mapped[n-1,56:68]-old_v)/dt
            last_command_velocity.copy_(torch.tensor((traj['ctrl'][-1]-traj['ctrl'][-2])/dt,device='cuda'))
            sim+=n
            controls=append_continuous_controls(controls[n:],tlatent[sim+H-n:sim+H])
            error=float(np.linalg.norm(traj['qpos'][-1][-14:-11]-qref[sim,-14:-11]))
            log.update(end_s=sim*dt,object_error_m=error,wall_s=time.monotonic()-window_start)
            if 'iterations' in log and args.fallback=='stable':
                angle,drop=window_violation(np.array(traj['qpos'][-n:]),qref[sim-n+1:sim+1])
                log.update(max_rotation_error_deg=angle,max_drop_m=drop)
                if drop>args.fall_drop:
                    drops=qref[sim-n+1:sim+1,-12]-np.array(traj['qpos'][-n:])[:,-12]
                    first_drop=sim-n+1+int(np.argmax(drops>args.fall_drop));upper=first_drop
                    while True:
                        i=stable_point(upper)
                        if i is None or attempts.get(history[i]['sim'],0)<args.fallback_retries:break
                        upper=history[i]['sim']-1  # this restart point is used up; look further back
                    fallbacks+=1
                    log.update(rejected=True,first_drop_s=first_drop*dt,fallbacks=fallbacks,
                        fallback_to_s=None if i is None else history[i]['sim']*dt)
                    with (out/'optimizer_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                    print(f'FALLBACK {log["time_s"]:.2f}->{sim*dt:.2f}s rotation={angle:.1f}deg drop={drop*1000:.0f}mm '
                          +(f'-> last stable {log["fallback_to_s"]:.2f}s' if i is not None else '-> no stable point left')+f' (fallback {fallbacks})',flush=True)
                    if i is None or fallbacks>args.fallback_max:
                        failure=f'fallback exhausted at {log["time_s"]:.2f}s'
                        if i is not None:sim,controls=restore_step(i)  # else keep the drop in the saved trajectory
                        break
                    attempts[history[i]['sim']]=attempts.get(history[i]['sim'],0)+1
                    sim,controls=restore_step(i);continue
                # Keep only the recent past as restart points.
                while history and history[0]['sim']<sim-round(args.stable_lookback/dt):history.pop(0)
            if 'iterations' in log and args.fallback=='window':
                angle,drop=window_violation(np.array(traj['qpos'][-n:]),qref[sim-n+1:sim+1])
                log.update(max_rotation_error_deg=angle,max_drop_m=drop)
                attempts[sim-n]=attempts.get(sim-n,0)+1
                if drop>args.fall_drop:
                    # Dropped: redo this window, or step back once its retries are used up.
                    dropped=None
                    while checkpoints and attempts.get(checkpoints[-1]['sim'],0)>args.fallback_retries:
                        dropped=checkpoints.pop();attempts.pop(dropped['sim'],None)
                    fallbacks+=1
                    log.update(rejected=True,fallback_to_s=checkpoints[-1]['sim']*dt if checkpoints else None,fallbacks=fallbacks)
                    with (out/'optimizer_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                    print(f'FALLBACK {log["time_s"]:.2f}->{sim*dt:.2f}s rotation={angle:.1f}deg drop={drop*1000:.0f}mm '
                          +(f'-> redo from {log["fallback_to_s"]:.2f}s' if checkpoints else '-> no retries left')+f' (fallback {fallbacks})',flush=True)
                    if not checkpoints or fallbacks>args.fallback_max:
                        # Give up; keep the trajectory up to the last accepted state.
                        failure=f'fallback exhausted at {log["time_s"]:.2f}s';sim,controls=restore(checkpoints[-1] if checkpoints else dropped);break
                    sim,controls=restore(checkpoints[-1]);continue
            with (out/'optimizer_log.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(f'WINDOW {log["time_s"]:.2f}->{sim*dt:.2f}s iters={len(log.get("iterations",[]))} '
                  f'object_error={error*1000:.1f}mm wall={log["wall_s"]:.1f}s'
                  +(f' rotation={log["max_rotation_error_deg"]:.1f}deg' if 'max_rotation_error_deg' in log else '')
                  +(f' dropped%={100*log["iterations"][-1]["dropped_fraction"]:.0f}' if log.get('iterations') and 'dropped_fraction' in log['iterations'][-1] else '')
                  +(f' retry={log["retry"]} noise=x{log["noise_scale"]:.2f}' if 'retry' in log else '')
                  +(f' executed=best(it{log["executed"]["iteration"]+1},{log["executed"]["score"]:.3f}) mean={log["executed"]["last_mean_score"]:.3f}' if 'executed' in log else ''),flush=True)
            save();status('running');progress.update('accepted',accepted_time_s=sim*dt,force=True)
    except Exception as ex:
        failure=f'{type(ex).__name__}: {ex}';traceback.print_exc()
    finally:
        save()
        state='failed' if failure else ('completed' if sim>=limit else 'stopped')
        status(state);progress.update('completed' if state=='completed' else state,accepted_time_s=sim*dt,force=True)
        result=dict(state=state,failure=failure,time_s=sim*dt,wall_s=time.monotonic()-started,arguments=vars(args))
        dump(out/'result.json',result);print(json.dumps(result),flush=True)
    if failure:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name',default='spider_arm_0001')
    p.add_argument('--plan-result',default='sqp_result_001.json')
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--improvement-threshold',type=float,default=.01)
    p.add_argument('--horizon',type=float,default=1.6)
    p.add_argument('--ctrl-dt',type=float,default=.4)
    p.add_argument('--knot-dt',type=float,default=.4)
    p.add_argument('--sim-dt',type=float,default=.01)
    p.add_argument('--temperature',type=float,default=.3)
    p.add_argument('--end',type=float,default=24.5)
    p.add_argument('--seed',type=int,default=63575)
    p.add_argument('--resume-from',help='trajectory.npz of an earlier run to continue from (needs --start)')
    p.add_argument('--start',type=float,default=0.,help='with --resume-from: time (s) to continue from')
    p.add_argument('--w-obj-pos',type=float,default=W_OBJ_POS,help='object position weight of the reward (SPIDER: 1)')
    p.add_argument('--obj-weight-after-contact',type=float,
        help='object position weight from the human\'s first contact (schedule.json) on (default: --w-obj-pos)')
    p.add_argument('--w-obj-rot',type=float,default=W_OBJ_ROT,help='object rotation weight of the reward, per radian (SPIDER: 0.3)')
    p.add_argument('--execute',choices=('best','mean'),default='mean',
        help='mean: SPIDER weighted mean; best: highest-scoring rollout of the window')
    p.add_argument('--fallback',choices=('none','stable','window'),default='none',
        help='when the object falls: stable = restart from the latest stable-grasp moment; window = redo the 0.4 s window with drop-aware sampling')
    p.add_argument('--stable-hands',type=int,choices=(1,2),default=1,help='stable: hands that must touch the object')
    p.add_argument('--sink-tol',type=float,default=.01,help='stable: object at most this far (m) below the reference')
    p.add_argument('--stable-hold',type=float,default=.1,help='stable: grasp must have been stable this long (s) before a restart point')
    p.add_argument('--stable-lookback',type=float,default=3.,help='stable: keep restart points from this many seconds back')
    p.add_argument('--fall-drop',type=float,default=.05,help='reject a window whose object drops this far below the reference (m)')
    p.add_argument('--retry-noise',type=float,default=1.5,help='sampling noise multiplier per retry of a rejected window')
    p.add_argument('--fallback-retries',type=int,default=2,help='restarts from the same point (window: re-optimizations) before going further back')
    p.add_argument('--fallback-depth',type=int,default=2,help='how many earlier windows may be redone')
    p.add_argument('--fallback-max',type=int,default=30,help='give up after this many fallbacks in total')
    run(p.parse_args())
