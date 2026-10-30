"""Collision-aware, temporally continuous simultaneous IK for both UR3e arms."""
import argparse
import json
import time
import numpy as np
import mujoco
import mink
from mink.limits.limit import Constraint,Limit
from scipy.spatial.transform import Rotation,Slerp
from build_arm_scene import ROOT,build_scene,SIDES
from search_placement import trajectory_metrics,VELOCITY_LIMIT,evaluate_collisions

class AccelerationLimit(Limit):
    def __init__(self,model,indices,acceleration):
        self.P=np.eye(model.nv)[indices]
        self.velocity=np.zeros(len(indices));self.acceleration=acceleration
    def compute_qp_inequalities(self,configuration,dt):
        del configuration
        return Constraint(G=np.r_[self.P,-self.P],h=np.r_[
            (self.velocity+self.acceleration*dt)*dt,
            (-self.velocity+self.acceleration*dt)*dt])

class FreezeLimit(Limit):
    def __init__(self,model,indices):self.P=np.eye(model.nv)[indices]
    def compute_qp_inequalities(self,configuration,dt):
        return Constraint(G=np.r_[self.P,-self.P],h=np.zeros(2*len(self.P)))

def solve_candidate(c,number,args):
    model,_=build_scene(ROOT/f'mink_scene_{number:03d}.xml',c['center_xy'],c['heading_rad'],base_height=c['height_m'])
    reference=dict(np.load(ROOT/'reference_30hz.npz'));T=len(reference['time'])
    if args.frames:T=min(T,args.frames)
    joints=[model.joint(s+'_arm_'+j).id for s in SIDES for j in
        ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']]
    qadr=model.jnt_qposadr[joints];dofs=model.jnt_dofadr[joints]
    arm_set=set(map(int,dofs))
    for j in range(model.njnt):
        if int(model.jnt_dofadr[j]) not in arm_set:model.jnt_limited[j]=False
    seed_path=ROOT/c['trajectory_file']
    seed=np.load(seed_path)['qpos'][0].copy()
    seed[-7:]=[0,0,-2,1,0,0,0]
    cfg=mink.Configuration(model,q=seed)
    tasks=[mink.FrameTask(frame_name=s+'_hand_C_MC',frame_type='body',position_cost=1.,orientation_cost=.3,gain=.8,lm_damping=1e-4) for s in SIDES]
    posture=mink.PostureTask(model,cost=1e-5);posture.set_target(seed)
    freeze=FreezeLimit(model,[d for d in range(model.nv) if d not in arm_set])
    limits=[freeze,mink.ConfigurationLimit(model,gain=.9),mink.VelocityLimit(model,{model.joint(j).name:float(VELOCITY_LIMIT[i%6]) for i,j in enumerate(joints)})]
    arm_geoms={s:[g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith(s+'_arm_') and model.geom_contype[g]] for s in SIDES}
    env=[g for g in range(model.ngeom) if model.geom_bodyid[g]==0 and model.geom_contype[g]]
    object_geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('right_object').id]
    palms={s:[model.geom('collision_hand_'+s+'_palm_0').id] for s in SIDES}
    pairs=[(arm_geoms['right'],arm_geoms['right']),(arm_geoms['left'],arm_geoms['left']),
           (arm_geoms['right'],arm_geoms['left'])]
    for s in SIDES:
        pairs.extend([(arm_geoms[s],env),(arm_geoms[s],object_geoms),
                      (arm_geoms[s],palms['left' if s=='right' else 'right'])])
    collision=mink.CollisionAvoidanceLimit(model,pairs,gain=args.collision_gain,
        minimum_distance_from_collisions=.005,collision_detection_distance=.08)
    limits.append(collision)
    acceleration=AccelerationLimit(model,dofs,args.acceleration)
    def set_targets(pos,rot,fingers,obj):
        for s,task in enumerate(tasks):
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot[s]),pos[s]))
            cfg.data.qpos[s*28+6:s*28+28]=fingers[s]
        cfg.data.qpos[-14:-7]=obj;cfg.data.qpos[-7:]=[0,0,-2,1,0,0,0]
        cfg.update()
    fingers=np.stack([reference['qpos'][:,6:28],reference['qpos'][:,34:56]],1)
    # Settle the IK configuration before starting the temporal sequence.
    set_targets(reference['wrist_pos'][0],reference['wrist_rot'][0],fingers[0],reference['qpos'][0,-14:-7])
    initial_failures=0
    for _ in range(80):
        try:velocity=mink.solve_ik(cfg,tasks+[posture],.05,'daqp',limits=limits)
        except mink.exceptions.NoSolutionFound:initial_failures+=1;break
        velocity[[d for d in range(model.nv) if d not in arm_set]]=0
        cfg.integrate_inplace(velocity,.05)
    acceleration.velocity[:]=0
    limits.append(acceleration)
    qout=[cfg.data.qpos.copy()];pout=[];Rout=[];elbow=[];failures=0
    def record():
        pout.append(np.stack([cfg.data.xpos[model.body(s+'_hand_C_MC').id].copy() for s in SIDES]))
        Rout.append(np.stack([cfg.data.xmat[model.body(s+'_hand_C_MC').id].reshape(3,3).copy() for s in SIDES]))
        elbow.append(np.stack([cfg.data.xpos[model.body(s+'_arm_forearm_link').id].copy() for s in SIDES]))
    record()
    dt=1/30/args.substeps
    for t in range(1,T):
        for sub in range(1,args.substeps+1):
            f=sub/args.substeps
            pos=(1-f)*reference['wrist_pos'][t-1]+f*reference['wrist_pos'][t]
            rots=[]
            for s in range(2):
                R0=reference['wrist_rot'][t-1,s];R1=reference['wrist_rot'][t,s]
                delta=Rotation.from_matrix(R1@R0.T).as_rotvec()
                rots.append(Rotation.from_rotvec(f*delta).as_matrix()@R0)
            obj=(1-f)*reference['qpos'][t-1,-14:-7]+f*reference['qpos'][t,-14:-7]
            obj[3:]/=np.linalg.norm(obj[3:])
            set_targets(pos,np.asarray(rots),(1-f)*fingers[t-1]+f*fingers[t],obj)
            try:
                velocity=mink.solve_ik(cfg,tasks+[posture],dt,'daqp',limits=limits)
                velocity[[d for d in range(model.nv) if d not in arm_set]]=0
                acceleration.velocity=velocity[dofs].copy()
                cfg.integrate_inplace(velocity,dt)
            except mink.exceptions.NoSolutionFound:
                failures+=1
                # Retain the last configuration; never drop a collision limit.
                acceleration.velocity[:]=0
        qout.append(cfg.data.qpos.copy());record()
    qout=np.asarray(qout);pout=np.asarray(pout);Rout=np.asarray(Rout);elbow=np.asarray(elbow)
    metrics={}
    for s,side in enumerate(SIDES):
        target={key:reference[name][:T,s] for key,name in [('p','wrist_pos'),('R','wrist_rot'),('v','wrist_vel'),('a','wrist_acc'),('w','wrist_angvel'),('alpha','wrist_angacc')]}
        metrics[side]=trajectory_metrics(pout[:,s],Rout[:,s],target,1/30)
    arms=set(arm_geoms['right']+arm_geoms['left'])
    collision_result=evaluate_collisions(model,qout,arms)
    elbow_margin=np.einsum('tsi,tsi->ts',pout-elbow,reference['wrist_rot'][:T,:,:,2]).min(axis=0)
    v=np.diff(qout[:,qadr],axis=0)*30
    ratios=np.max(abs(v)/np.tile(VELOCITY_LIMIT,2),axis=0)
    cost=sum((m['position_rmse_m']/.01)**2+(m['rotation_rmse_rad']/.05)**2+
        .1*(m['velocity_rmse_m_s']/.05)**2+.05*m['acceleration_rmse_m_s2']**2+
        .1*(m['angular_velocity_rmse_rad_s']/.25)**2+.05*(m['angular_acceleration_rmse_rad_s2']/5)**2 for m in metrics.values())
    result={k:c[k] for k in ['center_xy','heading_rad','height_m']}
    result.update({'metrics':metrics,'cost':float(cost),'qp_failures':failures,'initial_qp_failures':initial_failures,
        'collision':collision_result,'elbow_behind_margin_m':elbow_margin.tolist(),
        'joint_velocity_ratio_max':float(ratios.max()),'frame_count':T,'substeps':args.substeps,
        'trajectory_file':f'mink_trajectory_{number:03d}.npz'})
    np.savez_compressed(ROOT/result['trajectory_file'],qpos=qout,wrist_pos=pout,wrist_rot=Rout,time=reference['time'][:T])
    (ROOT/f'mink_result_{number:03d}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,default=0)
    p.add_argument('--frames',type=int,default=0);p.add_argument('--substeps',type=int,default=2)
    p.add_argument('--acceleration',type=float,default=20.)
    p.add_argument('--collision-gain',type=float,default=.2)
    a=p.parse_args()
    source=ROOT/'continuous_results.json'
    if not source.exists():source=ROOT/'placement_results.json'
    candidates=json.loads(source.read_text())
    solve_candidate(candidates[a.candidate],a.candidate,a)
