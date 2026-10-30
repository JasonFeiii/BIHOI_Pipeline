from pathlib import Path
import json
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from build_arm_scene import ROOT,OLD,TASK,SIDES

def derivatives(p,R,dt):
    v=np.gradient(p,dt,axis=0,edge_order=2)
    a=np.gradient(v,dt,axis=0,edge_order=2)
    interval=Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).as_rotvec()/dt
    w=np.r_[interval[:1],(interval[:-1]+interval[1:])/2,interval[-1:]]
    alpha=np.gradient(w,dt,axis=0,edge_order=2)
    return v,a,w,alpha

def main(exclude_last_seconds=0.):
    source=OLD/'sharpa/bimanual'/TASK/'0'
    raw=np.load(source/'trajectory_kinematic.npz')['qpos']
    model=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));data=mujoco.MjData(model)
    schedule=json.loads((OLD/'trial_accepted_v2/schedule.json').read_text())
    start=schedule['anchor_source_frame'];stop=len(raw)-round(exclude_last_seconds*30)
    if stop-start<3:raise ValueError('Exclusion removes the stable-grasp clip')
    q=raw[start:stop]
    pos=np.zeros((len(q),2,3));rot=np.zeros((len(q),2,3,3))
    for i,row in enumerate(q):
        data.qpos[:]=row;mujoco.mj_kinematics(model,data)
        for s,side in enumerate(SIDES):
            bid=model.body(side+'_hand_C_MC').id
            pos[i,s]=data.xpos[bid];rot[i,s]=data.xmat[bid].reshape(3,3)
    vel=np.empty_like(pos);acc=np.empty_like(pos);angvel=np.empty_like(pos);angacc=np.empty_like(pos)
    for s in range(2):vel[:,s],acc[:,s],angvel[:,s],angacc[:,s]=derivatives(pos[:,s],rot[:,s],1/30)
    forward=rot[:,:,:,2].mean(axis=(0,1));forward[2]=0;forward/=np.linalg.norm(forward)
    ROOT.mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'reference_30hz.npz',qpos=q,wrist_pos=pos,wrist_rot=rot,
        wrist_vel=vel,wrist_acc=acc,wrist_angvel=angvel,wrist_angacc=angacc,
        time=np.arange(len(q))/30,source_frame=np.arange(start,stop))
    info={'task':TASK,'hand_order':list(SIDES),'source_start_frame':start,'source_end_frame':stop-1,
        'frame_count':len(q),'duration_s':(len(q)-1)/30,'frequency':30,
        'selection':'geometric two-hand contact reference anchor through the requested cutoff; physical stability must be verified by simulation',
        'excluded_tail_seconds':exclude_last_seconds,'original_source_frame_count':len(raw),
        'source_poses_modified':False,'mean_hand_forward':forward.tolist(),
        'mean_wrist_position':pos.mean(axis=(0,1)).tolist(),
        'derivative_norm_max':{key:float(np.linalg.norm(value,axis=-1).max()) for key,value in
            [('velocity_m_s',vel),('acceleration_m_s2',acc),('angular_velocity_rad_s',angvel),('angular_acceleration_rad_s2',angacc)]}}
    (ROOT/'reference_info.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exclude-last-seconds',type=float,default=0.)
    main(p.parse_args().exclude_last_seconds)
