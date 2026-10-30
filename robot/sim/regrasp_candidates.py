"""Bounded free-hand candidates; every result still goes through IK and physics."""
import torch
def add_regrasp_candidates(latent,origin,current_ctrl,side_gaps,dt,hold_support=False,elapsed_s=0.,goals=None,free_side=None,support_goals=None,moving_goals=None):
    if origin is None or len(latent)<8:return None
    # Change only when one side is currently missing while the other touches.
    if free_side in (0,1):free=int(free_side)
    elif side_gaps[0]>=3 and side_gaps[1]<=1:free=0
    elif side_gaps[1]>=3 and side_gaps[0]<=1:free=1
    else:return None
    other=1-free;ofs=28*free;otherofs=28*other
    away=(origin[ofs:ofs+3]-origin[otherofs:otherofs+3]).clone()
    away[2]=0
    if torch.linalg.vector_norm(away)<1e-6:
        away=torch.tensor([1. if free==0 else -1.,0.,0.],device=latent.device,dtype=latent.dtype)
    away=away/torch.linalg.vector_norm(away)
    up=torch.tensor([0.,0.,1.],device=latent.device,dtype=latent.dtype)
    offsets=[.020*away+.010*up,.035*away+.025*up,.045*up,.045*away]
    t=torch.arange(latent.shape[1],device=latent.device,dtype=latent.dtype)*dt
    # Persist the handover clock across 0.2 s replans. Otherwise the
    # controller keeps restarting a 0.35 s retreat and never approaches.
    def smooth(x):
        x=x.clamp(0,1);return x*x*(3-2*x)
    time=t+elapsed_s
    away_ramp=smooth(time/.35)-smooth(torch.as_tensor(elapsed_s/.35,device=t.device))
    a0=smooth(torch.as_tensor((elapsed_s-.35)/.65,device=t.device))
    approach_ramp=((smooth((time-.35)/.65)-a0)/(1-a0).clamp(min=1e-6)).clamp(0,1)
    if elapsed_s>=1.:approach_ramp=smooth(t/.3)
    for j,delta in enumerate(offsets):
        row=4+j
        # Preserve the sampled continuation before constructing a collision
        # avoidance waypoint, so the released hand can approach a useful
        # future pose rather than retreating forever.
        future_pos=latent[row,:,ofs:ofs+3].clone()
        future_rot=latent[row,:,ofs+3:ofs+6].clone()
        future_fingers=latent[row,:,ofs+6:ofs+28].clone()
        if goals is not None:
            future_pos=goals[j%2,:,:3]
            future_rot=goals[j%2,:,3:6]
            future_fingers=goals[j%2,:,6:28]
        if hold_support and support_goals is not None and moving_goals is not None and j>=2:
            future_pos=moving_goals[j-2,:,:3]
            future_rot=moving_goals[j-2,:,3:6]
            future_fingers=moving_goals[j-2,:,6:28]
        if hold_support:
            latent[row,:,otherofs:otherofs+6]=origin[otherofs:otherofs+6]
            latent[row,:,otherofs+6:otherofs+28]=current_ctrl[otherofs+6:otherofs+28]
        if hold_support and support_goals is not None and j>=2:
            # Keep the grasp relation while gently turning the supporting
            # wrist along a ghost object trajectory. These are servo targets,
            # never object state edits or assistance forces.
            latent[row,:,otherofs:otherofs+6]=support_goals[j-2]
        waypoint=origin[ofs:ofs+3]+away_ramp[:,None]*delta
        # Two candidates approach the sampled future pose; two continue to
        # clear the other arm, giving the physics search both possibilities.
        alpha=approach_ramp if goals is not None or j in (0,1) else torch.zeros_like(approach_ramp)
        latent[row,:,ofs:ofs+3]=waypoint*(1-alpha[:,None])+future_pos*alpha[:,None]
        rot_delta=torch.atan2(torch.sin(future_rot-origin[ofs+3:ofs+6]),torch.cos(future_rot-origin[ofs+3:ofs+6]))
        latent[row,:,ofs+3:ofs+6]=origin[ofs+3:ofs+6]+alpha[:,None]*rot_delta
        opening=1. if j in (0,3) else .8
        opened=current_ctrl[ofs+6:ofs+28][None,:]*(1-opening*away_ramp[:,None])
        latent[row,:,ofs+6:ofs+28]=opened*(1-alpha[:,None])+future_fingers*alpha[:,None]
        # Do not ask the free wrist to erase a large accumulated pose error
        # in one 0.2 s commit. Trace a bounded path toward the same goal.
        speed=.08 if j==0 else .12
        position=origin[ofs:ofs+3].clone()
        angle=origin[ofs+3:ofs+6].clone()
        for k in range(latent.shape[1]):
            delta=latent[row,k,ofs:ofs+3]-position
            length=torch.linalg.vector_norm(delta)
            position=position+delta*torch.clamp(speed*dt/length.clamp(min=1e-9),max=1.)
            turn=torch.atan2(torch.sin(latent[row,k,ofs+3:ofs+6]-angle),torch.cos(latent[row,k,ofs+3:ofs+6]-angle))
            angle=angle+turn*torch.clamp(.8*dt/torch.linalg.vector_norm(turn).clamp(min=1e-9),max=1.)
            latent[row,k,ofs:ofs+3]=position
            latent[row,k,ofs+3:ofs+6]=angle
    return free

def object_relative_goals(ghost_qpos,reference_qpos,reference_wrist_pos,reference_wrist_rot,nominal,free):
    """Move the raw reference grasp to each reachable ghost object pose.

    The supporting hand is unchanged. This returns targets, never state edits.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    H=len(nominal);ofs=28*free
    ref=reference_qpos[:H];gr=ghost_qpos[:2,:H]
    rref=Rotation.from_quat(ref[:,[-10,-9,-8,-11]])
    local=rref.inv().apply(reference_wrist_pos[:H,free]-ref[:,-14:-11])
    local_R=rref.inv().as_matrix()@reference_wrist_rot[:H,free]
    out=np.repeat(nominal[None,:,ofs:ofs+28],2,axis=0).copy()
    for j in range(2):
        r=Rotation.from_quat(gr[j][:,[-10,-9,-8,-11]])
        out[j,:,:3]=gr[j,:,-14:-11]+r.apply(local)
        angles=Rotation.from_matrix(r.as_matrix()@local_R).as_euler('XYZ')
        original=nominal[:,ofs+3:ofs+6]
        out[j,:,3:6]=original+np.arctan2(np.sin(angles-original),np.cos(angles-original))
    return out

def supporting_turn_goals(current_q,current_wrist,ghost_qpos,dt):
    """Two bounded support trajectories, preserving current object-local grasp."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    q=np.asarray(current_q);w=np.asarray(current_wrist);ghost=np.asarray(ghost_qpos)
    now=Rotation.from_quat(q[[-10,-9,-8,-11]]);wr=Rotation.from_euler('XYZ',w[3:6])
    lp=now.inv().apply(w[:3]-q[-14:-11]);lr=now.inv()*wr
    out=[]
    for j,(speed,angular) in enumerate(((.06,.35),(.10,.60))):
        gr=Rotation.from_quat(ghost[j+1][:,[-10,-9,-8,-11]])
        targets=ghost[j+1,:,-14:-11]+gr.apply(lp);rots=gr*lr
        pos=w[:3].copy();rot=wr;arr=[];last_angle=w[3:6].copy()
        for k in range(len(targets)):
            delta=targets[k]-pos;pos+=delta*min(1.,speed*dt/max(np.linalg.norm(delta),1e-12))
            rv=(rots[k]*rot.inv()).as_rotvec();rv*=min(1.,angular*dt/max(np.linalg.norm(rv),1e-12));rot=Rotation.from_rotvec(rv)*rot
            angle=rot.as_euler('XYZ');last_angle=last_angle+np.arctan2(np.sin(angle-last_angle),np.cos(angle-last_angle))
            arr.append(np.r_[pos,last_angle])
        out.append(arr)
    return np.asarray(out)
