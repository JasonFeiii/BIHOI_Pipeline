"""Geometric pairing of bounded support motion and reference-local free grasp.

Targets only. No simulation state, force, physical checks, or IK is changed.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def paired_free_goals(current_q, support_wrist, support_targets,
                      reference_qpos, reference_wrist_pos, reference_wrist_rot,
                      nominal, free):
    q=np.asarray(current_q); w=np.asarray(support_wrist)
    turns=np.asarray(support_targets)
    if turns.ndim != 3 or turns.shape[2] != 6:
        raise ValueError('Support targets must have shape [modes, horizon, 6]')
    n,h,_=turns.shape; ofs=28*free
    if min(len(reference_qpos),len(reference_wrist_pos),len(reference_wrist_rot),len(nominal))<h:
        raise ValueError('Insufficient reference samples')
    ro=Rotation.from_quat(q[[-10,-9,-8,-11]])
    rs=Rotation.from_euler('XYZ',w[3:6])
    local_p=ro.inv().apply(w[:3]-q[-14:-11])
    local_r=ro.inv()*rs
    ref=np.asarray(reference_qpos)[:h]
    rr=Rotation.from_quat(ref[:,[-10,-9,-8,-11]])
    free_p=rr.inv().apply(np.asarray(reference_wrist_pos)[:h,free]-ref[:,-14:-11])
    free_r=rr.inv().as_matrix()@np.asarray(reference_wrist_rot)[:h,free]
    out=np.repeat(np.asarray(nominal)[None,:h,ofs:ofs+28],n,axis=0).copy()
    objects=np.repeat(np.repeat(q[None,None],n,axis=0),h,axis=1)
    for j in range(n):
        r=Rotation.from_euler('XYZ',turns[j,:,3:6])*local_r.inv()
        p=turns[j,:,:3]-r.apply(local_p)
        objects[j,:,-14:-11]=p
        quat=r.as_quat();objects[j,:,-11:-7]=quat[:,[3,0,1,2]]
        out[j,:,:3]=p+r.apply(free_p)
        angle=Rotation.from_matrix(r.as_matrix()@free_r).as_euler('XYZ')
        anchor=np.asarray(nominal)[:h,ofs+3:ofs+6]
        out[j,:,3:6]=anchor+np.arctan2(np.sin(angle-anchor),np.cos(angle-anchor))
    return out,objects


def paired_wrist_errors(position, rotation, goal_position, goal_rotation, free):
    """Choose one paired mode per step using existing wrist-score weights.

    Hands cannot independently pick contradictory object paths for the score.
    This does not force either hand to touch or change any physical guard.
    """
    import torch
    ep=torch.linalg.vector_norm(position[:,None]-goal_position[None],dim=-1)
    cc=((rotation[:,None].transpose(-1,-2)@goal_rotation[None]).diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
    er=torch.acos(cc.clamp(-1+1e-6,1-1e-6))
    loss=.3*ep.sum(-1)+.04*er.sum(-1)+1.5*ep[:,:,free]+.1*er[:,:,free]
    mode=loss.argmin(dim=1)
    chosen_p=ep.gather(1,mode[:,None,None].expand(-1,1,2)).squeeze(1)
    chosen_r=er.gather(1,mode[:,None,None].expand(-1,1,2)).squeeze(1)
    return chosen_p.sum(-1),chosen_r.sum(-1),chosen_p[:,free],chosen_r[:,free],mode
