"""Soft, frame-consistent angular-velocity guidance; never a hard guard."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
def ghost_angular_velocity_world(ghosts,dt):
    ghosts=np.asarray(ghosts,dtype=float);result=[]
    for row in ghosts:
        R=Rotation.from_quat(row[:,[-10,-9,-8,-11]])
        result.append((R[1:]*R[:-1].inv()).as_rotvec()/dt)
    return np.stack(result)
def rotate_body_to_world(quaternion,vector):
    quat=quaternion/torch.linalg.vector_norm(quaternion,dim=-1,keepdim=True).clamp_min(1e-12)
    s=quat[...,:1];u=quat[...,1:]
    t=2*torch.linalg.cross(u,vector,dim=-1)
    return vector+s*t+torch.linalg.cross(u,t,dim=-1)
def angular_tracking_penalty(q,body_omega,target_world,weight=.25):
    actual=rotate_body_to_world(q,body_omega)
    diff=torch.linalg.vector_norm(actual-target_world,dim=-1)
    denominator=torch.linalg.vector_norm(target_world,dim=-1).clamp_min(.35)
    return weight*((diff/denominator)**2).clamp(max=9.)
