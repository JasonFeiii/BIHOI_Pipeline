"""Map sampled wrist targets to bounded arm-joint trajectories before physics."""
import numpy as np
import torch
import mujoco
from arm_kinematics import ArmKinematics
from build_arm_scene import SIDES
from smooth_targets import smooth_targets

def euler_xyz(angles):
    x,y,z=angles.unbind(-1);cx,sx=torch.cos(x),torch.sin(x)
    cy,sy=torch.cos(y),torch.sin(y);cz,sz=torch.cos(z),torch.sin(z)
    return torch.stack([cy*cz,-cy*sz,sy,
        cx*sz+sx*sy*cz,cx*cz-sx*sy*sz,-sx*cy,
        sx*sz-cx*sy*cz,sx*cz+cx*sy*sz,cx*cy],-1).reshape(angles.shape[:-1]+(3,3))

class WristProjector:
    def __init__(self,model,dt,device='cuda',acceleration_limit=20.,jerk_limit=None,tracking_rate=25.):
        self.model=model;self.dt=dt;self.device=device
        self.kin=[ArmKinematics(model,s,device=device) for s in SIDES]
        data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        self.bp=[];self.br=[]
        for side in SIDES:
            bid=model.body(side+'_arm_base').id
            self.bp.append(torch.tensor(data.xpos[bid],device=device,dtype=torch.float32))
            self.br.append(torch.tensor(data.xmat[bid].reshape(3,3),device=device,dtype=torch.float32))
        self.vlimit=torch.tensor([np.pi]*3+[2*np.pi]*3,device=device,dtype=torch.float32)
        self.alimit=acceleration_limit
        self.jlimit=jerk_limit
        self.tracking_rate=tracking_rate
    @torch.no_grad()
    def project(self,latent,reference_arm_q,qstart,vstart=None,iterations=8,astart=None,ik_start=None):
        single=latent.ndim==2
        if single:latent=latent[None]
        N,H,_=latent.shape
        self.last_seed_stats=[]
        result=latent.clone();errors=[]
        if qstart.ndim==1:qstart=qstart[None].expand(N,-1)
        if ik_start is None:ik_start=qstart
        elif ik_start.ndim==1:ik_start=ik_start[None].expand(N,-1)
        if vstart is None:vstart=torch.zeros_like(qstart)
        elif vstart.ndim==1:vstart=vstart[None].expand(N,-1)
        for side,k in enumerate(self.kin):
            offset=side*28;ix=slice(offset,offset+6)
            target_p=(latent[...,offset:offset+3]-self.bp[side])@self.br[side]
            target_R=self.br[side].T@euler_xyz(latent[...,offset+3:offset+6])
            seed=reference_arm_q[...,ix]
            if seed.ndim==2:seed=seed[None].expand(N,-1,-1)
            actual=ik_start[...,ix][:,None,:].expand(N,H,6)
            both=torch.cat([seed.reshape(-1,6),actual.reshape(-1,6)],dim=0)
            targets_p=target_p.reshape(-1,3).repeat(2,1)
            targets_R=target_R.reshape(-1,3,3).repeat(2,1,1)
            candidates,_,_=k.ik(targets_p,targets_R,both,iterations=iterations,step_cap=.2)
            candidates=candidates.reshape(2,N,H,6)
            candidates=actual[None]+torch.atan2(torch.sin(candidates-actual[None]),torch.cos(candidates-actual[None]))
            candidates=candidates.clamp(k.lower,k.upper)
            pp,rr=k.fk(candidates)
            pos=torch.linalg.vector_norm(pp-target_p[None],dim=-1)
            cosine=((rr.transpose(-1,-2)@target_R[None]).diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
            ang=torch.acos(cosine.clamp(-1+1e-7,1-1e-7))
            continuity=torch.linalg.vector_norm(candidates-actual[None],dim=-1)
            cost=pos+.05*ang+.001*continuity
            use_actual=cost[1]<cost[0]
            q=torch.where(use_actual[...,None],candidates[1],candidates[0])
            self.last_seed_stats.append(dict(side=side,actual_seed_fraction=float(use_actual.float().mean()),reference_seed_max_position_m=float(pos[0].max()),chosen_max_position_m=float(torch.where(use_actual,pos[1],pos[0]).max())))
            if self.jlimit is not None:
                a0=None if astart is None else astart[...,side*6:side*6+6]
                q=smooth_targets(q,qstart[...,ix],vstart[...,ix],self.dt,self.vlimit,
                                 self.alimit,acceleration0=a0,jerk_limit=self.jlimit,tracking_rate=self.tracking_rate)
                q=q.clamp(k.lower,k.upper)
            else:
                prev=qstart[...,ix];velocity=vstart[...,ix];bounded=[]
                for t in range(H):
                    desired=((q[:,t]-prev)/self.dt).clamp(-self.vlimit,self.vlimit)
                    if self.alimit is not None:
                        desired=torch.maximum(torch.minimum(desired,velocity+self.alimit*self.dt),velocity-self.alimit*self.dt)
                    next_q=(prev+desired*self.dt).clamp(k.lower,k.upper)
                    velocity=(next_q-prev)/self.dt;prev=next_q;bounded.append(next_q)
                q=torch.stack(bounded,dim=1)
            result[...,ix]=q
            achieved_p,achieved_R=k.fk(q)
            errors.append(torch.linalg.vector_norm(achieved_p-target_p,dim=-1))
        return (result[0] if single else result),torch.stack(errors,-1)

    def wrists(self,qpos,jacobian=False):
        ps=[];Rs=[];Js=[]
        for side,k in enumerate(self.kin):
            result=k.fk(qpos[...,side*28:side*28+6],jacobian)
            p,R=result[:2]
            ps.append(p@self.br[side].T+self.bp[side]);Rs.append(self.br[side]@R)
            if jacobian:
                J=result[2].clone()
                J[...,:3,:]=self.br[side]@J[...,:3,:]
                J[...,3:,:]=self.br[side]@J[...,3:,:]
                Js.append(J)
        if jacobian:return torch.stack(ps,-2),torch.stack(Rs,-3),torch.stack(Js,-3)
        return torch.stack(ps,-2),torch.stack(Rs,-3)
