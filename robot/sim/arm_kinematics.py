"""Batched UR3e FK/Jacobian and local IK, derived from the compiled MuJoCo model."""
import numpy as np
import mujoco
import torch

def skew(v):
    z=torch.zeros_like(v[...,0]);x,y,w=v.unbind(-1)
    return torch.stack([z,-w,y,w,z,-x,-y,x,z],-1).reshape(v.shape[:-1]+(3,3))

def so3_exp(v):
    angle=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    K=skew(v)
    a=torch.sinc(angle/torch.pi)[...,None]
    b=(.5*torch.sinc(angle/(2*torch.pi))**2)[...,None]
    return torch.eye(3,device=v.device,dtype=v.dtype)+a*K+b*(K@K)

def so3_log(R):
    # atan2 form avoids acos derivatives at identity and is stable for local IK.
    v=torch.stack([R[...,2,1]-R[...,1,2],R[...,0,2]-R[...,2,0],R[...,1,0]-R[...,0,1]],-1)*.5
    s=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    c=((R.diagonal(dim1=-2,dim2=-1).sum(-1,keepdim=True)-1)*.5).clamp(-1,1)
    angle=torch.atan2(s,c)
    scale=angle/torch.clamp(s,min=1e-7)
    # Near pi choose the dominant diagonal axis; ordinary local steps stay away.
    near_pi=(c<-.9999)&(s<1e-4)
    if bool(near_pi.any()):
        diag=((R.diagonal(dim1=-2,dim2=-1)+1)*.5).clamp_min(0).sqrt()
        signs=torch.sign(v+1e-9)
        v=torch.where(near_pi,diag*signs,v)
        scale=torch.where(near_pi,angle/torch.linalg.vector_norm(v,dim=-1,keepdim=True).clamp_min(1e-7),scale)
    return v*scale

class ArmKinematics:
    def __init__(self,model,side,device='cuda',dtype=torch.float32):
        self.device=device;self.dtype=dtype;self.side=side
        data=mujoco.MjData(model);data.qpos[:]=0;mujoco.mj_kinematics(model,data)
        base=model.body(side+'_arm_base').id
        B=data.xmat[base].reshape(3,3);p=data.xpos[base]
        joints=[model.joint(side+'_arm_'+name).id for name in
                ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']]
        self.qadr=np.array([model.jnt_qposadr[j] for j in joints])
        self.axes=torch.tensor(np.array([B.T@data.xaxis[j] for j in joints]),device=device,dtype=dtype)
        self.pivots=torch.tensor(np.array([B.T@(data.xanchor[j]-p) for j in joints]),device=device,dtype=dtype)
        hand=model.body(side+'_hand_C_MC').id
        self.MR=torch.tensor(B.T@data.xmat[hand].reshape(3,3),device=device,dtype=dtype)
        self.Mp=torch.tensor(B.T@(data.xpos[hand]-p),device=device,dtype=dtype)
        self.lower=torch.tensor(model.jnt_range[joints,0],device=device,dtype=dtype)
        self.upper=torch.tensor(model.jnt_range[joints,1],device=device,dtype=dtype)
        self.lower[-1]=-100*torch.pi;self.upper[-1]=100*torch.pi
        elbow=model.body(side+'_arm_forearm_link').id
        self.elbow0=torch.tensor(B.T@(data.xpos[elbow]-p),device=device,dtype=dtype)
    def fk(self,q,jacobian=False,elbow=False):
        shape=q.shape[:-1]
        R=torch.eye(3,device=q.device,dtype=q.dtype).expand(*shape,3,3)
        p=torch.zeros(*shape,3,device=q.device,dtype=q.dtype)
        axes=[];pivots=[];elbow_p=None
        for i in range(6):
            if jacobian:
                axes.append((R@self.axes[i][...,None]).squeeze(-1))
                pivots.append((R@self.pivots[i][...,None]).squeeze(-1)+p)
            Ri=so3_exp(q[...,i,None]*self.axes[i])
            ti=self.pivots[i]-(Ri@self.pivots[i][...,None]).squeeze(-1)
            p=p+(R@ti[...,None]).squeeze(-1)
            R=R@Ri
            if i==2 and elbow:elbow_p=(R@self.elbow0[...,None]).squeeze(-1)+p
        result_p=(R@self.Mp[...,None]).squeeze(-1)+p
        result_R=R@self.MR
        if jacobian:
            w=torch.stack(axes,-1)
            a=torch.stack(pivots,-2)
            linear=torch.linalg.cross(w.transpose(-1,-2),result_p[...,None,:]-a,dim=-1).transpose(-1,-2)
            return result_p,result_R,torch.cat([linear,w],-2)
        if elbow:return result_p,result_R,elbow_p
        return result_p,result_R
    def ik(self,target_p,target_R,seed,iterations=30,damping=.002,step_cap=.25,posture=0.,bounds=None):
        q=seed.clone()
        eye=torch.eye(6,device=q.device,dtype=q.dtype)
        for _ in range(iterations):
            p,R,J=self.fk(q,True)
            e=torch.cat([target_p-p,so3_log(target_R@R.transpose(-1,-2))],-1)
            # Row scaling gives 1 rad orientation error the weight of 0.2 m.
            scale=torch.tensor([1,1,1,.2,.2,.2],device=q.device,dtype=q.dtype)
            Jw=J*scale[:,None];ew=e*scale
            A=Jw.transpose(-1,-2)@Jw+damping*damping*eye
            b=(Jw.transpose(-1,-2)@ew[...,None]).squeeze(-1)
            if posture:b=b+posture*(seed-q);A=A+posture*eye
            dq=torch.linalg.solve(A,b)
            factor=(step_cap/dq.abs().amax(-1,keepdim=True).clamp_min(step_cap))
            lower,upper=(self.lower,self.upper) if bounds is None else bounds
            q=(q+dq*factor).clamp(lower,upper)
        p,R=self.fk(q)
        return q,torch.linalg.vector_norm(target_p-p,dim=-1),torch.linalg.vector_norm(so3_log(target_R@R.transpose(-1,-2)),dim=-1)
