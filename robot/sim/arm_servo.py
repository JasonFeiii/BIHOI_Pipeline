"""Separate bounded IK trajectories from position-actuator control inputs."""
import numpy as np
import torch

class ArmTrajectoryServo:
    def __init__(self,model,dt,device='cuda'):
        self.dt=dt
        self.ids=torch.tensor(np.r_[0:6,28:34],device=device,dtype=torch.long)
        self.kp=torch.tensor(model.actuator_gainprm[np.r_[0:6,28:34],0],device=device,dtype=torch.float32)
        self.kd=torch.tensor(-model.actuator_biasprm[np.r_[0:6,28:34],2],device=device,dtype=torch.float32)
        self.q_des=None;self.v_des=None;self.a_des=None

    def reset(self,actual_q,actual_v):
        self.q_des=actual_q[self.ids].clone()
        self.v_des=actual_v[self.ids].clone()
        self.a_des=torch.zeros_like(self.v_des)

    def boundary(self,actual_q,actual_v):
        q=actual_q.clone();v=actual_v.clone()
        q[self.ids]=self.q_des;v[self.ids]=self.v_des
        return q,v

    def pack(self,desired_controls):
        """Append arm desired velocities to the 56 position/finger targets."""
        q=desired_controls[...,self.ids]
        first=self.q_des.expand(*q.shape[:-2],1,12)
        v=torch.diff(q,dim=-2,prepend=first)/self.dt
        return torch.cat([desired_controls,v],dim=-1)

    def control(self,action,bias_force):
        """tau = Kp(q_des-q)+Kd(v_des-v)+bias, with no object actuation."""
        target=action[...,:56]
        if target.ndim==1 and bias_force.ndim==2:
            target=target.expand(bias_force.shape[0],-1)
        control=target.clone()
        correction=(self.kd.to(action)*action[...,56:68]+bias_force[...,self.ids])/self.kp.to(action)
        control[...,self.ids]+=correction
        return control

    def accept(self,action):
        self.a_des=(action[56:68]-self.v_des)/self.dt
        self.q_des=action[self.ids].clone()
        self.v_des=action[56:68].clone()
