"""Bound short, controlled contact impulses; sustained angular overspeed fails."""
import numpy as np
MAX_EXTRA_RAD_S=2.
MAX_EXTRA_ANGLE=np.deg2rad(.75)
MAX_HIGH_FRAMES=2
class AngularBudget:
    def __init__(self,dt=1/150):self.dt=dt;self.high_frames=0;self.excess_angle=0.;self.peak=0.;self.failed=False
    def step(self,omega,limit,controlled,is_end=False):
        high=omega>limit;self.high_frames+=int(high);self.excess_angle+=max(0.,omega-limit)*self.dt;self.peak=max(self.peak,omega)
        bad=(not np.isfinite([omega,limit]).all() or omega>limit+MAX_EXTRA_RAD_S or self.high_frames>MAX_HIGH_FRAMES or self.excess_angle>MAX_EXTRA_ANGLE or (high and not controlled) or (is_end and high))
        self.failed |= bool(bad)
        return not self.failed
    def velocity_step(self,v,r,controlled,is_end=False):
        linear=float(np.linalg.norm(v[-12:-9]));angular=float(np.linalg.norm(v[-9:-6]));ll=max(.6,2*float(np.linalg.norm(r[-12:-9]))+.25);al=max(4.,2*float(np.linalg.norm(r[-9:-6]))+2.)
        if not np.isfinite(linear) or linear>ll:return False,f'object speed {linear:.3f} m/s exceeds {ll:.3f} m/s'
        ok=self.step(angular,al,controlled,is_end)
        return ok,None if ok else f'angular impulse outside control budget: peak={self.peak:.3f} rad/s high_frames={self.high_frames} excess_deg={np.rad2deg(self.excess_angle):.3f}'
class AngularBudgetTorch:
    def __init__(self,n,device,dt=1/150):
        import torch
        self.dt=dt;self.high_frames=torch.zeros(n,device=device);self.excess_angle=torch.zeros(n,device=device);self.failed=torch.zeros(n,device=device,dtype=torch.bool)
    def step(self,omega,limit,controlled,is_end=False):
        import torch
        high=omega>limit;self.high_frames+=high;self.excess_angle+=torch.clamp(omega-limit,min=0)*self.dt
        self.failed |= (~torch.isfinite(omega))|(~torch.isfinite(limit))|(omega>limit+MAX_EXTRA_RAD_S)|(self.high_frames>MAX_HIGH_FRAMES)|(self.excess_angle>MAX_EXTRA_ANGLE)|(high&~controlled)|(high if is_end else torch.zeros_like(high))
        return ~self.failed
