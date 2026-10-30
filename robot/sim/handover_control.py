"""Regrasp safety state. Explicit proxy checks do not guarantee stable grasp."""
from dataclasses import dataclass,field
import numpy as np

@dataclass
class HandoverState:
    support:int
    anchor_local:np.ndarray
    contact_gap:int=0
    weak_force_gap:int=0
    table_frames:int=0
    elapsed_frames:int=0
    other_contact_frames:int=0
    released:bool=True
    release_gap:int=0
    recent_local:list=field(default_factory=lambda:[[],[]])
    transfers:int=0
    role_elapsed_frames:int=0

    def copy(self):
        return HandoverState(self.support,self.anchor_local.copy(),self.contact_gap,self.weak_force_gap,self.table_frames,self.elapsed_frames,self.other_contact_frames,self.released,self.release_gap,[[p.copy() for p in side] for side in self.recent_local],self.transfers,self.role_elapsed_frames)

    def observe_step(self,hits,normal,relative_both,bottom,vertical_speed,table_contact,dt=1/150,check_force=True):
        # A role can transfer only to a hand already controlling the object:
        # 0.1 seconds of contact and bounded object-local drift, not a new touch.
        for side in range(2):
            valid=hits[side] and (not check_force or normal[side]>=.02)
            if valid:
                self.recent_local[side].append(np.asarray(relative_both[side]).copy())
                self.recent_local[side]=self.recent_local[side][-15:]
            else:self.recent_local[side]=[]
        old=self.support;other=1-old
        old_ok=hits[old] and (not check_force or normal[old]>=.02)
        h=self.recent_local[other]
        qualified=len(h)>=15 and np.max(np.linalg.norm(np.asarray(h)-h[0],axis=1))<=.02
        if not old_ok and qualified:
            self.support=other;self.anchor_local=np.asarray(relative_both[other]).copy()
            self.contact_gap=0;self.weak_force_gap=0;self.other_contact_frames=0
            self.released=True;self.release_gap=1;self.transfers+=1;self.role_elapsed_frames=0
            # elapsed_frames and table_frames deliberately persist across swaps.
        hs=self.support;self.role_elapsed_frames+=1
        return self.step(hits[hs],normal[hs],relative_both[hs],bottom,vertical_speed,table_contact,hits[1-hs] and (not check_force or normal[1-hs]>=.02),dt=dt,check_force=check_force)

    def step(self,support_contact,support_normal_force,relative_position,bottom,vertical_speed,table_contact,other_contact,dt=1/150,check_force=True):
        values=np.r_[support_normal_force,relative_position,bottom,vertical_speed]
        if not np.isfinite(values).all():return False,"non-finite handover measurement"
        self.elapsed_frames+=1
        self.contact_gap=0 if support_contact else self.contact_gap+1
        self.weak_force_gap=0 if not check_force or (support_contact and support_normal_force>=.02) else self.weak_force_gap+1
        self.release_gap=0 if other_contact else self.release_gap+1
        if self.release_gap>=3:self.released=True
        self.other_contact_frames=self.other_contact_frames+1 if other_contact and self.released else 0
        if table_contact:self.table_frames+=1
        if self.contact_gap>3:return False,"support hand lost contact for over 20 ms"
        if self.weak_force_gap>3:return False,"support contact has insufficient reconstructed normal force"
        if np.linalg.norm(np.asarray(relative_position)-self.anchor_local)>.04:return False,"object slipped over 4 cm relative to supporting wrist"
        if bottom<-.003:return False,"object penetrates support surface over 3 mm"
        if table_contact and vertical_speed<-.35:return False,"downward touchdown speed exceeds 0.35 m/s"
        if self.table_frames>round(.5/dt):return False,"handover table contact exceeds 0.5 s cumulative"
        return True,None

    def finished(self,bottom):
        return self.released and self.other_contact_frames>=15 and bottom>.012

def observe(model,data,q,objects,hands,support,measure_force=True):
    """Observe support side and table on current native MuJoCo state."""
    import mujoco
    from scipy.spatial.transform import Rotation
    hits=[False,False];normal=[0.,0.];table=False
    for i,c in enumerate(data.contact):
        x,y=map(int,c.geom);g=y if x in objects else x if y in objects else None
        if g is None or c.dist>.0005:continue
        name=model.geom(g).name or ""
        if g in hands:
            for side,label in enumerate(("right","left")):
                if name.startswith("collision_hand_"+label+"_"):
                    hits[side]=True
                    if measure_force:
                        f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f);normal[side]+=max(0.,float(f[0]))
        elif int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_PLANE):table=True
    rot=Rotation.from_quat(np.asarray(q)[[-10,-9,-8,-11]])
    relatives=np.stack([rot.inv().apply(data.xpos[model.body(side+'_hand_C_MC').id]-np.asarray(q)[-14:-11]) for side in ('right','left')])
    return dict(hits=hits,normal=normal,relative=relatives[support],relative_both=relatives,table=table)
