"""Evidence for independently ranking physical checkpoints and prefixes."""
import hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from recovery_beam import Node

class Evidence:
    def __init__(self,vertices):
        self.vertices=np.asarray(vertices);self.cache=OrderedDict();self.acquisition_cache=OrderedDict()
    def _trace(self,run):
        run=Path(run);key=str(run)
        if key not in self.cache:
            self.cache[key]=(dict(np.load(run/"native_trajectory.npz")),dict(np.load(run/"trajectory.npz")),dict(np.load(run/"reference.npz")))
        self.cache.move_to_end(key)
        while len(self.cache)>8:self.cache.popitem(last=False)
        return self.cache[key]
    def bottom(self,q):
        q=np.asarray(q);R=Rotation.from_quat(q[:,[-10,-9,-8,-11]]).as_matrix()
        out=[]
        for start in range(0,len(q),64):
            x=R[start:start+64,2,:]@self.vertices.T
            out.append(x.min(axis=-1)+q[start:start+64,-12])
        return np.concatenate(out) if out else np.empty(0)
    def inspect(self,run,checkpoint,start_s,through_s=None):
        run=Path(run);native,display,ref=self._trace(run)
        with np.load(checkpoint) as ck:
            frame=int(ck["sim"]);state_q=ck["state_qpos"].reshape(-1).copy();state_v=ck["state_qvel"].reshape(-1).copy();state_ctrl=ck["state_ctrl"].reshape(-1).copy()
        # A resume point must satisfy the same clearance used to screen the
        # first predicted step. Older accepted prefixes remain preserved.
        import mujoco
        from wrist_clearance import allowed as wrist_clearance_allowed
        if not hasattr(self,'clearance_models'):self.clearance_models={}
        key=str(run)
        if key not in self.clearance_models:
            scene=list(run.glob('model/sharpa/bimanual/*/0/scene.xml'))
            if len(scene)!=1:return None
            m=mujoco.MjModel.from_xml_path(str(scene[0]))
            self.clearance_models[key]=(m,mujoco.MjData(m))
            while len(self.clearance_models)>8:self.clearance_models.pop(next(iter(self.clearance_models)))
        m,d=self.clearance_models[key];d.qpos[:]=state_q;mujoco.mj_kinematics(m,d)
        if not wrist_clearance_allowed(m,d):return None
        end=frame/150.
        if through_s is not None and end>through_s+1e-7:return None
        ti=native["time"];end_idx=int(np.argmin(abs(ti-end)))
        if abs(float(ti[end_idx])-end)>1e-6:return None
        if not np.allclose(native["qpos"][end_idx],state_q,rtol=0,atol=1e-6):return None
        if not np.allclose(native["qvel"][end_idx],state_v,rtol=0,atol=1e-6):return None
        ids=np.flatnonzero((ti>=start_s-1e-7)&(ti<=end+1e-7))
        if not len(ids):return None
        if any(k not in native for k in ("gravity","object_weld_active","object_external_force")):return None
        if not np.isfinite(native["qpos"][ids]).all() or not np.isfinite(native["qvel"][ids]).all():return None
        if not np.allclose(native["gravity"][ids],[0,0,-9.81],atol=1e-5) or np.any(native["object_weld_active"][ids]) or np.any(native["object_external_force"][ids]):return None
        refi=np.clip(np.rint(ti[ids]*150).astype(int),0,len(ref["qpos"])-1)
        obj=self.bottom(native["qpos"][ids]);target=self.bottom(ref["qpos"][refi])
        # During a reference-indicated table placement, actual support is allowed.
        bad=(target>.010)&(obj<.0095)
        # Generated open-hand approach may lag the raw reference lift. Resting
        # on the real support before the first actual controlled lift is an
        # acquisition phase, not a later object drop.
        acquisition_key=(str(run),len(ti),float(ti[-1]))
        if acquisition_key not in self.acquisition_cache:
            full_ids=np.arange(len(ti));full_bottom=self.bottom(native['qpos'])
            self.acquisition_cache[acquisition_key]=self.initial_table_acquisition(run,native,full_ids,full_bottom)
            while len(self.acquisition_cache)>8:self.acquisition_cache.popitem(last=False)
        acquisition=self.acquisition_cache[acquisition_key][ids]
        bad &= ~acquisition
        controlled=np.zeros(len(ids),dtype=bool)
        if bad.any():
            controlled=self.verify_controlled_support(run,native,ids,obj)
            if (bad & ~controlled).any():return None
        recent=ids[ti[ids]>=max(start_s,end-.2)-1e-7]
        recent_bottom=self.bottom(native["qpos"][recent]);min_bottom=float(recent_bottom.min())
        speed=float(np.linalg.norm(state_v[-12:-9]));ang_speed=float(np.linalg.norm(state_v[-9:-6]))
        current_ref=ref["qpos"][min(frame,len(ref["qpos"])-1)]
        pos_err=float(np.linalg.norm(state_q[-14:-11]-current_ref[-14:-11]))
        rot_error=float((Rotation.from_quat(state_q[[-10,-9,-8,-11]]).inv()*Rotation.from_quat(current_ref[[-10,-9,-8,-11]])).magnitude())
        dmask=(display["time"]>=end-.2-1e-7)&(display["time"]<=end+1e-7)
        contacts=display["side_contacts"][dmask];bilateral=float(np.mean(np.all(contacts>0,axis=1))) if len(contacts) else 0.
        # Contact is evidence for ranking, never an every-frame hard requirement.
        quality=(.25*np.clip(min_bottom/.08,0,1)+.25*np.exp(-speed/.35)+.20*bilateral+
                 .15*np.exp(-pos_err/.08)+.10*np.exp(-rot_error/1.5)+.05*np.exp(-ang_speed/4.))
        fingerprint=hashlib.sha256(np.round(np.r_[state_q,state_v,state_ctrl],3).tobytes()).hexdigest()[:20]
        evidence=dict(controlled_low_clearance_frames=int(np.count_nonzero(bad & controlled)),start_s=float(start_s),end_s=end,min_segment_bottom_m=float(obj.min()),min_recent_bottom_m=min_bottom,
                      speed_m_s=speed,angular_speed_rad_s=ang_speed,bilateral_fraction=bilateral,position_error_m=pos_err,rotation_error_deg=float(np.rad2deg(rot_error)),quality=float(quality))
        return Node(key=f"{frame}:{fingerprint}",run=run.name,frame=frame,checkpoint=str(checkpoint),quality=float(quality)),evidence

    def initial_table_acquisition(self,run,native,ids,bottoms):
        import json,mujoco
        allow=np.zeros(len(ids),dtype=bool)
        policy=Path(run)/'reference_policy.json'
        if not policy.exists() or abs(float(native['time'][0]))>1e-7:return allow
        try:policy=json.loads(policy.read_text())
        except (OSError,ValueError):return allow
        if policy.get('source_start_frame')!=0 or policy.get('inserted_grasp_validation_hold_seconds')!=0:return allow
        allbottom=self.bottom(native['qpos']);force=native.get('actual_hand_normal_force')
        if force is None:return allow
        airborne=np.flatnonzero(allbottom>.01)
        lift_start=int(airborne[0]) if len(airborne) else len(allbottom)
        # Once airborne, acquisition never reopens after a later drop.
        candidate=np.flatnonzero((ids<lift_start)&(bottoms<.0095)&(bottoms>=-.003))
        if not len(candidate):return allow
        scenes=list(Path(run).glob('model/sharpa/bimanual/*/0/scene.xml'))
        if len(scenes)!=1:return allow
        m=mujoco.MjModel.from_xml_path(str(scenes[0]));d=mujoco.MjData(m)
        obj={g for g in range(m.ngeom) if m.geom_bodyid[g]==m.body('right_object').id}
        planes=[g for g in range(m.ngeom) if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_PLANE)]
        for j in candidate:
            i=ids[j]
            if abs(float(native['qvel'][i,-10]))>.35:continue
            d.qpos[:]=native['qpos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
            # Millimetre-scale passive bounce above the actual table during
            # initial acquisition is distinct from losing a lifted object.
            near_table=any(mujoco.mj_geomDistance(m,d,g,h,.003,None)<.003 for g in obj for h in planes)
            if near_table:allow[j]=True;continue
            force_ok=bool(native.get('actual_force_available',np.zeros(len(allbottom),bool))[i] and np.any(force[i]>.02))
            for con in d.contact:
                a,b=map(int,con.geom);other=b if a in obj else a if b in obj else None
                if other is not None and force_ok and (m.geom(other).name or '').startswith('collision_hand_') and con.dist<=.002:
                    allow[j]=True;break
                if other is not None and int(m.geom_type[other])==int(mujoco.mjtGeom.mjGEOM_PLANE) and con.dist<=.002:
                    allow[j]=True;break
        return allow

    def verify_controlled_support(self,run,native,ids,bottoms):
        from temporal_support import verify_temporal_support
        key=(str(run),len(native['time']),float(native['time'][-1]))
        if not hasattr(self,'temporal_cache'):self.temporal_cache=OrderedDict()
        if key not in self.temporal_cache:
            self.temporal_cache[key]=verify_temporal_support(run,native,self.bottom(native['qpos']))
            while len(self.temporal_cache)>8:self.temporal_cache.popitem(last=False)
        return self.temporal_cache[key][ids]
