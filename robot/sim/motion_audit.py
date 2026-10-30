"""Check executed rotational progress, including turns invisible to end-pose error."""
import numpy as np
from scipy.spatial.transform import Rotation
def rotation_progress(actual_wxyz,reference_wxyz,dt=1/150):
 a=Rotation.from_quat(np.asarray(actual_wxyz)[:,[1,2,3,0]]);r=Rotation.from_quat(np.asarray(reference_wxyz)[:,[1,2,3,0]])
 da=(a[1:]*a[:-1].inv()).as_rotvec();dr=(r[1:]*r[:-1].inv()).as_rotvec();size=np.linalg.norm(dr,axis=-1);active=size>.15*dt
 unit=dr/np.maximum(size[:,None],1e-12);progress=(da*unit).sum(-1)
 expected=float(size[active].sum());achieved=float(np.clip(progress[active],-2*size[active],2*size[active]).sum())
 return dict(reference_rotation_path_deg=float(np.rad2deg(expected)),signed_rotation_progress_deg=float(np.rad2deg(achieved)),progress_ratio=(None if expected<1e-9 else achieved/expected),actual_vertical_component_deg=float(np.rad2deg(da[:,2].sum())),reference_vertical_component_deg=float(np.rad2deg(dr[:,2].sum())),median_pose_error_deg=float(np.median(np.rad2deg((a.inv()*r).magnitude()))))
def first_stalled_motion(time,actual,reference,width=1.2):
 t=np.asarray(time);start=float(t[0]);end=float(t[-1]);streak=0;first=None
 for at in np.arange(start,max(start,end-width)+1e-8,.2):
  ids=np.flatnonzero((t>=at-1e-7)&(t<=at+width+1e-7))
  if len(ids)<2:continue
  m=rotation_progress(np.asarray(actual)[ids],np.asarray(reference)[ids],float(np.median(np.diff(t[ids]))))
  bad=m['reference_rotation_path_deg']>15 and m['progress_ratio']<.25
  if bad:
   if streak==0:first=at
   streak+=1
   if streak>=3:return dict(time_s=float(first),reason='sustained lack of reference-directed rotation',metrics=m)
  else:streak=0;first=None
 return None
