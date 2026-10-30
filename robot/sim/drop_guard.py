"""Physical object-clearance screening without a hard reference tracking bound."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
MAX_WINDOW_DROP=.030
MIN_CLEARANCE=.010
REFERENCE_SUPPORT=.010
SUPPORT_PENETRATION=.003

def bottom_height(q,vertices):
 q=np.asarray(q);r=Rotation.from_quat(q[[-10,-9,-8,-11]]).as_matrix()
 return float(np.min(vertices@r[2,:])+q[-12])

def bottom_height_tensor(q,vertices):
 quat=q[:,-11:-7];quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True).clamp_min(1e-12)
 w,x,y,z=quat.unbind(-1)
 row=torch.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],dim=-1)
 return q[:,-12]+torch.amin(row@vertices.T,dim=-1)

def min_safe_bottom(start_bottom, reference_start_bottom, reference_bottom):
 floor=MIN_CLEARANCE if reference_bottom>REFERENCE_SUPPORT else -SUPPORT_PENETRATION
 ref_descent=min(0.,reference_bottom-reference_start_bottom)
 return max(floor,start_bottom+ref_descent-MAX_WINDOW_DROP)

def allowed(q,reference_q,vertices,start_bottom,reference_start_bottom):
 ref_bottom=bottom_height(reference_q,vertices);bottom=bottom_height(q,vertices)
 limit=min_safe_bottom(start_bottom,reference_start_bottom,ref_bottom)
 if not np.isfinite([bottom,ref_bottom,start_bottom,reference_start_bottom]).all():return False,'non-finite drop geometry'
 if bottom<limit:return False,f'object bottom {bottom:.4f} m below safe bound {limit:.4f} m'
 return True,None

def preview_drop_cost(bottom,start_bottom,reference_start_bottom,reference_bottom):
    """Soft loss on unplanned bottom descent; does not reject candidates."""
    target=start_bottom+min(0.,reference_bottom-reference_start_bottom)
    loss=torch.clamp((target-bottom-.010)/.030,min=0.,max=4.)
    return 1.5*loss.square()

def speed_allowed(velocity,reference_velocity):
    """Reject exceptional object speed, allow reference-driven fast motion."""
    v=np.asarray(velocity);r=np.asarray(reference_velocity)
    linear=float(np.linalg.norm(v[-12:-9]));angular=float(np.linalg.norm(v[-9:-6]))
    linear_limit=max(.6,2.*float(np.linalg.norm(r[-12:-9]))+.25)
    angular_limit=max(4.,2.*float(np.linalg.norm(r[-9:-6]))+2.)
    if not np.isfinite([linear,angular,linear_limit,angular_limit]).all():return False,'non-finite object speed'
    if linear>linear_limit:return False,f'object speed {linear:.3f} m/s exceeds {linear_limit:.3f} m/s'
    if angular>angular_limit:return False,f'object angular speed {angular:.3f} rad/s exceeds {angular_limit:.3f} rad/s'
    return True,None

def speed_allowed_tensor(velocity,reference_velocity):
    lin=torch.linalg.vector_norm(velocity[:,-12:-9],dim=-1);ang=torch.linalg.vector_norm(velocity[:,-9:-6],dim=-1)
    ref_lin=torch.linalg.vector_norm(reference_velocity[-12:-9]);ref_ang=torch.linalg.vector_norm(reference_velocity[-9:-6])
    return torch.isfinite(lin)&torch.isfinite(ang)&(lin<=torch.clamp(2*ref_lin+.25,min=.6))&(ang<=torch.clamp(2*ref_ang+2.,min=4.))

def speed_preview_cost(velocity,reference_velocity):
    """Soft forecast penalty increasing before hard speed limits are crossed."""
    lin=torch.linalg.vector_norm(velocity[:,-12:-9],dim=-1);ang=torch.linalg.vector_norm(velocity[:,-9:-6],dim=-1)
    ref_lin=torch.linalg.vector_norm(reference_velocity[-12:-9]);ref_ang=torch.linalg.vector_norm(reference_velocity[-9:-6])
    lin_limit=torch.clamp(2*ref_lin+.25,min=.6);ang_limit=torch.clamp(2*ref_ang+2.,min=4.)
    le=torch.clamp((lin-.5*lin_limit)/.25,min=0.,max=4.);ae=torch.clamp((ang-.5*ang_limit)/1.5,min=0.,max=4.)
    return 1.5*le.square()+.25*ae.square()
