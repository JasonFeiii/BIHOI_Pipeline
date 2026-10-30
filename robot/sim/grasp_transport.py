"""Generate coupled wrist proposals by transporting the observed physical grasp."""
import numpy as np
from scipy.spatial.transform import Rotation

FINGERS=np.r_[6:28,34:56]

def grasp_transport(reference_q, nominal_latent, current_q, current_ctrl,
                    wrist_p, wrist_R, dt, tau=.6, follow_fingers=False):
    h=len(reference_q)
    actual_R=Rotation.from_quat(np.asarray(current_q)[[-10,-9,-8,-11]]).as_matrix()
    actual_p=np.asarray(current_q)[-14:-11]
    ref_R=Rotation.from_quat(reference_q[:,[-10,-9,-8,-11]]).as_matrix()
    ref_p=reference_q[:,-14:-11]
    decay=np.exp(-np.arange(h)*dt/tau)
    rotation_error=Rotation.from_matrix(actual_R@ref_R[0].T).as_rotvec()
    target_R=Rotation.from_rotvec(decay[:,None]*rotation_error).as_matrix()@ref_R
    target_p=ref_p+decay[:,None]*(actual_p-ref_p[0])
    latent=np.array(nominal_latent,copy=True)
    for side in range(2):
        relative_p=actual_R.T@(wrist_p[side]-actual_p)
        relative_R=actual_R.T@wrist_R[side]
        latent[:,28*side:28*side+3]=target_p+np.einsum('hij,j->hi',target_R,relative_p)
        angles=Rotation.from_matrix(target_R@relative_R).as_euler('XYZ')
        angles=np.unwrap(angles,axis=0)
        origin=nominal_latent[0,28*side+3:28*side+6]
        angles+=2*np.pi*np.round((origin-angles[0])/(2*np.pi))
        latent[:,28*side+3:28*side+6]=angles
    latent[:,FINGERS]=np.asarray(current_ctrl)[FINGERS]
    if follow_fingers:
        latent[:,FINGERS]+=nominal_latent[:,FINGERS]-nominal_latent[0,FINGERS]
    return latent

def proposals(reference_q,nominal,current_q,current_ctrl,wrist_p,wrist_R,dt):
    return np.stack([grasp_transport(reference_q,nominal,current_q,current_ctrl,wrist_p,wrist_R,dt,tau,follow)
                     for tau,follow in [(.6,False),(.6,True),(1.,False),(.3,True)]])
