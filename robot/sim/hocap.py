"""Raw HOCap takes: MANO hand meshes, object poses and hand-object contact analysis.

A take lives in data/raw/<source>/ (standardized_hoi_30fps): mano_{right,left}.npz
(standard MANO, flat_hand_mean, no PCA), obj_pose.npz (trans, quat wxyz) and
object_mesh.ply in the object frame, at 30 Hz with Z=0 the tabletop.
"""
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from robot_paths import DATA,MANO_SOURCE
RAW=DATA/'raw'
MANO=DATA/'mano'
SIDES=('right','left')
CONTACT_DISTANCE=.004  # a MANO vertex this close to the object surface touches it
CONTACT_VERTICES=5     # ... and a hand touches when at least this many vertices do
LIFT_HEIGHT=.008       # object bottom this far above the table: lifted

def mano_model(side):
    """Path of a chumpy-free MANO pickle (converted once from the operator's MANO_SOURCE)."""
    path=MANO/f'MANO_{side.upper()}.pkl'
    if path.exists():return path
    class Chumpy:
        def __init__(self,*args,**kwargs):pass
        def __setstate__(self,state):self.__dict__.update(state)
        def array(self):
            if hasattr(self,'idxs'):return self.a.array().ravel()[self.idxs].reshape(self.preferred_shape)  # chumpy Select
            return np.asarray(self.x)
    class Unpickler(pickle.Unpickler):
        def find_class(self,module,name):
            return Chumpy if module.startswith('chumpy') else super().find_class(module,name)
    with open(MANO_SOURCE/path.name,'rb') as stream:data=Unpickler(stream,encoding='latin1').load()
    MANO.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:pickle.dump({k:(v.array() if isinstance(v,Chumpy) else v) for k,v in data.items()},stream)
    temp.replace(path);return path

def mano_vertices(take,side):
    """(T, 778, 3) world vertices of one recorded hand, and the (1538, 3) faces."""
    import smplx,torch
    params=np.load(take/f'mano_{side}.npz');T=len(params['transl'])
    layer=smplx.create(str(mano_model(side)),model_type='mano',is_rhand=side=='right',use_pca=False,flat_hand_mean=True,batch_size=T)
    tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
    with torch.no_grad():
        out=layer(betas=tensor(params['betas'][None].repeat(T,0)),global_orient=tensor(params['global_orient']),
            hand_pose=tensor(params['hand_pose']),transl=tensor(params['transl']))
    return out.vertices.numpy(),layer.faces.astype(np.int32)

def load_take(source):
    """Hands, object poses and object mesh of data/raw/<source>."""
    import trimesh
    take=RAW/source
    pose=np.load(take/'obj_pose.npz')
    mesh=trimesh.load(take/'object_mesh.ply',process=False)
    hands={side:mano_vertices(take,side)[0] for side in SIDES}
    return dict(take=take,object_pos=pose['trans'].astype(float),object_quat=pose['quat'].astype(float),mesh=mesh,hands=hands)

_SIGNED_DISTANCE=None
def signed_distance(mesh,points,max_distance=.05):
    """Signed distance (m, negative inside) of points to a triangle mesh, on the GPU.

    Inside/outside comes from the generalized winding number, so open or
    inconsistently wound capture meshes are handled; points farther than
    max_distance get +max_distance."""
    global _SIGNED_DISTANCE
    import warp as wp
    if _SIGNED_DISTANCE is None:
        @wp.kernel
        def kernel(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),limit:float,out:wp.array(dtype=float)):
            i=wp.tid();p=points[i]
            q=wp.mesh_query_point_sign_winding_number(mesh,p,limit)
            out[i]=limit
            if q.result:out[i]=wp.length(p-wp.mesh_eval_position(mesh,q.face,q.u,q.v))*q.sign
        _SIGNED_DISTANCE=kernel
    device='cuda:0'
    m=wp.Mesh(points=wp.array(np.asarray(mesh.vertices,np.float32),dtype=wp.vec3,device=device),
        indices=wp.array(np.asarray(mesh.faces,np.int32).ravel(),dtype=int,device=device))
    pts=wp.array(np.asarray(points,np.float32).reshape(-1,3),dtype=wp.vec3,device=device)
    out=wp.zeros(len(pts),dtype=float,device=device)
    wp.launch(_SIGNED_DISTANCE,dim=len(pts),inputs=[m.id,pts,max_distance,out],device=device)
    return out.numpy().reshape(np.shape(points)[:-1])

def analyze(source):
    """Per-frame contact analysis of a take.

    counts (T, 2): MANO vertices within CONTACT_DISTANCE of (or inside) the object, per hand
    penetration (T, 2): deepest MANO vertex inside the object (m), per hand
    bottom (T,): lowest object point above the table (m)
    """
    data=load_take(source);mesh=data['mesh']
    R=Rotation.from_quat(data['object_quat'][:,[1,2,3,0]]).as_matrix();p=data['object_pos']
    T=len(p);counts=np.zeros((T,2),int);penetration=np.zeros((T,2))
    for h,side in enumerate(SIDES):
        local=np.einsum('tvi,tij->tvj',data['hands'][side]-p[:,None],R)  # vertices in the object frame
        d=signed_distance(mesh,local)
        counts[:,h]=(d<CONTACT_DISTANCE).sum(1);penetration[:,h]=np.maximum(-d,0).max(1)
    hull=mesh.convex_hull.vertices
    bottom=np.einsum('vj,tij->tvi',hull,R)[...,2].min(1)+p[:,2]
    return dict(counts=counts,penetration=penetration,bottom=bottom,object_pos=p,object_quat=data['object_quat'])

def persistent(flags,n):
    """Frames that start n consecutive True flags."""
    return np.flatnonzero(np.convolve(flags.astype(int),np.ones(n,int),'valid')>=n)

def contact_schedule(counts,bottom):
    """First contact (any hand, 5 frames) and the grasp anchor (both hands + lifted for 10 frames)."""
    touching=counts>=CONTACT_VERTICES
    first=persistent(touching.any(1),5);anchors=persistent(touching.all(1)&(bottom>=LIFT_HEIGHT),10)
    return (int(first[0]) if len(first) else None),(int(anchors[0]+9) if len(anchors) else None)
