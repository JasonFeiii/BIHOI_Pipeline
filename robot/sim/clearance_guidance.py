"""Exact capsule geometry and soft, velocity-aware collision anticipation.

This changes trajectory scores only. It never changes physics or feasibility.
"""
import torch


def segment_distance(p0, p1, q0, q1):
    u, v, w = p1-p0, q1-q0, p0-q0
    dot = lambda x, y: (x*y).sum(-1)
    a, b, c, d, e = dot(u,u), dot(u,v), dot(v,v), dot(u,w), dot(v,w)
    det = a*c-b*b
    safe_det = det.clamp_min(1e-20)
    s, t = (b*e-c*d)/safe_det, (a*e-b*d)/safe_det
    valid = (det > 1e-8*(a*c).clamp_min(1e-20)) & (s>=0) & (s<=1) & (t>=0) & (t<=1)
    inner = torch.linalg.vector_norm(w+s[...,None]*u-t[...,None]*v, dim=-1)
    inner = torch.where(valid, inner, torch.full_like(inner, torch.inf))
    def point_segment(p, x, direction, length2):
        alpha=(dot(p-x,direction)/length2.clamp_min(1e-20)).clamp(0,1)
        return torch.linalg.vector_norm(p-x-alpha[...,None]*direction,dim=-1)
    return torch.stack((inner,point_segment(p0,q0,v,c),point_segment(p1,q0,v,c),
                        point_segment(q0,p0,u,a),point_segment(q1,p0,u,a)),dim=-1).amin(-1)


class CapsuleClearance:
    def __init__(self, model):
        import mujoco
        self.geoms = [[g for g in range(model.ngeom) if model.geom_contype[g]
                       and (model.body(model.geom_bodyid[g]).name or '') == side+'_arm_wrist_2_link']
                      for side in ('right','left')]
        if not all(self.geoms): raise ValueError('Missing wrist geometry')
        if any(int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE) for side in self.geoms for g in side):
            raise ValueError('Exact capsule guidance requires capsule wrist geometry')
        self.sizes = [model.geom_size[g,:2].copy() for g in self.geoms]

    def __call__(self, positions, matrices):
        if matrices.shape[-1] == 9: matrices=matrices.reshape(matrices.shape[:-1]+(3,3))
        ends=[]
        for ids, size in zip(self.geoms,self.sizes):
            size=torch.as_tensor(size,device=positions.device,dtype=positions.dtype)
            center=positions[...,ids,:]
            axis=matrices[...,ids,:,2]
            half=size[:,1,None]*axis
            ends.append((center-half,center+half,size[:,0]))
        a,b,ra=ends[0];c,d,rb=ends[1]
        distances=segment_distance(a[..., :,None,:],b[..., :,None,:],c[...,None,:,:],d[...,None,:,:])
        distances=distances-ra[:,None]-rb[None,:]
        return distances.amin(dim=(-1,-2))


def clearance_risk_cost(distance, previous, dt, lookahead=.15, margin=.01, weight=30000.):
    # d_dot < 0 means closing. A constant-velocity projection supplies a soft
    # urgency signal; it is not a claim that a collision is inevitable.
    closing=((previous-distance)/dt).clamp_min(0.)
    projected=distance-lookahead*closing
    deficit=(margin-distance).clamp_min(0.)
    projected_deficit=(margin-projected).clamp_min(0.)
    cost=weight*deficit.square()+weight*projected_deficit.square()
    return cost, closing
