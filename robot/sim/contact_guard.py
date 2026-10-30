"""Geometric contact-patch guidance; never applies forces to the object."""
import numpy as np
import torch


class ContactGuide:
    def __init__(self, model):
        self.model=model
        self.object_body=model.body('right_object').id
        self.ready=False

    def initialize(self, data, device='cuda'):
        groups=[[],[]]
        seen=[set(),set()]
        objects={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g]==self.object_body}
        for c in sorted(data.contact,key=lambda c:float(c.dist)):
            a,b=map(int,c.geom)
            g=b if a in objects else a if b in objects else None
            if g is None or c.dist>.001:continue
            name=self.model.geom(g).name or ''
            for s,side in enumerate(('right','left')):
                body=int(self.model.geom_bodyid[g])
                if name.startswith('collision_hand_'+side+'_') and body not in seen[s]:
                    p=np.asarray(c.pos).copy()
                    hp=data.xmat[body].reshape(3,3).T@(p-data.xpos[body])
                    op=data.xmat[self.object_body].reshape(3,3).T@(p-data.xpos[self.object_body])
                    normal=data.xmat[self.object_body].reshape(3,3).T@np.asarray(c.frame).reshape(3,3)[0]
                    groups[s].append({'body':body,'geom':name,'hand_local':hp.tolist(),'object_local':op.tolist(),'object_normal':normal.tolist()})
                    seen[s].add(body)
        if not all(groups):raise ValueError('Contact checkpoint must have contact on both hands')
        self.description={'method':'soft contact-patch guidance in sampled physics; no object forces','anchors':groups}
        self.groups=[]
        for group in groups:
            self.groups.append((torch.tensor([x['body'] for x in group],device=device,dtype=torch.long),
                torch.tensor([x['hand_local'] for x in group],device=device,dtype=torch.float32),
                torch.tensor([x['object_local'] for x in group],device=device,dtype=torch.float32),
                torch.tensor([x['object_normal'] for x in group],device=device,dtype=torch.float32)))
        self.ready=True

    def errors(self, xpos, xmat):
        if not self.ready:return torch.zeros((xpos.shape[0],2),device=xpos.device,dtype=xpos.dtype)
        rotations=xmat.reshape(*xpos.shape[:-1],3,3)
        out=[]
        for ids,hp,op,normals in self.groups:
            hand=xpos[:,ids]+torch.einsum('nkij,kj->nki',rotations[:,ids],hp)
            obj=xpos[:,self.object_body,None,:]+torch.einsum('nij,kj->nki',rotations[:,self.object_body],op)
            displacement=hand-obj
            world_normals=torch.einsum('nij,kj->nki',rotations[:,self.object_body],normals)
            normal_distance=(displacement*world_normals).sum(-1).abs()
            tangent_distance=torch.clamp((displacement**2).sum(-1)-normal_distance**2,min=0).sqrt()
            distances=normal_distance+.05*tangent_distance
            # Permit a finger to roll/reposition, while retaining multiple patches.
            out.append(torch.topk(distances,min(2,len(ids)),largest=False).values.mean(dim=-1))
        return torch.stack(out,dim=-1)
