"""Read solver forces using each explicit constraint-row address.

Warp pyramidal contacts can have interleaved rows; contiguous slicing can
silently read another contact's forces. Never assume address[0]+offset.
"""
import numpy as np,torch,warp as wp

def decode_contact(force,addresses,friction,dim,pyramidal=True):
    out=np.zeros(6);addresses=np.asarray(addresses,dtype=int)
    if pyramidal and dim>1:
        n=2*(int(dim)-1);a=addresses[:n]
        if np.any(a<0) or np.any(a>=len(force)):return out
        edges=np.asarray(force)[a];out[0]=edges.sum()
        out[1:dim]=(edges[::2]-edges[1::2])*np.asarray(friction)[:dim-1]
    else:
        a=addresses[:dim]
        if np.any(a<0) or np.any(a>=len(force)):return out
        out[:dim]=np.asarray(force)[a]
    return out

class SolverContacts:
    def __init__(self,model):
        self.object_geoms={g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('right_object').id}
        self.hand_side={g:(0 if (model.geom(g).name or '').startswith('collision_hand_right_') else 1) for g in range(model.ngeom) if (model.geom(g).name or '').startswith('collision_hand_')}
    def read(self,model,data):
        wp.synchronize_device(data.qpos.device)
        count=int(wp.to_torch(data.nacon).reshape(-1)[0]);normal=np.zeros(2);world_force=np.zeros((2,3));contacts=np.zeros(2,dtype=int);total=np.zeros(3)
        if count:
            world=wp.to_torch(data.contact.worldid)[:count];ids=torch.nonzero(world==0,as_tuple=False).flatten()
            geom=wp.to_torch(data.contact.geom)[ids].cpu().numpy()
            chosen=[i for i,(x,y) in enumerate(geom) if int(x) in self.object_geoms or int(y) in self.object_geoms]
            if chosen:
                ci=ids[chosen]
                frames=wp.to_torch(data.contact.frame)[ci].cpu().numpy().reshape(-1,3,3)
                addresses=wp.to_torch(data.contact.efc_address)[ci].cpu().numpy()
                dims=wp.to_torch(data.contact.dim)[ci].cpu().numpy()
                friction=wp.to_torch(data.contact.friction)[ci].cpu().numpy()
                nefc=int(wp.to_torch(data.nefc)[0]);force=wp.to_torch(data.efc.force)[0,:nefc].cpu().numpy()
                for k,i in enumerate(chosen):
                    values=decode_contact(force,addresses[k],friction[k],int(dims[k]),int(model.opt.cone)==0)
                    x,y=map(int,geom[i]);other=x if y in self.object_geoms else y
                    sign=1. if y in self.object_geoms else -1.
                    contact_force=sign*(values[:3]@frames[k]);total+=contact_force
                    if other in self.hand_side:
                        side=self.hand_side[other];normal[side]+=max(0.,float(values[0]));world_force[side]+=contact_force
                        if values[0]>=.02:contacts[side]+=1
        return dict(normal=normal,world_force=world_force,counts=contacts,object_force_world=total,source='solver_explicit_constraint_rows')
