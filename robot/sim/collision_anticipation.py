"""Kinematic collision warning only; all actions still need physics verification."""
import numpy as np,mujoco
class CollisionAnticipation:
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        right=[g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith('right_arm_') and model.geom_contype[g]]
        left=[g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith('left_arm_') and model.geom_contype[g]]
        self.pairs=[(x,y) for x in right for y in left if (model.geom_contype[x]&model.geom_conaffinity[y]) or (model.geom_contype[y]&model.geom_conaffinity[x])]
    def _distances(self,q):
        self.data.qpos[:]=q;mujoco.mj_kinematics(self.model,self.data)
        return np.array([mujoco.mj_geomDistance(self.model,self.data,x,y,.15,None) for x,y in self.pairs])
    def evaluate(self,q,v,lookahead=.3):
        now=self._distances(q);future=np.asarray(q).copy();mujoco.mj_integratePos(self.model,future,np.asarray(v),lookahead);later=self._distances(future)
        if not len(now):return dict(risk=False,clearance_m=.15,predicted_clearance_m=.15,pair=[])
        at=int(np.argmin(later));closing=(now[at]-later[at])/lookahead
        risk=bool(now[at]<.075 and later[at]<.015 and closing>.015)
        return dict(risk=risk,clearance_m=float(now[at]),predicted_clearance_m=float(later[at]),closing_m_s=float(closing),pair=[self.model.body(self.model.geom_bodyid[g]).name for g in self.pairs[at]],scope='constant-velocity kinematic warning, not executed state')
