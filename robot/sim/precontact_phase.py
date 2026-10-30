"""Irreversible transition from deterministic approach to contact MPC."""
import numpy as np
class ContactPhase:
    def __init__(self,interacted=False):self.interacted=bool(interacted)
    @classmethod
    def from_history(cls,records,persisted=False):
        seen=bool(persisted)
        for row in records.values():
            counts=np.asarray(row.get('actual_contact_counts',[0,0]))
            normal=np.asarray(row.get('actual_hand_normal_force',[0.,0.]))
            seen=seen or bool(np.any(counts>0) or np.any(np.isfinite(normal)&(normal>1e-6)))
        return cls(seen)
    def observe(self,geometry,counts,normal):
        just_touched=bool(any(geometry) or np.any(np.asarray(counts)>0) or np.any(np.isfinite(normal)&(np.asarray(normal)>1e-6)))
        event=not self.interacted and just_touched
        self.interacted |= just_touched
        return event
