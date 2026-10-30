"""Safe checkpoint beam and feedback-guided search, independent of physics."""
from dataclasses import dataclass,asdict
from typing import Optional

@dataclass
class Node:
    key:str
    run:str
    frame:int
    checkpoint:str
    quality:float
    parent:Optional[str]=None
    attempts:int=0
    last_diagnostic:str="unknown"
    feedback_level:int=0

class RecoveryBeam:
    def __init__(self,capacity=6,nodes=None,memory=None):
        self.capacity=capacity;self.nodes={};self.memory=dict(memory or {})
        for n in (nodes or []):self.add(n)
    def _restore_history(self,node):
        history=self.memory.get(node.key)
        if history and history["attempts"]>node.attempts:
            node.attempts=history["attempts"]
            node.last_diagnostic=history["last_diagnostic"]
            node.feedback_level=history["feedback_level"]
        elif not history or node.attempts>=history["attempts"]:
            self.memory[node.key]=dict(attempts=node.attempts,last_diagnostic=node.last_diagnostic,feedback_level=node.feedback_level)
    def add(self,node):
        if node.key in self.nodes:return False
        self._restore_history(node)
        self.nodes[node.key]=node;self._trim();return node.key in self.nodes
    def _rank(self,node):
        # Safe-state score is a small tie-breaker; attempts discourage loops.
        return node.frame/150.+.40*node.quality-.20*node.attempts
    def _trim(self):
        while len(self.nodes)>self.capacity:
            # Retain both a recovery route and the best-quality frontier,
            # even if previous attempts temporarily lower its selection rank.
            oldest=min(self.nodes.values(),key=lambda n:n.frame).key
            latest=max(n.frame for n in self.nodes.values())
            best=max((n for n in self.nodes.values() if n.frame==latest),key=lambda n:n.quality).key
            protected={oldest,best} if self.capacity>=2 else {best}
            victim=min((n for n in self.nodes.values() if n.key not in protected),key=self._rank)
            del self.nodes[victim.key]
    def choose(self):
        if not self.nodes:raise RuntimeError("No validated safe checkpoint available")
        return max(self.nodes.values(),key=self._rank)
    def record_failure(self,key,diagnostic):
        node=self.nodes[key];node.attempts+=1;node.last_diagnostic=diagnostic;node.feedback_level=min(3,node.feedback_level+1)
        self.memory[key]=dict(attempts=node.attempts,last_diagnostic=node.last_diagnostic,feedback_level=node.feedback_level)
    def dump(self):return [asdict(x) for x in self.nodes.values()]
    @classmethod
    def load(cls,rows,capacity=6,memory=None):return cls(capacity,[Node(**row) for row in rows],memory)

def guidance_for(diagnostic,attempts):
    # The multiplier changes a soft score only, never the sampling range or guard.
    if diagnostic=="speed":return (1.,1.6,2.5,4.)[min(attempts,3)]
    if diagnostic=="drop":return (1.,1.25,1.6,2.)[min(attempts,3)]
    return 1.

def classify_diagnostics(diag):
    c=(diag or {}).get("constraints",{})
    speed=c.get("commit_bad_object_speed",{});drop=c.get("commit_bad_drop_clearance",{})
    if speed.get("zero_count")==0 and float(speed.get("min_fraction",0))>0:return "speed"
    if drop.get("zero_count")==0 and float(drop.get("min_fraction",0))>0:return "drop"
    hand=c.get("commit_bad_warp_hand",{});arm=c.get("commit_bad_warp_arm",{})
    if hand.get("zero_count")==0 or arm.get("zero_count")==0:return "collision"
    return "mixed"
