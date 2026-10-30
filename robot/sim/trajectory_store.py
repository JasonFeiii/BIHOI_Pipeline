"""Immutable trajectory chunks with an atomic, branch-aware active manifest."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json
import numpy as np


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def atomic_npz(path,arrays):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as out:np.savez_compressed(out,**arrays)
    temp.replace(path)


class TrajectoryReader:
    def __init__(self,max_cached_files=512):
        self.cache=OrderedDict();self.max_cached_files=max_cached_files

    def npz(self,path):
        path=Path(path);key=(str(path.resolve()),path.stat().st_mtime_ns)
        if key not in self.cache:
            with np.load(path) as data:self.cache[key]={k:data[k].copy() for k in data.files}
            while len(self.cache)>self.max_cached_files:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def read(self,run,kind='native',upto=None,seen=None):
        run=Path(run).resolve();seen=set() if seen is None else seen
        if run in seen:raise ValueError('Trajectory manifest cycle')
        seen=seen|{run};manifest=run/'trajectory_manifest.json'
        if not manifest.exists():
            data=self.npz(run/('native_trajectory.npz' if kind=='native' else 'trajectory.npz'))
            mask=np.ones(len(data['time']),bool) if upto is None else data['time']<=upto+1e-8
            return {key:value[mask] for key,value in data.items()}
        state=json.loads(manifest.read_text());limit=float(state['end_time_s'])
        if upto is not None:limit=min(limit,upto)
        parts=[]
        if state['base']:
            base=state['base']
            parts.append(self.read(base['run'],kind,min(limit,base['end_time_s']),seen))
        prefix=kind+'__'
        for entry in state['chunks']:
            arrays=self.npz(run/entry['file'])
            data={key[len(prefix):]:value for key,value in arrays.items() if key.startswith(prefix)}
            if not data or not len(data['time']):continue
            mask=data['time']<=min(limit,entry['end_time_s'])+1e-8
            if parts:mask&=data['time']>parts[-1]['time'][-1]+1e-8 if len(parts[-1]['time']) else True
            if mask.any():parts.append({key:value[mask] for key,value in data.items()})
        if not parts:return {'time':np.empty(0)}
        nonempty=[part for part in parts if len(part['time'])]
        if not nonempty:return parts[0]
        keys=set.intersection(*(set(part) for part in nonempty))
        return {key:np.concatenate([part[key] for part in nonempty]) for key in keys}


class TrajectoryJournal:
    def __init__(self,run,base_run=None,base_end=None):
        self.run=Path(run);self.run.mkdir(parents=True,exist_ok=True)
        (self.run/'chunks').mkdir(exist_ok=True)
        self.path=self.run/'trajectory_manifest.json'
        self.serial=max([int(p.stem) for p in (self.run/'chunks').glob('*.npz')]+[0])
        self.revision=0
        self.state=dict(version=1,revision=0,base=None,chunks=[],end_time_s=-1.)
        if base_run is not None:
            self.state.update(base=dict(run=str(Path(base_run).resolve()),end_time_s=float(base_end)),end_time_s=float(base_end))
        self.publish()

    def publish(self):
        self.revision+=1;self.state['revision']=self.revision;atomic_json(self.path,self.state)

    def snapshot(self):return deepcopy(self.state)

    def restore(self,state):
        self.state=deepcopy(state);self.publish()

    def append(self,native,display):
        if not len(native['time']):return
        times=native['time']
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic trajectory chunk')
        if times[0]<=self.state['end_time_s']+1e-8:raise ValueError('Duplicate or rewound trajectory append')
        self.serial+=1;file=f'chunks/{self.serial:08d}.npz'
        arrays={kind+'__'+key:value for kind,data in [('native',native),('display',display)] for key,value in data.items()}
        atomic_npz(self.run/file,arrays)
        self.state['chunks'].append(dict(file=file,start_time_s=float(times[0]),end_time_s=float(times[-1])))
        self.state['end_time_s']=float(times[-1]);self.publish()


def load_trajectory(run,kind='native',upto=None):return TrajectoryReader().read(run,kind,upto)
