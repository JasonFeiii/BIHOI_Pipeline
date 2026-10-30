"""Rank the HOCap takes of every object for a 0-WINDOW s simulation and pick the best one.

    python sim/rank_takes.py [--window 15] [--workers 25]

Per take (hocap.analyze over the first WINDOW seconds, after first contact):
  eligible      the object rests on the table in the first frame (it starts without
                hands in the simulation), and a two-hand lifted grasp (anchor) by
                WINDOW-3 s, so there is manipulation after it. Early contact is fine:
                prepare_task.py adds a still lead-in so the open-hand approach fits
                before it
  penetration   95th percentile of the deepest MANO vertex inside the object (mm)
  hold          fraction of frames after the anchor where both hands touch the lifted object
  jitter        RMS object acceleration at the 10 Hz capture rate (m/s^2, rad/s^2)
  wrist_error   mean retargeting wrist error (mm, data/retargeted metrics.json)
Eligible takes are ordered by the sum of their per-object ranks in these five
measures (lower is better). Writes data/tasks/take_ranking.json.
"""
import argparse,json,os
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from robot_paths import DATA
OUT=DATA/'tasks'/'take_ranking.json'

def measure(job):
    entry,window=job
    os.environ.setdefault('OMP_NUM_THREADS','1')
    import torch;torch.set_num_threads(1)
    import hocap
    from scipy.spatial.transform import Rotation
    a=hocap.analyze(entry['source']);end=int(window*30)+1
    first,anchor=hocap.contact_schedule(a['counts'][:end],a['bottom'][:end])
    metrics=json.loads((DATA/'retargeted'/'sharpa'/'bimanual'/entry['task']/'0'/'metrics.json').read_text())
    result=dict(task=entry['task'],object=entry['source'].split('/')[1],source=entry['source'],
        first_contact_frame=first,anchor_frame=anchor,wrist_error_mm=metrics['wrist_mean_mm'])
    result['start_bottom_m']=float(a['bottom'][0])
    result['eligible']=first is not None and anchor is not None and anchor<=(window-3)*30 and bool(a['bottom'][0]<hocap.LIFT_HEIGHT)
    if first is None:return result
    s=slice(first,end);pen=a['penetration'][s].max(1)
    touching=(a['counts']>=hocap.CONTACT_VERTICES).all(1)&(a['bottom']>=hocap.LIFT_HEIGHT)
    k=np.arange(first-first%3,end,3)  # 10 Hz capture frames
    acc=np.diff(a['object_pos'][k],2,axis=0)*100
    rot=Rotation.from_quat(a['object_quat'][k][:,[1,2,3,0]]);w=(rot[1:]*rot[:-1].inv()).as_rotvec()*10
    result.update(penetration_mm=float(np.percentile(pen,95)*1000),
        hold=float(touching[anchor:end].mean()) if anchor is not None else 0.,
        jitter_linear=float(np.sqrt((acc**2).sum(1).mean())),jitter_angular=float(np.sqrt((np.diff(w,axis=0)*10)**2).sum(1).mean()**.5),
        lift_max_m=float(a['bottom'][s].max()))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--window',type=float,default=15.);p.add_argument('--workers',type=int,default=25)
    a=p.parse_args()
    manifest=json.loads((DATA/'manifest.json').read_text())
    with Pool(a.workers) as pool:rows=pool.map(measure,[(e,a.window) for e in manifest],chunksize=1)
    keys=(('penetration_mm',1),('hold',-1),('jitter_linear',1),('jitter_angular',1),('wrist_error_mm',1))
    ranking={}
    for obj in sorted({r['object'] for r in rows}):
        ok=[r for r in rows if r['object']==obj and r['eligible']]
        for key,sign in keys:
            for rank,r in enumerate(sorted(ok,key=lambda r:sign*r[key])):r['rank_sum']=r.get('rank_sum',0)+rank
        order=sorted(ok,key=lambda r:(r['rank_sum'],r['penetration_mm']))
        ranking[obj]=dict(best=order[0]['task'] if order else None,order=[r['task'] for r in order],
            takes=sorted([r for r in rows if r['object']==obj],key=lambda r:r['task']))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(dict(window_s=a.window,criteria=__doc__,objects=ranking),indent=2))
    for obj,v in ranking.items():
        best=next((r for r in v['takes'] if r['task']==v['best']),None)
        print(f'{obj:28s} {len(v["order"])}/{len(v["takes"])} eligible  best={v["best"]}'+
            (f'  anchor={best["anchor_frame"]/30:.1f}s pen={best["penetration_mm"]:.1f}mm hold={best["hold"]:.2f}' if best else ''))
    print(OUT)

if __name__=='__main__':main()
