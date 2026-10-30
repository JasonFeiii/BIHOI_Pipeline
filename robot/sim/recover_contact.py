"""Bounded branch search from physically stable saved checkpoints."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from build_arm_scene import ROOT


def stability(model, trace, ref, end, width=.6):
    data=mujoco.MjData(model);objects={g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('right_object').id}
    indices=np.flatnonzero((trace['time']>=end-width-1e-8)&(trace['time']<=end+1e-8))
    idx=np.unique(np.r_[indices[::5],indices[-1:]])
    rows=[]
    for i in idx:
        q=trace['qpos'][i];t=float(trace['time'][i]);data.qpos[:]=q;mujoco.mj_forward(model,data)
        touches=[False,False];pen=0.
        for c in data.contact:
            x,y=map(int,c.geom);g=y if x in objects else x if y in objects else None
            if g is None:continue
            name=model.geom(g).name or ''
            for s,side in enumerate(('right','left')):
                if name.startswith('collision_hand_'+side+'_'):
                    touches[s]|=c.dist<=.0005;pen=max(pen,-float(c.dist))
        target=ref['qpos'][round(t*150)]
        angle=np.degrees((Rotation.from_quat(q[[-10,-9,-8,-11]]).inv()*Rotation.from_quat(target[[-10,-9,-8,-11]])).magnitude())
        rows.append((touches,np.linalg.norm(q[-14:-11]-target[-14:-11]),angle,pen))
    if not rows:return {'stable':False}
    both=np.array([all(r[0]) for r in rows]);maxpos=max(r[1] for r in rows);maxrot=max(r[2] for r in rows)
    gaps=[]
    for side in range(2):
        longest=gap=0
        for row in rows:
            gap=0 if row[0][side] else gap+1;longest=max(longest,gap)
        gaps.append(longest/30)
    stable=bool(both.mean()>=.75 and max(gaps)<=.10 and maxpos<.055 and maxrot<40. and max(r[3] for r in rows)<.008)
    return {'stable':stable,'bilateral_fraction':float(both.mean()),'maximum_side_gap_s':gaps,'position_max_m':float(maxpos),'rotation_max_deg':float(maxrot)}


def choose_checkpoint(model,ref,sources,before,used):
    candidates=[];traces={}
    for name in sources:
        path=ROOT/'physics'/name
        if not (path/'native_trajectory.npz').exists():continue
        traces[name]=np.load(path/'native_trajectory.npz')
        checkpoints=list(path.glob('checkpoint_*.npz'))
        if (path/'checkpoint.npz').exists():checkpoints.append(path/'checkpoint.npz')
        for ck in checkpoints:
            with np.load(ck) as z:
                frame=int(z['sim'])
                if frame<300 or frame/150>before+1e-8:continue
                key=str(frame)+':'+hashlib.sha256(z['state_qpos'].tobytes()).hexdigest()[:12]
            if used.get(key,0)>=2:continue
            candidates.append((frame,-used.get(key,0),name,ck,key))
    candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
    for frame,count,name,ck,key in candidates:
        metrics=stability(model,traces[name],ref,frame/150)
        if metrics['stable']:return (frame,count,name,ck,key,metrics)
    raise RuntimeError('No saved checkpoint passes bilateral stability checks')


def copy_prefix(source,checkpoint,target,frame):
    target.mkdir(parents=True,exist_ok=False)
    shutil.copy2(checkpoint,target/'checkpoint.npz')
    for name in ('native_trajectory','trajectory'):
        z=dict(np.load(source/(name+'.npz')));mask=z['time']<=frame/150+1e-8
        np.savez_compressed(target/(name+'.npz'),**{k:v[mask] for k,v in z.items()})
    (target/'provenance.json').write_text(json.dumps({'source':str(source),'frame':frame,'prefix_exact':True},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempts',type=int,default=3);p.add_argument('--end',type=float,default=12.6);p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--tag',default='contact_recovery');p.add_argument('--initial-before',type=float,default=8.);p.add_argument('--finger-warm-seed',action='store_true')
    p.add_argument('--source',action='append',default=[])
    p.add_argument('--contact-prior',action='store_true')
    p.add_argument('--samples',type=int,default=128);p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--seed-base',type=int,default=17);p.add_argument('--checkpoint-interval',type=float,default=.2)
    p.add_argument('--horizon',type=float,default=1.2)
    a=p.parse_args()
    model=mujoco.MjModel.from_xml_path(str(ROOT/'sqp_scene_001.xml'))
    ref=np.load(ROOT/'physics/motion_smooth_fast/reference.npz')
    sources=list(dict.fromkeys(a.source+['contact_recovery_01','smooth_seed_8s','servo_extended','motion_smooth_fast']));used={};before=a.initial_before;ledger=[]
    ledger_path=ROOT/(a.tag+'_ledger.json')
    if a.resume and ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        for entry in ledger:
            sources.insert(0,entry['run'])
            seed=ROOT/'physics'/entry['result']['arguments']['resume_from']/'checkpoint.npz'
            with np.load(seed) as z:
                key=str(int(z['sim']))+':'+hashlib.sha256(z['state_qpos'].tobytes()).hexdigest()[:12]
            used[key]=used.get(key,0)+1
        before=max(ledger[-1]['from_s'],ledger[-1]['result']['time_s']-.4)
    for attempt in range(len(ledger),a.attempts):
        frame,_,source,ck,key,quality=choose_checkpoint(model,ref,sources,before,used)
        print(json.dumps({'attempt':attempt+1,'source':source,'rewind_s':frame/150,'stability':quality}),flush=True)
        if a.dry_run:return
        seed_name=f'{a.tag}_seed_{attempt+1:02d}';name=f'{a.tag}_{attempt+1:02d}'
        copy_prefix(ROOT/'physics'/source,ck,ROOT/'physics'/seed_name,frame)
        if (ROOT/'physics'/name).exists():raise RuntimeError('Refusing to overwrite '+name)
        used[key]=used.get(key,0)+1
        command=[sys.executable,'-u',str(Path(__file__).with_name('arm_mpc.py')),
          '--name',name,'--plan-result','sqp_result_001.json','--resume-from',seed_name,
          '--samples',str(a.samples),'--iterations',str(a.iterations),'--horizon',str(a.horizon),'--native-screen','--first-pass',
          '--seed',str(a.seed_base+attempt*1009),'--trajectory-servo','--smooth-commands',
          '--arm-jerk','1200','--arm-tracking-rate','40','--contact-guard',
          '--native-retries','2','--checkpoint-interval',str(a.checkpoint_interval),'--max-seconds',str(a.end),'--port','8100']
        if a.finger_warm_seed:command.append('--accepted-finger-seed')
        if a.contact_prior:command.append('--contact-prior')
        with (ROOT/(name+'.log')).open('w') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        result_path=ROOT/'physics'/name/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {'completed':False,'failure':'worker exited without result','exit_code':code,'time_s':frame/150}
        run=ROOT/'physics'/name
        trace=np.load(run/'native_trajectory.npz') if (run/'native_trajectory.npz').exists() else None
        quality_end=stability(model,trace,ref,float(result['time_s']),width=.4) if trace is not None else {'stable':False}
        accepted=bool(result.get('completed') and result.get('grasp_success') and quality_end['stable'])
        entry={'run':name,'from_s':frame/150,'seed':a.seed_base+attempt*1009,'result':result,'end_stability':quality_end,'bridge_accepted':accepted,'checkpoint_stability':quality}
        ledger.append(entry);ledger_path.write_text(json.dumps(ledger,indent=2))
        print(json.dumps({'run':name,'accepted':accepted,'end_s':result['time_s'],'reason':result.get('failure')}),flush=True)
        if accepted:
            (ROOT/(a.tag+'_selected.json')).write_text(json.dumps(entry,indent=2));return
        sources.insert(0,name)
        before=max(frame/150,float(result['time_s'])-.4)
    print('No branch passed the full bridge acceptance test; failed branches and checkpoints retained.',flush=True)

if __name__=='__main__':main()
