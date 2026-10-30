"""Autonomous full-sequence regression with verified physical checkpoint recovery."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback,shutil
from pathlib import Path
import numpy as np

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def command(code,name,seed,end,resume=None):
    cmd=[sys.executable,'-u',str(code/'arm_mpc.py'),'--name',name,'--plan-result','sqp_result_001.json',
      '--from-open-hand','--hold-seconds','0','--continue-on-task-failure','--grasp-transport','--contact-prior',
      '--native-screen','--native-shortlist','12','--samples','96','--iterations','4','--horizon','1.2','--ctrl-dt','.2',
      '--arm-jerk','1200','--arm-tracking-rate','40','--checkpoint-interval','.2','--native-retries','2',
      '--headless-worker','--max-seconds',str(end),'--seed',str(seed)]
    if resume:cmd+=['--resume-from',resume]
    return cmd

def tracking(run,start=0.):
    from motion_audit import rotation_progress,first_stalled_motion
    x=np.load(run/'native_trajectory.npz');ref=np.load(run/'reference.npz');ids=np.flatnonzero(x['time']>=start-1e-7)
    if len(ids)<2:return dict(time_s=float(x['time'][-1]),rotation=None,first_stall=None)
    ri=np.clip(np.rint(x['time'][ids]*150).astype(int),0,len(ref['qpos'])-1)
    actual=x['qpos'][ids,-11:-7];target=ref['qpos'][ri,-11:-7]
    return dict(time_s=float(x['time'][-1]),rotation=rotation_progress(actual,target),first_stall=first_stalled_motion(x['time'][ids],actual,target),
      max_position_error_m=float(np.linalg.norm(x['qpos'][ids,-14:-11]-ref['qpos'][ri,-14:-11],axis=-1).max()),
      normal_gravity=bool(np.allclose(x['gravity'],[0,0,-9.81])),no_assistance=bool(not np.any(x['object_weld_active']) and not np.any(x['object_external_force'])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh-start',action='store_true');parser.add_argument('--seed',type=int,required=True);parser.add_argument('--end',type=float,default=24.5);a=parser.parse_args()
    from build_arm_scene import ROOT,OLD
    from trajectory_evidence import Evidence
    from recover_contact import copy_prefix
    from recovery_beam import RecoveryBeam
    from lineage import ancestors
    import trimesh
    code=Path(__file__).parent;ROOT.mkdir(exist_ok=True);(ROOT/'physics').mkdir(exist_ok=True)
    lock=(ROOT/'continuous.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=ROOT/'from_zero_state.json';ledgerpath=ROOT/'from_zero_ledger.json';statuspath=ROOT/'continuous_status.json'
    ev=Evidence(trimesh.load(OLD/'assets/objects/bowl/source_mesh.ply',process=False).convex_hull.vertices)
    stored=json.loads(statepath.read_text()) if statepath.exists() else {};beam=RecoveryBeam.load(stored.get('nodes',[]),memory=stored.get('failure_memory',{}));parents=stored.get('parents',{})
    # Revalidate legacy frontier nodes against the repaired resume boundary.
    for key,node in list(beam.nodes.items()):
        if ev.inspect(ROOT/'physics'/node.run,Path(node.checkpoint),0.) is None:
            del beam.nodes[key]
    ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else [];stop=[False]
    recovery_floor_s=float(stored.get('recovery_floor_s',.2 if a.fresh_start else 7.))
    fresh_pending=bool(a.fresh_start and not stored and not ledger and not any((ROOT/'physics').iterdir()))
    resume_once_checkpoint=stored.get('resume_once_checkpoint')
    if not beam.nodes and not fresh_pending:raise RuntimeError('No verified recovery checkpoint; automatic restart from zero forbidden')
    def signal_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    def persist():write(statepath,dict(nodes=beam.dump(),failure_memory=beam.memory,parents=parents,updated_unix=time.time(),initial_seed=a.seed,recovery_floor_s=recovery_floor_s))
    def notify(**kw):write(statuspath,dict(supervisor_pid=os.getpid(),updated_unix=time.time(),requested_end_s=a.end,automatic_retry=True,**kw))
    while not stop[0] and not (ROOT/'STOP_CONTINUOUS').exists():
        index=len(ledger)+1;name=f'from_zero_{index:04d}'
        while (ROOT/'physics'/name).exists():index+=1;name=f'from_zero_{index:04d}'
        source=None;start=0.;resume=None
        if fresh_pending:
            fresh_pending=False
            persist()  # One-shot marker survives crashes before the first checkpoint.
        else:
            allowed=[n for n in beam.nodes.values() if n.frame/150>=recovery_floor_s-1e-7]
            if not allowed:raise RuntimeError('No safe local resume state; refusing initial restart')
            if resume_once_checkpoint:
                ck=Path(resume_once_checkpoint);verified=ev.inspect(ck.parent,ck,0.)
                if verified is None:raise RuntimeError('Requested local boundary checkpoint did not pass verification')
                source=verified[0];beam.nodes[source.key]=source;resume_once_checkpoint=None
            else:
                source=max(allowed,key=beam._rank)
                from frontier_backoff import from_history
                policy=from_history(ROOT,ledger)
                if policy:
                    upper=policy['max_recovery_time_s'];lower=max(recovery_floor_s,upper-.2)
                    nearby=[n for n in allowed if lower-1e-7<=n.frame/150<=upper+1e-7]
                    # Import a local alternative on demand instead of allowing the
                    # fixed-size beam to force a jump back to the oldest prefix.
                    choices=[]
                    for src in ancestors(ROOT,ledger[-1]['run'],parents):
                        for ck in (ROOT/'physics'/src).glob('checkpoint_*.npz'):
                            try:
                                with np.load(ck) as z:frame=int(z['sim'])
                                if lower-1e-7<=frame/150<=upper+1e-7:choices.append((frame,src,ck))
                            except (OSError,ValueError,KeyError,EOFError):continue
                    for frame,src,ck in sorted(choices,reverse=True):
                        verified=ev.inspect(ROOT/'physics'/src,ck,0.)
                        if verified:
                            node=verified[0];beam._restore_history(node);nearby.append(node);break
                    if nearby:
                        source=max(nearby,key=beam._rank);beam.nodes[source.key]=source
                    policy.update(applied=bool(nearby),selected_run=source.run,selected_time_s=source.frame/150)
                    write(ROOT/'local_recovery_policy.json',policy)
                else:write(ROOT/'local_recovery_policy.json',dict(applied=False,reason='No repeated immediate-collision evidence',selected_run=source.run,selected_time_s=source.frame/150))
            start=source.frame/150;resume=None
            if source:
                resume=name+'_prefix';copy_prefix(ROOT/'physics'/source.run,Path(source.checkpoint),ROOT/'physics'/resume,source.frame)
        run=ROOT/'physics'/name;run.mkdir()
        if resume:
            for filename in ('trajectory.npz','native_trajectory.npz'):
                shutil.copy2(ROOT/'physics'/resume/filename,run/filename)
            shutil.copy2(ROOT/'physics'/source.run/'reference.npz',run/'reference.npz')
            write(ROOT/'live_status.json',dict(run=name,state='running',phase='restoring_checkpoint',time_s=start))
        cmd=command(code,name,a.seed+1009*(index-1),a.end,resume)
        write(run/'launch.json',dict(command=cmd,from_s=start,source=source.run if source else None,seed=a.seed+1009*(index-1)))
        with (ROOT/(name+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            notify(state='running',run=name,child_pid=child.pid,from_s=start,best_time_s=start,search_scale=1.)
            next_audit=0.;last_time=start;last_advance=time.monotonic();sent_stop=False
            while child.poll() is None:
                if stop[0] or (ROOT/'STOP_CONTINUOUS').exists():
                    if not sent_stop:child.terminate();sent_stop=True
                now=time.monotonic()
                if now>=next_audit and (run/'native_trajectory.npz').exists():
                    try:
                        audit=tracking(run);write(ROOT/'pipeline_diagnostics.json',dict(run=name,**audit))
                        if audit['time_s']>last_time:last_time=audit['time_s'];last_advance=now
                        notify(state='running',run=name,child_pid=child.pid,from_s=start,best_time_s=audit['time_s'],search_scale=1.)
                    except (OSError,ValueError,KeyError,EOFError):pass
                    next_audit=now+15
                if now-last_advance>1800 and not sent_stop:child.terminate();sent_stop=True
                time.sleep(2)
        result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else dict(failure='worker exited without result',time_s=start)
        failure=result.get('failure');end=float(result.get('time_s',start));audit=None
        if (run/'native_trajectory.npz').exists():
            try:audit=tracking(run);write(run/'execution_audit.json',audit)
            except (OSError,ValueError,KeyError,EOFError):traceback.print_exc()
        task_ok=bool(result.get('completed') and end>=a.end-1e-6 and audit and audit['normal_gravity'] and audit['no_assistance'] and audit['rotation'] and (audit['rotation']['progress_ratio'] or 0)>=.65 and audit['max_position_error_m']<=.10)
        # This is a declared regression screen, not a proof of physical task success.
        selection_end=end
        if result.get('completed') and not task_ok:
            failure='Completed timeline but action-tracking regression screen failed'
            # Keep a completed comparison intact for review; no early rewind.
        if source and failure:beam.record_failure(source.key,'task_tracking' if result.get('completed') else 'collision_or_control')
        if source:parents[name]=source.run
        added=[]
        # Prefer latest safe state and earlier alternatives. Provenance is
        # followed across handoffs; all physical histories remain on disk.
        sources=ancestors(ROOT,name,parents)
        targets=[selection_end,selection_end-.2,selection_end-.4,selection_end-.8,selection_end-1.2,0.]
        for target in targets:
            if target<recovery_floor_s-1e-7:continue
            choices=[]
            for src in sources:
                for ck in (ROOT/'physics'/src).glob('checkpoint_*.npz'):
                    try:
                        with np.load(ck) as z:frame=int(z['sim'])
                        if recovery_floor_s-1e-7<=frame/150<=target+1e-7:choices.append((frame,src,ck))
                    except (OSError,ValueError,KeyError,EOFError):continue
            for frame,src,ck in sorted(choices,reverse=True):
                try:ins=ev.inspect(ROOT/'physics'/src,ck,0.)
                except (OSError,ValueError,KeyError,EOFError):continue
                if ins:
                    node,evidence=ins
                    if source:node.parent=source.key
                    if beam.add(node):added.append(dict(run=src,time_s=frame/150,evidence=evidence))
                    break
        entry=dict(run=name,from_s=start,to_s=end,failure=failure,exit_code=child.returncode,audit=audit,regression_screen_passed=task_ok,added_checkpoints=added)
        ledger.append(entry);write(ledgerpath,ledger);persist()
        if stop[0] or (ROOT/'STOP_CONTINUOUS').exists():notify(state='paused',run=name,best_time_s=end);break
        if result.get('completed'):
            write(ROOT/'completion.json',dict(run=name,regression_screen_passed=task_ok,physical_success=None,audit=audit,criteria={'min_rotation_progress_ratio':.65,'max_position_error_m':.10}))
            notify(state='completed_awaiting_review',run=name,best_time_s=end,physical_success=None);break
        notify(state='resampling',run=name,best_time_s=end,reason=failure,active_branches=len(beam.nodes))
        if not beam.nodes:
            notify(state='needs_review',run=name,best_time_s=end,reason='No independently verified local checkpoint; initial restart forbidden')
            break

if __name__=='__main__':main()
