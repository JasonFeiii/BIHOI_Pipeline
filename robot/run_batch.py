"""Simulate the best HOCap take of every object, many at once on one GPU.

    python run_batch.py [--end 15] [--parallel 4] [--objects Bottle_0820_fwd ...] [--batch-dir DIR]
    python run_batch.py --tasks fwd_capture__0818_fwd_04__take05   # specific takes, ranked or not
    python run_batch.py --tasks <take> --resume-from <run>/physics/spider_arm_0001/trajectory.npz --start 3 --obj-weight-after-contact 5

Takes come from data/tasks/take_ranking.json (sim/rank_takes.py). For each
object the best take is prepared with sim/prepare_task.py (if not done yet);
when preparation fails, the next take in its ranking is used. Each object then
runs sim/spider_arm_mpc.py from 0 s to --end (HOCap time; the simulation adds the
task's still lead-in, see sim/prepare_task.py) with the start_demo1.py settings
(2048 samples, <= 16 iterations, weighted-mean execution, no fallback, no
drop-aware sampling), at most --parallel simulations at a time.

Output: outputs/robot/demo_runs/batch_<timestamp>/<object>/ (same layout as start_demo1.py runs,
log in sim.log) and batch_status.json.

A Viser page (batch_viewer.py) starts with the batch at http://localhost:<--port>
(the next free port if it is taken; --no-viewer to skip) and plays every
simulation as it runs. When all simulations are done it keeps playing until
Ctrl+C. Ctrl+C / SIGTERM stops every simulation gracefully (each finishes its
current iteration and saves) and the viewer.
"""
import argparse,json,os,shutil,signal,socket,subprocess,sys,time
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent;CODE=HERE/'sim'
sys.path.insert(0,str(CODE))
from robot_paths import DATA,OUTPUTS,PYTHON
TASKS=DATA/'tasks';PHYS_PY=PYTHON
PLAN_FILES=('sqp_result_001.json','sqp_scene_001.xml','reference_info.json')

def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def dump(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def prepared(task):
    """Prepare the task if needed; True when its plan is ready."""
    if read_json(TASKS/task/'prepare.json').get('stage')=='ready':return True
    env=dict(os.environ,PYTHONPATH=str(CODE),OMP_NUM_THREADS='4')
    return subprocess.call([PHYS_PY,'-u',str(CODE/'prepare_task.py'),'--task',task],env=env,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)==0

def launch(obj,task,run_dir,a):
    plan=TASKS/task/'plan';run_dir.mkdir(parents=True)
    result=read_json(plan/'sqp_result_001.json')
    for name in (*PLAN_FILES,result['trajectory_file']):shutil.copy2(plan/name,run_dir/name)
    # --end is HOCap time; the simulation timeline starts with the task's still lead-in (prepare_task.py).
    lead_in=read_json(TASKS/task/'contact'/'trial_accepted_v2'/'schedule.json').get('lead_in_s',0.)
    cmd=[PHYS_PY,'-Bu',str(CODE/'spider_arm_mpc.py'),'--end',str(round(a.end+lead_in,4)),'--seed',str(a.seed),
        '--samples',str(a.samples),'--iterations',str(a.iterations),'--execute','mean','--fallback','none',
        '--w-obj-pos',str(a.w_obj_pos),'--w-obj-rot',str(a.w_obj_rot)]
    if a.obj_weight_after_contact is not None:cmd+=['--obj-weight-after-contact',str(a.obj_weight_after_contact)]
    if a.resume_from:cmd+=['--resume-from',str(a.resume_from.resolve()),'--start',str(a.start)]
    env=dict(os.environ,BIHOI_ARM_ROOT=str(run_dir),BIHOI_TASK=task,BIHOI_CONTACT_ROOT=str(TASKS/task/'contact'),
        PYTHONPATH=str(CODE),CUDA_VISIBLE_DEVICES=a.gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MUJOCO_GL='disable',
        TORCHINDUCTOR_COMPILE_THREADS='1')  # one compile worker, not one per core, for each of the parallel runs
    dump(run_dir/'batch_job.json',dict(object=obj,task=task,command=cmd,lead_in_s=lead_in,started=datetime.now().isoformat(timespec='seconds')))
    log=(run_dir/'sim.log').open('w')
    return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)

def free_port(port):
    """`port` or the next one that nothing listens on."""
    while True:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:probe.bind(('',port));return port
            except OSError:port+=1

def start_viewer(batch,port):
    log=(batch/'viewer.log').open('a')
    return subprocess.Popen([PHYS_PY,'-Bu',str(HERE/'batch_viewer.py'),'--batch-dir',str(batch),'--port',str(port)],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--end',type=float,default=15.);p.add_argument('--parallel',type=int,default=4,help='simulations at a time (each needs ~18 GB of GPU memory)')
    p.add_argument('--samples',type=int,default=2048);p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--seed',type=int,default=63575);p.add_argument('--gpu',default='0')
    p.add_argument('--objects',nargs='*',help='only these objects (default: all)')
    p.add_argument('--tasks',nargs='*',help='run exactly these takes (e.g. fwd_capture__0818_fwd_04__take05), ranked or not')
    p.add_argument('--batch-dir',type=Path)
    p.add_argument('--w-obj-pos',type=float,default=1.,help='object position reward weight (spider_arm_mpc.py)')
    p.add_argument('--obj-weight-after-contact',type=float,help='object position weight from the human\'s first contact on (spider_arm_mpc.py)')
    p.add_argument('--w-obj-rot',type=float,default=.3,help='object rotation reward weight (spider_arm_mpc.py)')
    p.add_argument('--resume-from',type=Path,help='with one --tasks take: trajectory.npz of an earlier run to continue from at --start')
    p.add_argument('--start',type=float,default=0.,help='with --resume-from: simulation time (s) to continue from')
    p.add_argument('--port',type=int,default=8150,help='Viser port of the live viewer')
    p.add_argument('--no-viewer',action='store_true',help='do not start the Viser viewer')
    a=p.parse_args()
    ranking=read_json(TASKS/'take_ranking.json')['objects']
    objects=[o for o in ranking if ranking[o]['order'] and (not a.objects or o in a.objects)]
    batch=(a.batch_dir or OUTPUTS/'demo_runs'/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve();batch.mkdir(parents=True,exist_ok=True)
    jobs={o:dict(object=o,state='queued',candidates=list(ranking[o]['order'])) for o in objects}
    if a.tasks:jobs={t.split('__')[1]+'__'+t.split('__')[2]:dict(object=t.split('__')[1]+'__'+t.split('__')[2],state='queued',candidates=[t]) for t in a.tasks}
    running={};stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def status():
        for job in jobs.values():
            if job['state']=='running':
                progress=read_json(batch/job['object']/'continuous_status.json');job['time_s']=progress.get('best_time_s',0.)
        dump(batch/'batch_status.json',dict(end_s=a.end,parallel=a.parallel,jobs=list(jobs.values())))
    print(f'batch {batch}: {len(jobs)} jobs, {a.parallel} in parallel, 0 -> {a.end} s',flush=True)
    status();viewer=None
    if not a.no_viewer:
        port=free_port(a.port);viewer=start_viewer(batch,port)
        print(f'viewer: http://localhost:{port}  (log {batch/"viewer.log"}; a run appears once its simulation has started)',flush=True)
    while not stop[0]:
        for obj,proc in list(running.items()):
            if proc.poll() is None:continue
            del running[obj];job=jobs[obj];result=read_json(batch/obj/'physics'/'spider_arm_0001'/'result.json')
            job.update(state=result.get('state','failed'),time_s=result.get('time_s'),failure=result.get('failure'),wall_s=result.get('wall_s'))
            print(f'{datetime.now():%H:%M:%S} {obj}: {job["state"]} at {job["time_s"]} s'+(f' ({job["failure"]})' if job.get('failure') else ''),flush=True)
        queued=[o for o,j in jobs.items() if j['state']=='queued']
        while queued and len(running)<a.parallel and not stop[0]:
            obj=queued.pop(0);job=jobs[obj];job['state']='preparing';status()
            while job['candidates'] and not prepared(job['candidates'][0]):
                job.setdefault('preparation_failed',[]).append(job['candidates'].pop(0))
            if not job['candidates']:job['state']='no prepared take';print(f'{obj}: no take could be prepared',flush=True);continue
            job.update(task=job['candidates'][0],state='running',started=datetime.now().isoformat(timespec='seconds'))
            running[obj]=launch(obj,job['task'],batch/obj,a)
            print(f'{datetime.now():%H:%M:%S} {obj}: started {job["task"]} (pid {running[obj].pid})',flush=True)
        status()
        if not running and not any(j['state']=='queued' for j in jobs.values()):break
        time.sleep(5)
    if running:
        print(f'stopping {len(running)} simulations (each finishes its iteration and saves)...',flush=True)
        for proc in running.values():proc.send_signal(signal.SIGTERM)
        for obj,proc in running.items():
            proc.wait();result=read_json(batch/obj/'physics'/'spider_arm_0001'/'result.json')
            jobs[obj].update(state=result.get('state','stopped'),time_s=result.get('time_s'))
    status();print(json.dumps({o:(j['state'],j.get('time_s')) for o,j in jobs.items()},indent=1),flush=True)
    if viewer is not None:
        if not stop[0] and viewer.poll() is None:
            print(f'all simulations done; the viewer keeps playing at http://localhost:{port} (Ctrl+C to exit)',flush=True)
            while not stop[0] and viewer.poll() is None:time.sleep(1)
        if viewer.poll() is None:
            viewer.terminate()
            try:viewer.wait(10)
            except subprocess.TimeoutExpired:viewer.kill()

if __name__=='__main__':main()
