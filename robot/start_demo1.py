"""Start the SPIDER-style arm + hand simulation from 0 s and open a Viser viewer.

    python start_demo1.py [--gpu 0] [--port 8148] [--seed 63575] [--end 24.5] [--samples 2048] [--iterations 16]
                          [--execute mean|best] [--fallback none|stable|window]
                          [--resume-from RUN_OR_TRAJECTORY --start 10]

Same robot (dual UR3e + Sharpa hands, wrist samples projected through arm IK),
scene and viewer as start_demo.py, but the optimizer is sim/spider_arm_mpc.py:
2048 samples, <= 16 iterations with SPIDER early stopping, 1.6 s horizon /
0.4 s execution, sim_dt 0.01, SPIDER tracking reward, soft constraints only,
weighted-mean execution (--execute best runs the best rollout). When the object
drops 5 cm below the reference, --fallback stable restarts from the latest
moment a hand held it stably; --fallback window redoes the 0.4 s window
with drop-aware sampling (see sim/spider_arm_mpc.py). There is no checkpoint-recovery supervisor: one run
goes from 0 s to --end.

Each launch writes to a new directory outputs/robot/demo_runs/spider_arm_<timestamp>/.
Ctrl+C stops the viewer and the simulation (it finishes the current iteration, then saves).
"""
import argparse,json,os,signal,subprocess,sys
from datetime import datetime
from pathlib import Path
from start_demo import OUTPUTS,PHYS_PY,CODE,PYTHONPATH,prepare,read_json,run_viewer

def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu',default='0',help='CUDA device for the simulation')
    p.add_argument('--port',type=int,default=8148,help='Viser port')
    p.add_argument('--seed',type=int,default=63575)
    p.add_argument('--end',type=float,default=24.5,help='simulate until this time (s)')
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--execute',choices=('mean','best'),default='mean',help='execute the weighted mean (SPIDER) or the best rollout')
    p.add_argument('--fallback',choices=('none','stable','window'),default='none',help='what to do when the object falls (see sim/spider_arm_mpc.py)')
    p.add_argument('--resume-from',type=Path,help='earlier run dir (demo_runs/spider_arm_*) or its trajectory.npz to continue from')
    p.add_argument('--start',type=float,default=0.,help='with --resume-from: continue from this time (s)')
    p.add_argument('--run-dir',type=Path,help='new empty output directory (default: outputs/robot/demo_runs/spider_arm_<timestamp>)')
    return p.parse_args()

def ensure_env(gpu):
    """Re-exec this script under physics-env with the simulation environment if needed."""
    path=':'.join(map(str,PYTHONPATH))
    if os.path.realpath(sys.executable)==os.path.realpath(PHYS_PY) and os.environ.get('PYTHONPATH')==path:return
    env=dict(os.environ,PYTHONPATH=path,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MUJOCO_GL='disable')
    os.execve(PHYS_PY,[PHYS_PY,'-Bu',str(Path(__file__).resolve()),*sys.argv[1:]],env)

def start_sim(run_dir,a):
    env=dict(os.environ,BIHOI_ARM_ROOT=str(run_dir))
    cmd=[PHYS_PY,'-Bu',str(CODE/'spider_arm_mpc.py'),'--end',str(a.end),'--seed',str(a.seed),
        '--samples',str(a.samples),'--iterations',str(a.iterations),'--execute',a.execute,'--fallback',a.fallback]
    if a.resume_from:
        source=a.resume_from.resolve()
        if source.is_dir():source=source/'physics'/'spider_arm_0001'/'trajectory.npz'
        cmd+=['--resume-from',str(source),'--start',str(a.start)]
    (run_dir/'start_demo1.json').write_text(json.dumps(dict(command=cmd,gpu=a.gpu,port=a.port,started=datetime.now().isoformat(timespec='seconds')),indent=2))
    log=(run_dir/'sim.log').open('w')
    # Own session: Ctrl+C reaches only this launcher, which then stops the sim gracefully.
    return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)

def stop_sim(sim,timeout=600):
    if sim.poll() is not None:return
    print(f'\nStopping simulation (pid {sim.pid}); it finishes the current iteration and saves first...',flush=True)
    sim.send_signal(signal.SIGTERM)
    try:sim.wait(timeout)
    except subprocess.TimeoutExpired:
        print('Simulation did not exit in time; killing it.',flush=True);os.killpg(sim.pid,signal.SIGKILL);sim.wait()

def main():
    a=parse();ensure_env(a.gpu)
    run_dir=(a.run_dir or OUTPUTS/'demo_runs'/('spider_arm_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    prepare(run_dir)
    sim=start_sim(run_dir,a)
    print(f'SPIDER-style arm simulation started from 0 s on GPU {a.gpu} (pid {sim.pid})\n  output: {run_dir}\n  log:    {run_dir/"sim.log"}',flush=True)
    try:run_viewer(run_dir,a.port,sim,log='sim.log')
    except KeyboardInterrupt:pass
    finally:stop_sim(sim)
    status=read_json(run_dir/'continuous_status.json')
    print(f'Stopped. state={status.get("state","?")} reached={status.get("best_time_s","?")} s',flush=True)

if __name__=='__main__':main()
