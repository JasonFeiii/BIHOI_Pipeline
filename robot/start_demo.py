"""Start the bowl demo physics simulation from 0 s and open a Viser viewer.

    python start_demo.py [--gpu 0] [--port 8148] [--seed 63575] [--end 24.5]

Each launch writes to a new empty directory under outputs/robot/demo_runs/, so earlier results
are never touched and the supervisor really starts from 0 s (fresh_start only
applies to an empty root). Ctrl+C stops the viewer and gracefully stops the
simulation (SIGTERM: the supervisor terminates the worker and saves its state).
"""
import argparse,json,os,shutil,signal,subprocess,sys,threading,time
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sim'))
from robot_paths import DATA,OUTPUTS,PYTHON
PHYS_PY=PYTHON
CODE=HERE/'sim'
PYTHONPATH=[CODE]
DEMO_DATA=DATA/'demo'
INPUT_ROOT=DEMO_DATA/'plan'
INPUTS=('sqp_result_001.json','sqp_trajectory_001.npz','reference_info.json')
SCENE='sqp_scene_001.xml'  # template: @DEMO_DATA@ is replaced with the absolute data path
RAW_SHARPA=DEMO_DATA/'physics_contact_bowl'/'sharpa'/'bimanual'

def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu',default='0',help='CUDA device for the simulation')
    p.add_argument('--port',type=int,default=8148,help='Viser port')
    p.add_argument('--seed',type=int,default=63575)
    p.add_argument('--end',type=float,default=24.5,help='simulate until this time (s)')
    p.add_argument('--run-dir',type=Path,help='new empty output directory (default: outputs/robot/demo_runs/<timestamp>)')
    return p.parse_args()

def ensure_env(gpu):
    """Re-exec under physics-env with the simulation environment if needed."""
    path=':'.join(map(str,PYTHONPATH))
    if os.path.realpath(sys.executable)==os.path.realpath(PHYS_PY) and os.environ.get('PYTHONPATH')==path:return
    env=dict(os.environ,PYTHONPATH=path,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MUJOCO_GL='disable')
    os.execve(PHYS_PY,[PHYS_PY,'-Bu',str(Path(__file__).resolve()),*sys.argv[1:]],env)

def prepare(run_dir):
    if run_dir.exists() and any(run_dir.iterdir()):sys.exit(f'{run_dir} is not empty; fresh start needs an empty directory')
    (run_dir/'physics').mkdir(parents=True,exist_ok=True)
    for name in INPUTS:shutil.copy2(INPUT_ROOT/name,run_dir/name)
    (run_dir/SCENE).write_text((INPUT_ROOT/SCENE).read_text().replace('@DEMO_DATA@',str(DEMO_DATA)))

def start_sim(run_dir,a):
    env=dict(os.environ,BIHOI_ARM_ROOT=str(run_dir))
    cmd=[PHYS_PY,'-Bu',str(CODE/'run_from_zero.py'),'--fresh-start','--seed',str(a.seed),'--end',str(a.end)]
    (run_dir/'start_demo.json').write_text(json.dumps(dict(command=cmd,gpu=a.gpu,port=a.port,started=datetime.now().isoformat(timespec='seconds')),indent=2))
    log=(run_dir/'supervisor.log').open('w')
    # Own session: Ctrl+C reaches only this launcher, which then stops the sim gracefully.
    return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)

def stop_sim(sim,timeout=300):
    if sim.poll() is not None:return
    print(f'\nStopping simulation (pid {sim.pid}); the supervisor saves its state first...',flush=True)
    sim.send_signal(signal.SIGTERM)
    try:sim.wait(timeout)
    except subprocess.TimeoutExpired:
        print('Supervisor did not exit in time; killing it.',flush=True);os.killpg(sim.pid,signal.SIGKILL);sim.wait()

def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def load_view_model(scene):
    """Lightweight visual model: keep inertias, drop non-colliding helper geoms."""
    import xml.etree.ElementTree as ET
    import mujoco
    original=mujoco.MjModel.from_xml_path(str(scene))
    root=ET.parse(scene).getroot()
    for body in root.iter('body'):
        if body.get('name') and body.find('inertial') is None:
            bid=original.body(body.get('name')).id
            if original.body_mass[bid]>0:
                ET.SubElement(body,'inertial',mass=str(original.body_mass[bid]),pos=' '.join(map(str,original.body_ipos[bid])),quat=' '.join(map(str,original.body_iquat[bid])),diaginertia=' '.join(map(str,original.body_inertia[bid])))
    for parent in root.iter():
        for child in list(parent):
            if child.tag=='geom' and child.get('name') and original.geom_contype[original.geom(child.get('name')).id]==0:parent.remove(child)
    spec=mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'));model=spec.compile();data=mujoco.MjData(model)
    for body in spec.bodies:
        for g in body.geoms:
            if g.name:
                i=model.geom(g.name).id;g.type=mujoco.mjtGeom(int(model.geom_type[i]));g.group=int(model.geom_group[i])
    return spec,model,data

def run_viewer(run_dir,port,sim,log='supervisor.log'):
    import numpy as np,mujoco
    from retargeting.utils import viser_viewer as live
    from trajectory_store import TrajectoryReader
    from reference_overlay import ReferenceOverlay

    scene=run_dir/SCENE;spec,model,data=load_view_model(scene)
    live.init_viser(f'Bowl demo · {run_dir.name}',port=port);server=live._get_server()
    server.gui.add_markdown(f'**从 0 秒开始的新仿真**\n\n输出目录：`{run_dir}`\n\n蓝色为原始 retarget reference。')
    with server.gui.add_folder('计算进度'):
        compute_info=server.gui.add_markdown('等待仿真启动…')
        sequence_bar=server.gui.add_progress_bar(0,color='green')
    follow=server.gui.add_checkbox('跟随最新计算',initial_value=True)
    frame=server.gui.add_slider('回放帧',min=0,max=1,step=1,initial_value=0)
    play=server.gui.add_checkbox('循环回放',initial_value=True)
    speed=server.gui.add_slider('播放倍率',min=.1,max=2.,step=.1,initial_value=1.)
    restart=server.gui.add_button('从头播放')
    info=server.gui.add_markdown('等待第一段物理轨迹…')
    handles=live.build_and_log_scene_from_spec(spec,model,xml_path=scene,build_ref=False)
    live._STATE.scene_checkboxes['collision_meshes'].value=True
    task=json.loads((run_dir/'reference_info.json').read_text())['task']
    reference=ReferenceOverlay(server,live,RAW_SHARPA/task/'0'/'scene.xml',RAW_SHARPA/task/'0'/'trajectory_kinematic.npz')
    @server.on_client_connect
    def _(client):
        client.camera.position=(-.65,-.65,.55);client.camera.look_at=(.15,-.1,.18);client.camera.up_direction=(0,0,1)
    reset=[False]
    @restart.on_click
    def _(_event):follow.value=False;play.value=True;reset[0]=True

    # Disk reads stay in this thread; the render loop only consumes snapshots.
    snapshot={};lock=threading.Lock();reader=TrajectoryReader()
    def load():
        stamp=None
        while True:
            try:
                control=read_json(run_dir/'continuous_status.json');status=read_json(run_dir/'live_status.json')
                name=status.get('run') or control.get('run')
                if name:
                    rd=run_dir/'physics'/name;mf=rd/'trajectory_manifest.json';history=mf if mf.exists() else rd/'trajectory.npz'
                    new=(name,history.stat().st_mtime_ns);update=dict(control=control,progress=read_json(run_dir/'compute_progress.json'))
                    if new!=stamp:
                        arr=reader.read(rd,'display')
                        with np.load(rd/'reference.npz') as z:
                            ref_time=z['time'].copy();source=z['source_frame'].copy() if 'source_frame' in z else ref_time*30
                        update.update(stamp=new,run=name,qpos=arr['qpos'].copy(),time=arr['time'].copy(),ref_time=ref_time,source_map=source);stamp=new
                    with lock:snapshot.update(update)
            except (OSError,ValueError,KeyError,EOFError):pass  # files are mid-write or not created yet
            time.sleep(1.)
    threading.Thread(target=load,daemon=True).start()

    labels={'precontact_approach':'接触前连续接近','first_contact':'首次接触，切换 MPC','preparing':'恢复检查点','loading_physics':'加载物理模型','sampling':'生成候选','ik_projection':'机械臂 IK','rollout':'物理预测','scoring':'候选评分','native_check':'原生碰撞检查','executing':'物理执行','accepted':'已保存本段','saving':'保存检查点','resampling':'局部重新采样','completed':'计算结束','paused':'已暂停'}
    url=f'http://localhost:{server.get_port()}' if hasattr(server,'get_port') else f'http://localhost:{port}'
    print(f'Viser: {url}   (Ctrl+C to stop simulation and viewer)',flush=True)
    stamp=None;position=0.;last=-1;clock=time.monotonic();next_render=clock;next_panel=clock;sim_done=False
    while True:
        now=time.monotonic()
        if not sim_done and sim.poll() is not None:
            sim_done=True;print(f'Simulation exited (code {sim.returncode}); see {run_dir/log}. Viewer keeps running for playback.',flush=True)
        with lock:entry=dict(snapshot)
        if 'time' not in entry:
            if now>=next_panel:
                state=entry.get('control',{}).get('state') or ('已退出' if sim_done else '启动中')
                compute_info.content=f'**{labels.get(state,state)}**\n\n还没有已接受的物理轨迹。';next_panel=now+.5
            clock=now;time.sleep(.03);continue
        ts=entry['time']
        if entry['stamp']!=stamp:
            stamp=entry['stamp'];reference.reference_time=entry['ref_time'];reference.source_map=entry['source_map'];frame.max=max(1,len(ts)-1)
        dt=now-clock;clock=now
        if reset[0]:reset[0]=False;position=float(ts[0])
        elif int(frame.value)!=last:position=float(ts[min(int(frame.value),len(ts)-1)])
        if follow.value:position=float(ts[-1])
        elif play.value:
            position+=dt*float(speed.value)
            if position>ts[-1]+1/30:position=ts[0]+(position-ts[0])%(ts[-1]-ts[0]+1/30)
        index=int(np.clip(np.searchsorted(ts,position,side='right')-1,0,len(ts)-1))
        if now>=next_render:
            frame.value=index;last=index
            data.qpos[:]=entry['qpos'][index];mujoco.mj_kinematics(model,data)  # playback is kinematics only
            with server.atomic():
                live.log_frame(data,float(ts[index]),handles,record=False);reference.render(float(ts[index]))
            next_render=max(next_render+1/30,time.monotonic())
        if now>=next_panel:
            control=entry['control'];progress=entry['progress'];end=float(ts[-1]);target=float(control.get('requested_end_s',end))
            phase=progress.get('phase',control.get('state','preparing')) if progress.get('run')==entry['run'] else control.get('state','preparing')
            if sim_done and control.get('state') not in ('completed','paused'):phase='已退出'
            compute_info.content=f'**{labels.get(phase,phase)}**\n\n已接受 **{end:.2f} / {target:.2f} 秒** · 当前运行 {entry["run"]}'
            sequence_bar.value=100*min(1,end/max(target,.001))
            info.content=f'播放 **{ts[index]:.2f} 秒** / {end:.2f} 秒'
            next_panel=now+.5
        time.sleep(max(.001,min(.005,next_render-time.monotonic())))

def main():
    a=parse();ensure_env(a.gpu)
    run_dir=(a.run_dir or OUTPUTS/'demo_runs'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    prepare(run_dir)
    sim=start_sim(run_dir,a)
    print(f'Simulation started from 0 s on GPU {a.gpu} (pid {sim.pid})\n  output: {run_dir}\n  log:    {run_dir/"supervisor.log"}',flush=True)
    try:run_viewer(run_dir,a.port,sim)
    except KeyboardInterrupt:pass
    finally:stop_sim(sim)
    status=read_json(run_dir/'continuous_status.json')
    print(f'Stopped. state={status.get("state","?")} run={status.get("run","?")} reached={status.get("best_time_s","?")} s',flush=True)

if __name__=='__main__':main()
