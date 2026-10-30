"""Viser view of the SPIDER optimization inside a start_demo1.py run (read-only).

    python opt_viewer.py [RUN_DIR] [--port 8153]

RUN_DIR defaults to the newest demo_runs/spider_arm_*. Nothing is started or
stopped; the page only reads files the simulation writes, so it can be opened
and closed at any time while the simulation keeps running.

Shown per optimized window and iteration:
  * predicted right-wrist / left-wrist / object paths of the best samples
    (green = best .. yellow) and of evenly spaced samples (gray .. red by score),
    the reference (blue) and what was actually executed (white)
  * reward / improvement per iteration, score histogram of all samples,
    iterations and object error over time, live progress
Sample paths exist only for runs started after sim/spider_arm_mpc.py saved them
(physics/<run>/samples/); older runs show the statistics only.
"""
import argparse,json,os,sys,threading,time
from pathlib import Path
from start_demo import OUTPUTS,PHYS_PY,PYTHONPATH,SCENE,load_view_model,read_json

POINTS=('右手腕','左手腕','物体')
THRESHOLD=.01  # SPIDER improvement_threshold used by spider_arm_mpc.py

def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir',nargs='?',type=Path)
    p.add_argument('--port',type=int,default=8153)
    a=p.parse_args()
    if a.run_dir is None:
        runs=sorted((OUTPUTS/'demo_runs').glob('spider_arm_*'))
        if not runs:sys.exit('no demo_runs/spider_arm_* run found; pass RUN_DIR')
        a.run_dir=runs[-1]
    a.run_dir=a.run_dir.resolve()
    return a

def ensure_env():
    """Re-exec this script under physics-env (CPU only: MuJoCo kinematics for display)."""
    path=':'.join(map(str,PYTHONPATH))
    if os.path.realpath(sys.executable)==os.path.realpath(PHYS_PY) and os.environ.get('PYTHONPATH')==path:return
    env=dict(os.environ,PYTHONPATH=path,OPENBLAS_NUM_THREADS='1',MUJOCO_GL='disable')
    os.execve(PHYS_PY,[PHYS_PY,'-Bu',str(Path(__file__).resolve()),*sys.argv[1:]],env)

def read_log(path):
    rows=[]
    try:
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass  # last line may be mid-write
    except OSError:pass
    for i,row in enumerate(rows):row.setdefault('window',i+1)
    return rows

def segments(paths):
    """(K, T, 3) polylines -> (K*(T-1), 2, 3) line segments."""
    import numpy as np
    return np.stack([paths[:,:-1],paths[:,1:]],axis=2).reshape(-1,2,3)

def main():
    a=parse();ensure_env()
    import numpy as np,mujoco
    import plotly.graph_objects as go
    from retargeting.utils import viser_viewer as live

    run_dir=a.run_dir
    while not read_json(run_dir/'continuous_status.json').get('run'):
        print(f'waiting for {run_dir/"continuous_status.json"} ...',flush=True);time.sleep(2)
    run=run_dir/'physics'/read_json(run_dir/'continuous_status.json')['run']
    spec,model,data=load_view_model(run_dir/SCENE)
    wrist_bodies=[model.body(side+'_hand_C_MC').id for side in ('right','left')]

    live.init_viser(f'SPIDER 优化过程 · {run_dir.name}',port=a.port);server=live._get_server()
    server.gui.add_markdown(f'**SPIDER 采样优化过程**（只读，不影响仿真）\n\n`{run}`')
    with server.gui.add_folder('实时进度'):
        live_info=server.gui.add_markdown('读取中…')
    with server.gui.add_folder('优化窗口'):
        follow=server.gui.add_checkbox('跟随最新窗口（手动选窗口前先取消）',initial_value=True)
        window_slider=server.gui.add_slider('窗口',min=1,max=2,step=1,initial_value=1,disabled=True)
        iteration_slider=server.gui.add_slider('迭代',min=1,max=2,step=1,initial_value=1,disabled=True)
        show_top=server.gui.add_checkbox('得分最高的采样（绿→黄）',initial_value=True)
        show_uniform=server.gui.add_checkbox('均匀抽取的采样（灰→红=差）',initial_value=True)
        show_reference=server.gui.add_checkbox('参考轨迹（蓝）',initial_value=True)
        show_actual=server.gui.add_checkbox('实际执行（白）',initial_value=True)
        point_boxes=[server.gui.add_checkbox(name,initial_value=True) for name in POINTS]
        window_info=server.gui.add_markdown('')
    with server.gui.add_folder('曲线'):
        timeline_plot=server.gui.add_plotly(go.Figure(),aspect=1.6)
        iteration_plot=server.gui.add_plotly(go.Figure(),aspect=1.6)
        histogram_plot=server.gui.add_plotly(go.Figure(),aspect=1.6)
    handles=live.build_and_log_scene_from_spec(spec,model,xml_path=run_dir/SCENE,build_ref=False)
    live._STATE.scene_checkboxes['collision_meshes'].value=True
    @server.on_client_connect
    def _(client):
        client.camera.position=(-.65,-.65,.55);client.camera.look_at=(.15,-.1,.18);client.camera.up_direction=(0,0,1)

    layout=dict(margin=dict(l=40,r=10,t=30,b=30),height=260,template='plotly_white',legend=dict(orientation='h',y=-.25))
    state={'log':[],'trajectory':None,'traj_stamp':None,'samples':{},'dirty':True,'drawn':None}
    lock=threading.Lock()
    def load():
        while True:
            try:
                rows=read_log(run/'optimizer_log.jsonl')
                path=run/'trajectory.npz';stamp=path.stat().st_mtime_ns if path.exists() else None
                trajectory=None
                if stamp!=state['traj_stamp']:
                    with np.load(path) as z:trajectory=dict(time=z['time'].copy(),qpos=z['qpos'].copy())
                files={int(f.stem):f for f in (run/'samples').glob('*.npz')} if (run/'samples').exists() else {}
                with lock:
                    if len(rows)!=len(state['log']) or trajectory is not None or files.keys()!=state['samples'].keys():state['dirty']=True
                    state['log']=rows;state['samples']=files
                    if trajectory is not None:state['trajectory']=trajectory;state['traj_stamp']=stamp
            except (OSError,ValueError,KeyError,EOFError):pass  # files mid-write
            progress=read_json(run_dir/'compute_progress.json');status=read_json(run_dir/'continuous_status.json')
            live_info.content=(f"**{status.get('state','?')}** · 阶段 `{progress.get('phase','?')}`\n\n"
                f"已执行 {progress.get('accepted_time_s',0):.2f} / {status.get('requested_end_s','?')} 秒 · "
                f"迭代 {progress.get('iteration',0)}/{progress.get('iteration_total','?')} · "
                f"rollout {progress.get('rollout_step',0)}/{progress.get('rollout_total','?')} 步")
            time.sleep(1.)
    threading.Thread(target=load,daemon=True).start()
    for control in (window_slider,iteration_slider,show_top,show_uniform,show_reference,show_actual,*point_boxes):
        control.on_update(lambda _e:state.update(dirty=True))

    def sample_file(window):
        with lock:path=state['samples'].get(window)
        if path is None:return None
        try:
            with np.load(path) as z:return {k:z[k].copy() for k in z.files}
        except (OSError,ValueError,EOFError):return None

    def draw_timeline(rows):
        optimized=[r for r in rows if r.get('iterations')]
        fig=go.Figure(layout=dict(**layout,title='每个窗口：迭代次数 / 执行后物体误差',yaxis2=dict(overlaying='y',side='right',title='mm')))
        fig.add_bar(x=[r['time_s'] for r in optimized],y=[len(r['iterations']) for r in optimized],name='迭代次数',marker_color='#8fb3ff')
        fig.add_scatter(x=[r['end_s'] for r in rows],y=[r['object_error_m']*1000 for r in rows],name='物体误差 mm',yaxis='y2',mode='lines+markers',line_color='#d9534f')
        timeline_plot.figure=fig

    def draw_window(rows,sample):
        w=int(window_slider.value);row=next((r for r in rows if r['window']==w),None)
        if row is None:return
        its=row.get('iterations',[]);n=max(1,len(its))
        iteration_slider.max=max(2,n);iteration_slider.disabled=n<2
        if iteration_slider.value>n:iteration_slider.value=n
        i=int(iteration_slider.value)-1
        stop=('提前停止：改进 < 0.01' if its and its[-1]['improvement']<THRESHOLD else '达到迭代上限') if its else '接触前：不优化'
        window_info.content=(f"**窗口 {w}** · {row['time_s']:.2f} → {row['end_s']:.2f} 秒 · 耗时 {row.get('wall_s',0):.0f} 秒\n\n"
            f"{len(its)} 次迭代（{stop}）· 执行后物体误差 {row['object_error_m']*1000:.1f} mm"
            +('' if sample is not None or not its else '\n\n_这个窗口没有保存采样轨迹（运行启动于该功能加入之前）_'))
        fig=go.Figure(layout=dict(**layout,title=f'窗口 {w}：每轮迭代'))
        x=list(range(1,len(its)+1))
        fig.add_scatter(x=x,y=[r['rew_max'] for r in its],name='最高奖励',mode='lines+markers')
        fig.add_scatter(x=x,y=[r['rew_mean'] for r in its],name='平均奖励',mode='lines+markers')
        fig.add_scatter(x=x,y=[r['improvement'] for r in its],name='改进量',mode='lines+markers')
        fig.add_scatter(x=[.8,max(1,len(its))+.2],y=[THRESHOLD]*2,name='停止阈值 0.01',mode='lines',line=dict(dash='dash',color='gray'))
        iteration_plot.figure=fig
        hist=go.Figure(layout=dict(**layout,title=f'第 {i+1} 轮：全部采样的奖励分布'))
        if sample is not None and i<len(sample['all_scores']):
            scores=sample['all_scores'][i];scores=scores[np.isfinite(scores)]
            hist.add_histogram(x=scores,nbinsx=60,name='采样',marker_color='#999')
            cut=np.sort(scores)[-max(1,int(.1*len(scores)))]
            hist.add_scatter(x=[cut,cut],y=[0,len(scores)/10],mode='lines',name='前 10%（参与加权）',line=dict(color='green',dash='dash'))
        histogram_plot.figure=hist

    drawn={}
    def draw_scene(rows,sample,trajectory):
        w=int(window_slider.value);row=next((r for r in rows if r['window']==w),None)
        visible=np.array([b.value for b in point_boxes])
        for handle in drawn.values():handle.remove()
        drawn.clear()
        if row is None or trajectory is None:return
        t0=row['time_s'];ts=trajectory['time']
        k=int(np.clip(np.searchsorted(ts,t0+1e-6)-1,0,len(ts)-1))
        data.qpos[:]=trajectory['qpos'][k];mujoco.mj_kinematics(model,data)
        live.log_frame(data,float(ts[k]),handles,record=False)
        def lines(name,paths,colors,thickness):
            """paths (K, T, 3 points, xyz), one RGB color per path (K, 3)."""
            paths=paths[:,:,visible];K,T,P,_=paths.shape
            if K==0 or P==0 or T<2:return
            pts=segments(paths.transpose(0,2,1,3).reshape(K*P,T,3))
            col=np.repeat(np.repeat(colors,P,axis=0),T-1,axis=0)
            drawn[name]=server.scene.add_line_segments(f'/opt/{name}',pts,np.repeat(col[:,None],2,axis=1),
                thickness=thickness,thickness_units='screen')
        # Actually executed path over the same horizon (only what exists so far).
        horizon=(ts>=t0-1e-6)&(ts<=t0+1.6+1e-6)
        if show_actual.value and horizon.sum()>1:
            path=[]
            for q in trajectory['qpos'][horizon][::4]:
                data.qpos[:]=q;mujoco.mj_kinematics(model,data)
                path.append([*data.xpos[wrist_bodies],q[-14:-11]])
            data.qpos[:]=trajectory['qpos'][k];mujoco.mj_kinematics(model,data)
            lines('actual',np.array(path)[None],np.array([[245,245,245]]),4.)
        if sample is None:return
        i=min(int(iteration_slider.value)-1,len(sample['paths'])-1);top=int(sample['top'])
        paths=sample['paths'][i];score=sample['score'][i]
        if show_reference.value:lines('reference',sample['reference'][None],np.array([[40,110,255]]),4.)
        if show_top.value:
            rank=np.linspace(0,1,top)[:,None]
            lines('top',paths[:top],(np.array([[30,200,60]])*(1-rank)+np.array([[240,220,40]])*rank).astype(np.uint8),2.)
        if show_uniform.value:
            all_scores=sample['all_scores'][i];finite=all_scores[np.isfinite(all_scores)]
            lo,hi=(np.percentile(finite,5),finite.max()) if len(finite) else (0.,1.)
            u=np.clip((score[top:]-lo)/max(hi-lo,1e-9),0,1)[:,None]
            lines('uniform',paths[top:],(np.array([[220,50,50]])*(1-u)+np.array([[160,160,160]])*u).astype(np.uint8),1.)

    print(f'Viser: http://localhost:{a.port}   (read-only; Ctrl+C closes only this viewer)',flush=True)
    try:
        while True:
            with lock:
                dirty=state['dirty'];state['dirty']=False;rows=list(state['log']);trajectory=state['trajectory']
            if rows:
                window_slider.max=max(2,rows[-1]['window']);window_slider.disabled=len(rows)<2
                if follow.value:
                    optimized=[r['window'] for r in rows if r.get('iterations')]
                    target=optimized[-1] if optimized else rows[-1]['window']
                    if window_slider.value!=target:window_slider.value=target;dirty=True
                    if sample_file(target) is not None:
                        n=len(next(r for r in rows if r['window']==target).get('iterations',[]))
                        if n and iteration_slider.value!=n:iteration_slider.value=n;dirty=True
            key=(int(window_slider.value),int(iteration_slider.value))
            if rows and (dirty or key!=state['drawn']):
                state['drawn']=key;sample=sample_file(key[0])
                draw_timeline(rows);draw_window(rows,sample);draw_scene(rows,sample,trajectory)
            time.sleep(.3)
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
