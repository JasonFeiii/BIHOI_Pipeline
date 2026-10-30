"""Watch the simulations of a run_batch.py batch in one Viser page.

    python batch_viewer.py [--batch-dir demo_runs/batch_...] [--port 8150]

Pick an object from the drop-down: its simulated arms, hands and object are
played back from the run's trajectory.npz (reloaded whenever the simulation
saves a window), with the reference object pose as a translucent blue ghost.
The table at the top shows every job of the batch. Read-only: it never touches
the simulations.
"""
import argparse,json,os,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sim'))
from robot_paths import OUTPUTS,PYTHON
PHYS_PY=PYTHON
GHOST=(.2,.55,1.,.35)

def ensure_env():
    if os.path.realpath(sys.executable)!=os.path.realpath(PHYS_PY):
        os.execv(PHYS_PY,[PHYS_PY,'-Bu',str(Path(__file__).resolve()),*sys.argv[1:]])

def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def geom_mesh(model,g):
    """Trimesh of geom g in its own frame, or None for planes/unsupported types."""
    import mujoco,numpy as np,trimesh
    t=model.geom_type[g];size=model.geom_size[g]
    if t==mujoco.mjtGeom.mjGEOM_MESH:
        m=model.geom_dataid[g];v0,nv=model.mesh_vertadr[m],model.mesh_vertnum[m];f0,nf=model.mesh_faceadr[m],model.mesh_facenum[m]
        mesh=trimesh.Trimesh(model.mesh_vert[v0:v0+nv].copy(),model.mesh_face[f0:f0+nf].copy(),process=False)
        if nf>30000:mesh=mesh.simplify_quadric_decimation(face_count=30000)
        return mesh
    if t==mujoco.mjtGeom.mjGEOM_BOX:return trimesh.creation.box(extents=2*size)
    if t==mujoco.mjtGeom.mjGEOM_SPHERE:return trimesh.creation.icosphere(radius=size[0],subdivisions=2)
    if t==mujoco.mjtGeom.mjGEOM_CAPSULE:return trimesh.creation.capsule(radius=size[0],height=2*size[1])
    if t==mujoco.mjtGeom.mjGEOM_CYLINDER:return trimesh.creation.cylinder(radius=size[0],height=2*size[1])
    return None

class Case:
    """Viser meshes of one run's scene, posed from qpos by forward kinematics."""
    def __init__(self,server,run_dir):
        import mujoco,numpy as np
        self.run_dir=run_dir;self.server=server
        self.model=mujoco.MjModel.from_xml_path(str(run_dir/'sqp_scene_001.xml'));self.data=mujoco.MjData(self.model)
        m=self.model;self.geoms=[];self.handles=[]
        obj=m.body('right_object').id
        for g in range(m.ngeom):
            body=m.body(m.geom_bodyid[g]).name
            if m.geom_group[g]>2 or body.startswith(('ref_','left_object')):continue
            mesh=geom_mesh(m,g)
            if mesh is None:continue
            rgba=m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g]>=0 else m.geom_rgba[g]
            self.handles.append(server.scene.add_mesh_simple(f'/case/g{g}',mesh.vertices,mesh.faces,color=tuple(rgba[:3]),opacity=float(rgba[3]) if rgba[3]<1 else None))
            self.geoms.append(g)
            if m.geom_bodyid[g]==obj and m.geom_group[g]==0:self.ghost_geom=(g,mesh)
        g,mesh=self.ghost_geom
        self.ghost=server.scene.add_mesh_simple('/case/reference_object',mesh.vertices,mesh.faces,color=GHOST[:3],opacity=GHOST[3])
        self.stamp=None;self.time=self.qpos=None
        with np.load(self.physics/'reference.npz') as z:self.ref_time=z['time'].copy();self.ref_object=z['qpos'][:,-14:-7].copy()
        self.ghost_offset=(m.geom_pos[g].copy(),m.geom_quat[g].copy())  # visual geom pose in the object body frame
    @property
    def physics(self):return self.run_dir/'physics'/'spider_arm_0001'
    def reload(self):
        """Reload the trajectory if the simulation saved a new window; True when it changed."""
        import numpy as np
        path=self.physics/'trajectory.npz'
        try:stamp=path.stat().st_mtime_ns
        except OSError:return False
        if stamp==self.stamp:return False
        try:
            with np.load(path) as z:self.time,self.qpos=z['time'].copy(),z['qpos'].copy()
        except (OSError,ValueError,KeyError,EOFError):return False
        self.stamp=stamp;return True
    def show(self,index,ghost):
        import mujoco,numpy as np
        m,d=self.model,self.data;d.qpos[:]=self.qpos[index];mujoco.mj_kinematics(m,d)
        quat=np.empty(4)
        for g,h in zip(self.geoms,self.handles):
            mujoco.mju_mat2Quat(quat,d.geom_xmat[g]);h.position=d.geom_xpos[g];h.wxyz=quat
        self.ghost.visible=ghost
        if ghost:
            j=min(np.searchsorted(self.ref_time,self.time[index]-1e-9),len(self.ref_time)-1);pose=self.ref_object[j]
            pos,q=self.ghost_offset;world=np.empty(4);mujoco.mju_mulQuat(world,pose[3:],q)
            rot=np.empty(9);mujoco.mju_quat2Mat(rot,pose[3:])
            self.ghost.position=pose[:3]+rot.reshape(3,3)@pos;self.ghost.wxyz=world
        return float(np.linalg.norm(self.qpos[index,-14:-11]-self.ref_object[min(np.searchsorted(self.ref_time,self.time[index]-1e-9),len(self.ref_time)-1),:3]))
    def remove(self):
        for h in self.handles:h.remove()
        self.ghost.remove()

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch-dir',type=Path);p.add_argument('--port',type=int,default=8150)
    a=p.parse_args();ensure_env()
    import numpy as np,viser
    batch=(a.batch_dir or max((OUTPUTS/'demo_runs').glob('batch_*'),key=lambda x:x.stat().st_mtime)).resolve()
    server=viser.ViserServer(port=a.port);server.gui.configure_theme(control_width='large')
    server.scene.add_grid('/grid',width=2,height=2,cell_size=.1,position=(.2,0,0))
    table=server.gui.add_markdown('读取 batch_status.json…')
    def jobs():return read_json(batch/'batch_status.json').get('jobs',[])
    # A case is shown once its simulation has written the reference (after its start-up).
    ready=lambda name:(batch/name/'sqp_scene_001.xml').exists() and (batch/name/'physics'/'spider_arm_0001'/'reference.npz').exists()
    def options():
        names=[j['object'] for j in jobs() if ready(j['object'])]
        return names or ['(还没有开始的仿真)']
    choice=server.gui.add_dropdown('物体',options=options())
    follow=server.gui.add_checkbox('跟随最新',initial_value=True)
    play=server.gui.add_checkbox('循环回放',initial_value=True)
    speed=server.gui.add_slider('播放倍率',min=.1,max=2.,step=.1,initial_value=1.)
    frame=server.gui.add_slider('帧',min=0,max=1,step=1,initial_value=0)
    ghost=server.gui.add_checkbox('参考物体（蓝色）',initial_value=True)
    info=server.gui.add_markdown('')
    state=dict(case=None,name=None,position=0.,last=-1)
    lock=threading.Lock()
    def select(name):
        with lock:
            if state['case']:state['case'].remove();state['case']=None
            state['name']=name
            if not ready(name):return
            info.content=f'加载 **{name}** 场景…'
            case=Case(server,batch/name);case.reload();state['case']=case;state['position']=0.;state['last']=-1
            if case.qpos is not None:
                target=case.qpos[0,-14:-11]
                for client in server.get_clients().values():
                    client.camera.look_at=tuple(target);client.camera.position=tuple(target+np.array([-.55,-.55,.45]));client.camera.up_direction=(0,0,1)
    @choice.on_update
    def _(_):select(choice.value)
    @server.on_client_connect
    def _(client):
        case=state['case']
        target=case.qpos[0,-14:-11] if case is not None and case.qpos is not None else np.array([.2,0,.1])
        client.camera.look_at=tuple(target);client.camera.position=tuple(target+np.array([-.55,-.55,.45]));client.camera.up_direction=(0,0,1)
    select(choice.value)
    print(f'Batch viewer for {batch}\n  http://localhost:{a.port}',flush=True)
    clock=time.monotonic();next_table=0.
    while True:
        now=time.monotonic();dt=now-clock;clock=now
        if now>=next_table:
            rows=['| 物体 | take | 状态 | 已到 (s) |','|---|---|---|---|']
            for j in jobs():
                rows.append(f"| {j['object']} | {j.get('task','').split('__')[-1]} | {j['state']} | {j.get('time_s') or 0:.2f} |")
            table.content=f'**{batch.name}**\n\n'+'\n'.join(rows)
            current=options()
            if current!=list(choice.options):choice.options=current
            if state['name']!=choice.value:select(choice.value)
            next_table=now+3.
        with lock:
            case=state['case']
            if case is not None:
                case.reload()
                if case.qpos is not None:
                    ts=case.time;frame.max=max(1,len(ts)-1)
                    if int(frame.value)!=state['last']:state['position']=float(ts[min(int(frame.value),len(ts)-1)])
                    if follow.value:state['position']=float(ts[-1])
                    elif play.value:
                        state['position']+=dt*float(speed.value)
                        if state['position']>ts[-1]:state['position']=float(ts[0])
                    index=int(np.clip(np.searchsorted(ts,state['position'],side='right')-1,0,len(ts)-1))
                    with server.atomic():error=case.show(index,ghost.value)
                    frame.value=index;state['last']=index
                    status=next((j for j in jobs() if j['object']==state['name']),{})
                    info.content=(f"**{state['name']}** · {status.get('task','')}\n\n播放 **{ts[index]:.2f} s** / 已仿真 {ts[-1]:.2f} s"
                        f" · 物体误差 {error*1000:.0f} mm · 状态 {status.get('state','?')}")
        time.sleep(1/30)

if __name__=='__main__':main()
