"""Combined demo video: raw camera | HOCap capture | dual-arm simulation, for several runs at once.

    python sim/render_demo.py [--end 13] [--out outputs/robot/results/demo_6cases_13s/demo_6cases_13s.mp4]

Each case is one row of three panels: the raw capture video from the camera
that sees both hands and the object best (picked by eye from all 30 views),
the HOCap MANO hands and object, and the simulated UR3e arms + Sharpa hands
seen from the object side so both arms are fully visible; HOCap uses the same
view direction, closer. All panels follow the simulation clock: raw video and
HOCap show capture time t - lead_in_s and hold their first frame during the
still lead-in (sim/prepare_task.py). Raw frames are matched by their recorded
timestamps (<camera>_frames.csv). The cases are rendered in parallel, one
process each (--case), then stacked top to bottom (<out>) and played one
after another (<out stem>_sequential.mp4).
"""
import argparse,csv,json,subprocess,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
from robot_paths import DATA,OUTPUTS
from render_retarget import ensure_env,egl_default_display,HumanScene,label

# (title, saved run under outputs/robot/results, task, raw camera)
CASES=[
    ('Big steel bowl take01','spider_arm_demo_0-14s/run','fwd_capture__8.27_big_steel_bowl_fwd__take01','node34_cam0'),
    ('Blue bowl take02','9.3_blue_bowl_fwd_take02_objw10_rot01/run','fwd_capture__9.3_blue_bowl_fwd__take02','node37_cam0'),
    ('Mouse take01','Mouse_0820_fwd_take01_objw5_rot01/run','fwd_capture__Mouse_0820_fwd__take01','node35_cam1'),
    ('Cheese slicer take01','9.1_cheese_slicer_fwd_take01_objw5_rot01/run','fwd_capture__9.1_cheese_slicer_fwd__take01','node38_cam1'),
    ('0818_fwd_03 take02','0818_fwd_03_take02_objw10_rot01/run','fwd_capture__0818_fwd_03__take02','node33_cam0'),
    ('0818_fwd_06 take01','0818_fwd_06_take01_leadin/run','fwd_capture__0818_fwd_06__take01','node35_cam1'),
]
W,H=640,360          # one panel (16:9 like the raw videos)
ELEVATION=-30.       # both rendered panels
SIM_DISTANCE=1.1;HUMAN_DISTANCE=.55

def raw_frames(task,camera):
    """Raw video reader: frame for capture time t (s), nearest recorded timestamp."""
    import cv2,numpy as np
    _,obj,take=task.split('__');videos=DATA/'raw'/'fwd_capture'/obj/'videos'/take
    with open(videos/f'{camera}_frames.csv') as stream:stamps=np.array([int(r['rel_timestamp_us']) for r in csv.DictReader(stream)])/1e6
    cap=cv2.VideoCapture(str(videos/f'{camera}.mp4'));state=dict(index=-1,image=None)
    def frame(t):
        k=int(np.abs(stamps-t).argmin())
        if k<state['index']:cap.set(cv2.CAP_PROP_POS_FRAMES,k);state['index']=k-1
        while state['index']<k:
            ok,image=cap.read();state['index']+=1
            if ok:state['image']=image
        return cv2.resize(state['image'],(W,H),interpolation=cv2.INTER_AREA)[:,:,::-1].copy()
    return frame

def render_case(index,end,out):
    import imageio,mujoco,numpy as np
    title,run,task,camera_name=CASES[index];run=OUTPUTS/'results'/run
    job=json.loads((run/'batch_job.json').read_text()) if (run/'batch_job.json').exists() else {}
    lead_in=job.get('lead_in_s',0.)
    info=json.loads((DATA/'retargeted'/'sharpa'/'bimanual'/task/'task_info.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(run/'sqp_scene_001.xml'))
    model.vis.global_.offwidth=max(W,model.vis.global_.offwidth);model.vis.global_.offheight=max(H,model.vis.global_.offheight)
    model.vis.headlight.ambient[:]=.35;model.vis.headlight.diffuse[:]=.6
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name.startswith(('ref_','left_object')):model.geom_rgba[g,3]=0
    data=mujoco.MjData(model);option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    robot=mujoco.Renderer(model,H,W)
    trajectory=np.load(run/'physics'/'spider_arm_0001'/'trajectory.npz');time,qpos=trajectory['time'],trajectory['qpos']
    error=trajectory['object_error']

    # View: from the object towards the middle of the two arm bases (they face the object).
    data.qpos[:]=qpos[0];mujoco.mj_forward(model,data)
    bases=(data.body('right_arm_base').xpos+data.body('left_arm_base').xpos)/2
    person=HumanScene(info['source_relative_path'],W,H);T=len(person.capture['object_pos'])
    floor=[g for g in range(person.model.ngeom) if person.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
    person.model.geom_size[floor,:2]=4  # no visible floor edge in the close HOCap view
    shown=person.capture['object_pos'][:max(1,int((end-lead_in)*30))]
    centre=shown.mean(0);azimuth=float(np.degrees(np.arctan2(*(bases-centre)[[1,0]])))
    def camera(lookat,distance):
        c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE;c.lookat[:]=lookat
        c.distance=distance;c.azimuth=azimuth;c.elevation=ELEVATION;return c
    sim_camera=camera((bases+centre)/2+[0,0,.1],SIM_DISTANCE)
    human_camera=camera((centre[0],centre[1],max(centre[2],.06)),HUMAN_DISTANCE)
    raw=raw_frames(task,camera_name)

    frames=np.arange(0,min(end,time[-1])+1e-9,1/30)
    with imageio.get_writer(out,fps=30,codec='libx264',quality=8,macro_block_size=8) as writer:
        for t in frames:
            i=min(np.searchsorted(time,t-1e-9),len(time)-1);capture=max(time[i]-lead_in,0.)
            k=int(np.clip(round(capture*30),0,T-1))
            data.qpos[:]=qpos[i];mujoco.mj_forward(model,data);robot.update_scene(data,sim_camera,scene_option=option)
            still=' (still lead-in)' if time[i]<lead_in else ''
            e=1000*error[i];color=(255,255,255) if e<30 else (255,180,60) if e<60 else (255,80,80)
            panels=[label(raw(capture),[(f'Raw video  {camera_name}',(255,255,255))]),
                label(person.render(k,human_camera),[(f'HOCap  capture t={capture:5.2f}s{still}',(255,255,255))]),
                label(robot.render().copy(),[(f'Simulation  object err {e:3.0f} mm',color)])]
            row=np.concatenate(panels,axis=1);header=np.full((44,row.shape[1],3),12,np.uint8)
            import cv2
            cv2.putText(header,f'{title}   t={time[i]:5.2f}s',(14,31),cv2.FONT_HERSHEY_SIMPLEX,.95,(255,255,255),2,cv2.LINE_AA)
            writer.append_data(np.concatenate([header,row],axis=0))
    print(f'case {index}: wrote {out} ({len(frames)} frames)',flush=True)

def compose(cells,out):
    """All cases stacked top to bottom, in CASES order."""
    import imageio,numpy as np
    readers=[imageio.get_reader(c) for c in cells]
    with imageio.get_writer(out,fps=30,codec='libx264',quality=8,macro_block_size=8) as writer:
        for frames in zip(*readers):
            writer.append_data(np.concatenate(frames,axis=0))

def sequence(cells,out):
    """The cases one after another, each full width (raw | HOCap | simulation)."""
    import imageio
    with imageio.get_writer(out,fps=30,codec='libx264',quality=8,macro_block_size=8) as writer:
        for cell in cells:
            for frame in imageio.get_reader(cell):writer.append_data(frame)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--end',type=float,default=13.,help='simulation time to show (s)')
    p.add_argument('--out',type=Path,default=OUTPUTS/'results'/'demo_6cases_13s'/'demo_6cases_13s.mp4')
    p.add_argument('--case',type=int,help='render only this case into <out dir>/cells/ (used by the parallel run)')
    a=p.parse_args();ensure_env();egl_default_display()
    cells=a.out.parent/'cells';cells.mkdir(parents=True,exist_ok=True)
    cell=lambda i:cells/f'{i}_{CASES[i][0].replace(" ","_")}.mp4'
    if a.case is not None:render_case(a.case,a.end,cell(a.case));return
    jobs=[subprocess.Popen([sys.executable,'-Bu',str(Path(__file__).resolve()),'--end',str(a.end),'--out',str(a.out),'--case',str(i)]) for i in range(len(CASES))]
    if any(j.wait() for j in jobs):raise SystemExit('a case failed')
    compose([str(cell(i)) for i in range(len(CASES))],a.out)
    sequence([str(cell(i)) for i in range(len(CASES))],a.out.with_name(a.out.stem+'_sequential.mp4'))
    (a.out.parent/'cases.json').write_text(json.dumps([dict(title=t,run=r,task=k,raw_camera=c) for t,r,k,c in CASES],indent=2))
    print(f'wrote {a.out}')

if __name__=='__main__':main()
