"""Render one simulation run to MP4: HOCap capture (left) next to the physics simulation (right).

    python sim/render_run.py --run outputs/robot/demo_runs/<batch>/<object> [--out FILE] [--end 16] [--speed 1]

--run is a run_batch.py object directory (batch_job.json, sqp_scene_001.xml,
physics/spider_arm_0001/). Both panels share the simulation clock: the HOCap
frame shown is (t - lead_in_s) * 30, holding frame 0 during the still lead-in
(sim/prepare_task.py). Right: the dual-arm simulation, with the reference
object pose as a translucent blue ghost and the object error in the header.
Default output: <run>/video_hocap_vs_sim.mp4.
"""
import argparse,json,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
from robot_paths import DATA
from render_retarget import ensure_env,egl_default_display,HumanScene,camera_for,label

GHOST=(.2,.55,1.,.35)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--end',type=float,help='last simulation time to render (s, default: end of the trajectory)')
    p.add_argument('--fps',type=int,default=30);p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--width',type=int,default=800);p.add_argument('--height',type=int,default=600)
    a=p.parse_args();ensure_env();egl_default_display()
    import imageio,mujoco,numpy as np
    from scipy.spatial.transform import Rotation
    run=a.run.resolve();job=json.loads((run/'batch_job.json').read_text());lead_in=job.get('lead_in_s',0.)
    physics=run/'physics'/'spider_arm_0001';out=a.out or run/'video_hocap_vs_sim.mp4'
    task=job['task'];info=json.loads((DATA/'retargeted'/'sharpa'/'bimanual'/task/'task_info.json').read_text())

    # Right: simulation scene, visual geoms only (groups 0-2), plus the reference object as a ghost.
    model=mujoco.MjModel.from_xml_path(str(run/'sqp_scene_001.xml'))
    model.vis.global_.offwidth=max(a.width,model.vis.global_.offwidth);model.vis.global_.offheight=max(a.height,model.vis.global_.offheight)
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name.startswith(('ref_','left_object')):model.geom_rgba[g,3]=0
    obj=model.body('right_object').id
    sim_data,ref_data=mujoco.MjData(model),mujoco.MjData(model)
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    robot=mujoco.Renderer(model,a.height,a.width)
    trajectory=np.load(physics/'trajectory.npz');reference=np.load(physics/'reference.npz')
    time,qpos=trajectory['time'],trajectory['qpos'];ref_time,ref_qpos=reference['time'],reference['qpos']

    person=HumanScene(info['source_relative_path'],a.width,a.height)
    camera=camera_for(person.capture['object_pos'])
    T=len(person.capture['object_pos']);quat=lambda q:Rotation.from_quat(q[[1,2,3,0]])
    name=task.replace('fwd_capture__','')
    end=min(time[-1],a.end) if a.end else time[-1]
    frames=np.arange(0,end+1e-9,a.speed/a.fps)
    with imageio.get_writer(out,fps=a.fps,codec='libx264',quality=8,macro_block_size=8) as writer:
        for t in frames:
            i=min(np.searchsorted(time,t-1e-9),len(time)-1);j=min(np.searchsorted(ref_time,time[i]-1e-9),len(ref_time)-1)
            hocap_t=time[i]-lead_in;k=int(np.clip(round(hocap_t*30),0,T-1))
            sim_data.qpos[:]=qpos[i];mujoco.mj_forward(model,sim_data)
            ref_data.qpos[:]=ref_qpos[j];mujoco.mj_forward(model,ref_data)
            robot.update_scene(sim_data,camera,scene_option=option)
            scene=robot.scene;first=scene.ngeom
            mujoco.mjv_addGeoms(model,ref_data,option,mujoco.MjvPerturb(),mujoco.mjtCatBit.mjCAT_DYNAMIC,scene)
            for n in range(first,scene.ngeom):
                geom=scene.geoms[n]
                keep=geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid]==obj and model.geom_group[geom.objid]==0
                geom.rgba[:]=GHOST if keep else (0,0,0,0)
            position=1000*np.linalg.norm(qpos[i,-14:-11]-ref_qpos[j,-14:-11])
            angle=np.degrees((quat(qpos[i,-11:-7]).inv()*quat(ref_qpos[j,-11:-7])).magnitude())
            color=(255,255,255) if position<30 else (255,180,60) if position<60 else (255,80,80)
            still=' (still lead-in)' if hocap_t<0 else ''
            left=label(person.render(k,camera),[(f'HOCap capture (MANO + object)  t={time[i]:5.2f}s',(255,255,255)),
                (f'{name}  capture t={max(hocap_t,0):5.2f}s{still}',(180,180,180))])
            right=label(robot.render().copy(),[(f'Simulation  t={time[i]:5.2f}s',(255,255,255)),
                ('blue ghost = reference object pose',(150,200,255)),(f'object err {position:4.0f} mm  {angle:5.1f} deg',color)])
            writer.append_data(np.concatenate([left,right],axis=1))
    print(f'wrote {out} ({len(frames)} frames, sim 0-{end:.2f} s, lead-in {lead_in:.2f} s)')

if __name__=='__main__':main()
