"""Prepare everything spider_arm_mpc.py needs to simulate one HOCap take.

    python sim/prepare_task.py --task fwd_capture__Bottle_0820_fwd__take02 [--mass 0.2] [--reference-end 17]

Writes <BIHOI_ROBOT_DATA>/tasks/<task>/:
  contact/   object convex decomposition (CoACD), physics scene.xml with the
             Sharpa hands, retargeted trajectories, contact schedule
             (BIHOI_CONTACT_ROOT for the simulation)
             Lead-in: the retargeted trajectory starts with a still copy of its
             first frame so that the human's first hand-object contact happens
             at >= --first-contact-at seconds; the simulation then has time for
             its open-hand approach before the human touches. Simulation time =
             HOCap time + lead_in_s (schedule.json).
  plan/      reference_info.json and the UR3e base placement + arm trajectory:
             search_placement -> plan_mink -> global_sqp -> sqp_result_001.json
  prepare.json  stage, timings and the plan metrics (or the error)
Each stage is skipped when its output exists, so a rerun continues.
"""
import argparse,json,os,shutil,subprocess,sys,time,traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from robot_paths import DATA,SIM as CODE
RETARGETED=DATA/'retargeted';TASKS=DATA/'tasks'
# CoACD settings used for the bowl demo (bihoi_contact_prepare.py of the original bowl run).
COACD=dict(threshold=.025,max_convex_hull=48,preprocess_mode='auto',preprocess_resolution=80,resolution=2000,
    mcts_nodes=20,mcts_iterations=100,mcts_max_depth=3,pca=False,merge=True,decimate=True,max_ch_vertex=64,seed=0)

def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def convex_parts(mesh_dir,out):
    """CoACD decomposition of the object's source mesh (object frame unchanged), shared by takes of one mesh."""
    import coacd,trimesh,numpy as np
    out.mkdir(parents=True,exist_ok=True)
    for name in ('visual.obj','source_mesh.ply'):
        if not (out/name).exists():shutil.copy2(mesh_dir/name,out/name)
    convex=out/'convex'
    if list(convex.glob('*.obj')):return
    mesh=trimesh.load(mesh_dir/'source_mesh.ply',process=False)
    simple=mesh.simplify_quadric_decimation(face_count=24000) if len(mesh.faces)>24000 else mesh
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(simple.vertices),np.asarray(simple.faces)),**COACD)
    temp=out/'convex.tmp';shutil.rmtree(temp,ignore_errors=True);temp.mkdir()
    for i,(v,f) in enumerate(parts):trimesh.Trimesh(v,f,process=False).export(temp/f'{i}.obj')
    dump(out/'collision_preparation.json',dict(method='CoACD',source_faces=len(mesh.faces),simplified_faces=len(simple.faces),hulls=len(parts),**COACD))
    temp.replace(convex)

def physics_scene(task,contact,mass):
    """Sharpa bimanual scene with the convex object (same generator settings as the bowl demo)."""
    import mujoco
    import retargeting.pipeline.generate_scene as generator
    directory=contact/'sharpa'/'bimanual'/task/'0';scene=directory/'scene.xml'
    if scene.exists():return
    generator.ROOT=str(RETARGETED)  # robot assets come from data/retargeted/assets/robots
    generator.main(output_root_dir=str(contact),dataset_name='bihoi',robot_type='sharpa',embodiment_type='bimanual',task=task,
        force=True,show_viewer=False,use_visual_mesh_as_collision=False,object_floor_collision=True,hand_floor_collision=False,
        object_armature=0,object_damping=0,object_frictionloss=0)
    tree=ET.parse(directory/'scene_ik.xml');body=tree.find(".//body[@name='right_object']")
    current=mujoco.MjModel.from_xml_path(str(directory/'scene_ik.xml'));scale=mass/current.body_mass[current.body('right_object').id]
    for geom in body.findall('geom'):
        if geom.get('density') is not None or geom.get('contype')!='0':geom.set('density',str(float(geom.get('density','1000'))*scale))
    temp=directory/'scene.tmp.xml';tree.write(temp,encoding='unicode')
    check=mujoco.MjModel.from_xml_path(str(temp))
    if abs(check.body_mass[check.body('right_object').id]-mass)>1e-6:raise RuntimeError(f'object mass {check.body_mass[check.body("right_object").id]} != {mass}')
    temp.replace(scene)

def contact_root(task,contact,schedule,mass,lead_in):
    info=json.loads((RETARGETED/'sharpa'/'bimanual'/task/'task_info.json').read_text())
    mesh_dir=RETARGETED/info['right_object_mesh_dir'];key=Path(info['right_object_mesh_dir']).name
    convex_parts(mesh_dir,TASKS/'_objects'/key)
    objects=contact/'assets'/'objects';objects.mkdir(parents=True,exist_ok=True)
    if not (objects/key).exists():(objects/key).symlink_to(TASKS/'_objects'/key,target_is_directory=True)
    for robot,name in (('mano','trajectory_keypoints.npz'),('sharpa','trajectory_kinematic.npz')):
        dest=contact/robot/'bimanual'/task/'0';dest.mkdir(parents=True,exist_ok=True)
        if not (dest/name).exists():shutil.copy2(RETARGETED/robot/'bimanual'/task/'0'/name,dest/name)
        if robot=='sharpa':
            import numpy as np
            source=np.load(RETARGETED/robot/'bimanual'/task/'0'/name)
            pad=lambda x:np.r_[np.repeat(x[:1],lead_in,axis=0),x]
            np.savez_compressed(dest/name,qpos=pad(source['qpos']),qvel=np.r_[np.zeros((lead_in,)+source['qvel'].shape[1:]),source['qvel']],
                frequency=source['frequency'],lead_in_frames=lead_in)
        dump(dest.parent/'task_info.json',dict(info,collision_preparation='CoACD convex parts, original coordinates'))
    physics_scene(task,contact,mass)
    (contact/'trial_accepted_v2').mkdir(exist_ok=True)
    dump(contact/'trial_accepted_v2'/'schedule.json',schedule)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task',required=True)
    p.add_argument('--mass',type=float,default=.2,help='object mass (kg)')
    p.add_argument('--reference-end',type=float,default=17.,help='reference/plan length (s of HOCap time): simulated end + MPC horizon')
    p.add_argument('--first-contact-at',type=float,default=3.,help='lead-in: first human contact at this simulation time (s) or later')
    a=p.parse_args()
    root=TASKS/a.task;contact=root/'contact';plan=root/'plan';plan.mkdir(parents=True,exist_ok=True);contact.mkdir(exist_ok=True)
    state=dict(task=a.task,mass_kg=a.mass,reference_end_s=a.reference_end,stages={},started=time.time())
    def stage(name):state['stage']=name;dump(root/'prepare.json',state);return time.monotonic()
    env=dict(os.environ,BIHOI_TASK=a.task,BIHOI_CONTACT_ROOT=str(contact),BIHOI_ARM_ROOT=str(plan),PYTHONPATH=str(CODE),MUJOCO_GL='disable')
    def run(name,*args):
        t=stage(name)
        with (root/f'{name}.log').open('w') as log:
            code=subprocess.call([sys.executable,'-u',str(CODE/args[0]),*args[1:]],env=env,stdout=log,stderr=subprocess.STDOUT)
        if code:raise RuntimeError(f'{name} exited {code}; see {root/(name+".log")}')
        state['stages'][name]=time.monotonic()-t
    try:
        t=stage('contact')
        ranking=json.loads((TASKS/'take_ranking.json').read_text())
        row=next(r for v in ranking['objects'].values() for r in v['takes'] if r['task']==a.task)
        lead_in=max(0,round(a.first_contact_at*30)-row['first_contact_frame'])
        schedule=dict(first_contact_frame=row['first_contact_frame']+lead_in,anchor_source_frame=row['anchor_frame']+lead_in,
            lead_in_frames=lead_in,lead_in_s=lead_in/30,hocap_first_contact_frame=row['first_contact_frame'],hocap_anchor_frame=row['anchor_frame'],
            note='frames are simulation frames (30 Hz) = HOCap frames + lead_in_frames',
            source='sim/hocap.py contact analysis of the raw take (rank_takes.py)',reference_only=True)
        old=contact/'trial_accepted_v2'/'schedule.json'
        if old.exists() and json.loads(old.read_text()).get('lead_in_frames')!=lead_in:
            # Prepared with another lead-in: the timeline changed, so redo the trajectory and the plan (scene and convex parts stay).
            (contact/'sharpa'/'bimanual'/a.task/'0'/'trajectory_kinematic.npz').unlink(missing_ok=True)
            shutil.rmtree(plan);plan.mkdir()
        state.update(lead_in_s=lead_in/30)
        contact_root(a.task,contact,schedule,a.mass,lead_in);state['stages']['contact']=time.monotonic()-t
        frames=json.loads((RETARGETED/'sharpa'/'bimanual'/a.task/'task_info.json').read_text())['n_frames']
        if not (plan/'reference_info.json').exists():
            run('reference','prepare_reference.py','--exclude-last-seconds',str((frames-round(a.reference_end*30)-1)/30))
        if not (plan/'best_placement.json').exists():
            run('placement','search_placement.py','--headings','9','--distances','5','--lateral','5','--frames','41','--iterations','45','--finalists','12')
        if not (plan/'mink_result_000.json').exists():run('continuous_ik','plan_mink.py','--candidate','0')
        if not (plan/'sqp_result_000.json').exists():run('smooth_refinement','global_sqp.py','--candidate','0')
        shutil.copy2(plan/'sqp_result_000.json',plan/'sqp_result_001.json');shutil.copy2(plan/'sqp_scene_000.xml',plan/'sqp_scene_001.xml')
        result=json.loads((plan/'sqp_result_001.json').read_text())
        state.update(stage='ready',plan={k:v for k,v in result.items() if k!='history'})
    except Exception as ex:
        state.update(stage='failed',error=f'{type(ex).__name__}: {ex}',traceback=traceback.format_exc())
    state['wall_s']=time.time()-state.pop('started')
    dump(root/'prepare.json',state);print(json.dumps({k:state[k] for k in ('task','stage','stages','wall_s')}|({'error':state['error']} if 'error' in state else {})))
    if state['stage']!='ready':raise SystemExit(1)

if __name__=='__main__':main()
