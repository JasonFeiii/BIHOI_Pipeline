"""Render a take's retargeting to MP4: HOCap capture (left) next to the retargeted Sharpa hands (right).

    python sim/render_retarget.py --task fwd_capture__Bottle_0820_fwd__take03 [--out FILE] [--speed 1]

Left: MANO hands and the object at its captured pose (data/raw). Right: the
kinematic retargeting result, floating Sharpa hands and object from
data/retargeted/sharpa/bimanual/<task>/0 (scene_ik.xml + trajectory_kinematic.npz).
No physics. Default output: outputs/robot/retarget_videos/<task>.mp4.
"""
import argparse,json,os,sys
from pathlib import Path
from robot_paths import DATA,OUTPUTS,PYTHON as PHYS_PY,SIM as CODE
SKIN=(.93,.75,.62,1.)

def ensure_env():
    """Re-exec under the robot environment with headless EGL rendering if needed."""
    if os.path.realpath(sys.executable)==os.path.realpath(PHYS_PY) and os.environ.get('MUJOCO_GL')=='egl':return
    os.execve(PHYS_PY,[PHYS_PY,'-Bu',str(Path(sys.argv[0]).resolve()),*sys.argv[1:]],dict(os.environ,MUJOCO_GL='egl',PYTHONPATH=str(CODE)))

def egl_default_display():
    # The container exposes EGL devices that fail eglInitialize; the default display works.
    import mujoco.egl
    from OpenGL import EGL
    def create():
        display=EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY);EGL.eglInitialize(display,None,None);return display
    mujoco.egl.create_initialized_egl_device_display=create

class HumanScene:
    """HOCap capture renderer: the object mesh as a mocap body and the MANO meshes rewritten per frame."""
    def __init__(self,source,width,height):
        import mujoco,numpy as np
        import hocap
        self.mujoco=mujoco;self.np=np
        take=hocap.RAW/source;self.capture=capture=hocap.load_take(source)
        spec=mujoco.MjSpec();spec.visual.global_.offwidth=width;spec.visual.global_.offheight=height
        spec.worldbody.add_light(pos=[.2,-.2,1.5],dir=[0,0,-1],diffuse=[.6,.6,.6])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[.8,.8,.01],rgba=[.35,.35,.35,1])
        mesh=capture['mesh']
        if len(mesh.faces)>60000:mesh=mesh.simplify_quadric_decimation(face_count=60000)
        spec.add_mesh(name='object',uservert=np.asarray(mesh.vertices,np.float32).ravel(),userface=np.asarray(mesh.faces,np.int32).ravel())
        spec.worldbody.add_body(name='object',mocap=True).add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname='object',contype=0,conaffinity=0,density=0,rgba=[.75,.75,.78,1])
        for side in hocap.SIDES:
            spec.add_mesh(name=side,uservert=capture['hands'][side][0].ravel(),userface=hocap.mano_vertices(take,side)[1].ravel())
            spec.worldbody.add_geom(name=side,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=side,contype=0,conaffinity=0,density=0,rgba=SKIN)
        self.model=spec.compile();self.data=mujoco.MjData(self.model);self.renderer=mujoco.Renderer(self.model,height,width)
        for side in hocap.SIDES:
            g=self.model.geom(side).id;self.model.geom_pos[g]=0;self.model.geom_quat[g]=(1,0,0,0);self.model.geom_rbound[g]=0  # vertices are in world coordinates
        self.sides=hocap.SIDES
    def set_hand(self,side,vertices):
        np,m=self.np,self.model;mesh=m.mesh(side).id;start=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh]
        f=m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh]+m.mesh_facenum[mesh]]
        normal=np.zeros_like(vertices);cross=np.cross(vertices[f[:,1]]-vertices[f[:,0]],vertices[f[:,2]]-vertices[f[:,0]])
        for k in range(3):np.add.at(normal,f[:,k],cross)
        m.mesh_vert[start:start+n]=vertices
        m.mesh_normal[m.mesh_normaladr[mesh]:m.mesh_normaladr[mesh]+n]=normal/np.linalg.norm(normal,axis=1,keepdims=True).clip(1e-12)
        self.renderer._gl_context.make_current();self.mujoco.mjr_uploadMesh(m,self.renderer._mjr_context,mesh)
    def render(self,frame,camera):
        """RGB image of capture frame `frame` (30 Hz)."""
        c=self.capture
        for side in self.sides:self.set_hand(side,c['hands'][side][frame])
        self.data.mocap_pos[0]=c['object_pos'][frame];self.data.mocap_quat[0]=c['object_quat'][frame]
        self.mujoco.mj_forward(self.model,self.data);self.renderer.update_scene(self.data,camera)
        return self.renderer.render().copy()

def camera_for(object_pos):
    """Free camera looking at the mean object position, same view on both panels."""
    import mujoco
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    centre=object_pos.mean(0);camera.lookat[:]=(centre[0],centre[1],max(centre[2],.08))
    camera.distance=.8;camera.azimuth=34.6;camera.elevation=-25.
    return camera

def label(image,lines):
    import cv2
    cv2.rectangle(image,(0,0),(image.shape[1],12+30*len(lines)),(25,25,25),-1)
    for i,(text,color) in enumerate(lines):cv2.putText(image,text,(14,32+30*i),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2,cv2.LINE_AA)
    return image

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task',required=True);p.add_argument('--out',type=Path)
    p.add_argument('--fps',type=int,default=30);p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--width',type=int,default=800);p.add_argument('--height',type=int,default=600)
    a=p.parse_args();ensure_env();egl_default_display()
    import cv2,imageio,mujoco,numpy as np
    retarget=DATA/'retargeted'/'sharpa'/'bimanual'/a.task
    info=json.loads((retarget/'task_info.json').read_text())
    out=a.out or OUTPUTS/'retarget_videos'/f'{a.task}.mp4';out.parent.mkdir(parents=True,exist_ok=True)

    # Right: retargeted Sharpa scene, visual geoms only.
    model=mujoco.MjModel.from_xml_path(str(retarget/'0'/'scene_ik.xml'))
    model.vis.global_.offwidth=max(a.width,model.vis.global_.offwidth);model.vis.global_.offheight=max(a.height,model.vis.global_.offheight)
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name.startswith(('ref_','left_object')):model.geom_rgba[g,3]=0
    qpos=np.load(retarget/'0'/'trajectory_kinematic.npz')['qpos'];data=mujoco.MjData(model)
    robot=mujoco.Renderer(model,a.height,a.width);option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]

    person=HumanScene(info['source_relative_path'],a.width,a.height)
    camera=camera_for(person.capture['object_pos'])
    name=a.task.replace('fwd_capture__','')
    T=min(len(qpos),len(person.capture['object_pos']));frames=np.arange(0,(T-1)/30+1e-9,a.speed/a.fps)
    with imageio.get_writer(out,fps=a.fps,codec='libx264',quality=8,macro_block_size=8) as writer:
        for t in frames:
            i=min(int(round(t*30)),T-1)
            data.qpos[:]=qpos[i];mujoco.mj_forward(model,data);robot.update_scene(data,camera,scene_option=option)
            left=label(person.render(i,camera),[(f'HOCap capture (MANO + object)  t={i/30:5.2f}s',(255,255,255)),(name,(180,180,180))])
            right=label(robot.render().copy(),[(f'Retargeted Sharpa (kinematic)  t={i/30:5.2f}s',(255,255,255)),(f'frame {i}/{T-1}',(180,180,180))])
            writer.append_data(np.concatenate([left,right],axis=1))
    print(f'wrote {out} ({len(frames)} frames)')

if __name__=='__main__':main()
