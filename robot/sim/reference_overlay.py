"""Display raw retargeted Sharpa hands and object, aligned by source frame."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp


class ReferenceOverlay:
    def __init__(self, server, live, raw_scene, raw_trajectory):
        self.server=server;self.live=live
        raw_scene=Path(raw_scene)
        original=mujoco.MjModel.from_xml_path(str(raw_scene))
        tree=ET.parse(raw_scene);xml=tree.getroot()
        compiler=xml.find('compiler')
        for attribute in ['meshdir','texturedir']:
            if compiler is not None and compiler.get(attribute):
                directory=Path(compiler.get(attribute))
                if not directory.is_absolute():compiler.set(attribute,str((raw_scene.parent/directory).resolve()))
        # Keep inertias while removing unneeded render geometry, so the
        # original floating-hand kinematic tree and joint ordering survive.
        for body in xml.iter('body'):
            name=body.get('name')
            if name and body.find('inertial') is None:
                bid=original.body(name).id
                if original.body_mass[bid]>0:
                    ET.SubElement(body,'inertial',mass=str(original.body_mass[bid]),
                        pos=' '.join(map(str,original.body_ipos[bid])),
                        quat=' '.join(map(str,original.body_iquat[bid])),
                        diaginertia=' '.join(map(str,original.body_inertia[bid])))
        for parent in xml.iter():
            for child in list(parent):
                if child.tag!='geom':continue
                name=child.get('name','')
                keep=name.startswith('collision_hand_') or name=='right_object_visual'
                if not keep:parent.remove(child)
                else:child.set('group','0')
        # Contact pairs are irrelevant for a kinematics-only visualization.
        contact=xml.find('contact')
        if contact is not None:xml.remove(contact)
        self.spec=mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'))
        self.model=self.spec.compile();self.data=mujoco.MjData(self.model)
        for body in self.spec.bodies:
            for geom in body.geoms:
                if geom.name:
                    gid=self.model.geom(geom.name).id
                    geom.type=mujoco.mjtGeom(int(self.model.geom_type[gid]));geom.group=0
        self.raw=np.load(raw_trajectory)['qpos'].copy()
        for offset in [0,28]:self.raw[:,offset+3:offset+6]=np.unwrap(self.raw[:,offset+3:offset+6],axis=0)
        self.raw[:,-7:]=[0,0,-2,1,0,0,0]
        source_frames=np.arange(len(self.raw),dtype=float)
        self.object_rotation=Slerp(source_frames,Rotation.from_quat(self.raw[:,[60,61,62,59]]))
        self.handles=[[],[]]
        before=len(live._STATE.ref_geom_handles);before_info=len(live._STATE.ref_geom_info)
        blue=np.array([.12,.5,1.,.28],dtype=np.float32)
        for body in self.spec.bodies:
            if not body.geoms:continue
            category=1 if body.name=='right_object' else 0
            entry=live._build_one_ref_body(self.spec,self.model,body,server,'retarget_reference',blue)
            self.handles[category].append(entry)
        # Own these overlay handles independently of the main scene's model.
        # Main and reference trees have different body IDs (arms vs free wrists).
        del live._STATE.ref_geom_handles[before:]
        del live._STATE.ref_geom_info[before_info:]
        self.show_hands=live._STATE.scene_checkboxes['kin_hand']
        self.show_object=live._STATE.scene_checkboxes['kin_obj']
        self.show_hands.on_update(lambda _:self.update_visibility())
        self.show_object.on_update(lambda _:self.update_visibility())
        with live._STATE.gui_folder_kinematic:
            server.gui.add_markdown('**Reference：蓝色半透明**\n\nHand / Object 开关控制未经过物理优化的 retarget 参考；与当前回放按源帧对齐。')
        self.reference_time=None;self.source_map=None;self.source_path=None;self.source_frame=0.
        self.update_visibility()

    def update_visibility(self):
        for category,visible in enumerate([self.show_hands.value,self.show_object.value]):
            for handle,_ in self.handles[category]:handle.visible=bool(visible)

    def load_run(self,path):
        path=Path(path)
        if self.source_path==path:return
        if not path.exists():return
        with np.load(path) as z:
            self.reference_time=z['time'].copy()
            self.source_map=z['source_frame'].copy() if 'source_frame' in z else self.reference_time*30
        self.source_path=path

    def render(self,time_s):
        frame=time_s*30 if self.source_map is None else np.interp(time_s,self.reference_time,self.source_map)
        frame=float(np.clip(frame,0,len(self.raw)-1));self.source_frame=frame
        lo=int(np.floor(frame));hi=min(lo+1,len(self.raw)-1);weight=frame-lo
        q=(1-weight)*self.raw[lo]+weight*self.raw[hi]
        q[59:63]=self.object_rotation(frame).as_quat(scalar_first=True)
        self.data.qpos[:]=q;mujoco.mj_kinematics(self.model,self.data)
        with self.server.atomic():
            for entries in self.handles:
                for handle,bid in entries:
                    handle.position=tuple(self.data.xpos[bid]);handle.wxyz=tuple(self.data.xquat[bid])

