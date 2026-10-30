"""Attach the original BIHOI Sharpa hands to the repository's dual UR3e mounts."""
from pathlib import Path
from copy import deepcopy
import json
import math
import os
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robot_paths import DATA,OUTPUTS
DEMO_DATA=DATA/'demo'
# Task and its physics contact data (object scene, retargeted hands, contact schedule).
# Defaults are the bowl demo; prepare_task.py / run_batch.py set them for other takes.
TASK=os.environ.get('BIHOI_TASK','fwd_capture__8.27_big_steel_bowl_fwd__take01')
OLD=Path(os.environ.get('BIHOI_CONTACT_ROOT',DEMO_DATA/'physics_contact_bowl'))
# Output directory of one simulation run; start_demo.py sets it per launch.
ROOT=Path(os.environ.get('BIHOI_ARM_ROOT',OUTPUTS/'demo_runs'/'manual'))
UR=DEMO_DATA/'ur3e'
SEPARATION=.6604
BASE_HEIGHT=.0481
MOUNT_OFFSET=.03075
SIDES=('right','left')
YAW_OFFSETS={'right':-111.73,'left':-68.27}
MOUNT_YAWS={'right':135.,'left':45.}

def quat_z(angle):
    return f'{math.cos(angle/2):.17g} 0 0 {math.sin(angle/2):.17g}'

def build_scene(path,center=(0.,0.),heading=0.,separation=SEPARATION,base_height=BASE_HEIGHT):
    original=ET.parse(OLD/'sharpa/bimanual'/TASK/'0/scene.xml').getroot()
    original.set('model','bihoi_dual_ur3e_sharpa')
    original.find('compiler').set('meshdir',str(OLD/'assets'))
    old_asset=original.find('asset');world=original.find('worldbody')
    hand_defaults=ET.SubElement(original.find('default'),'default',{'class':'sharpa_original'})
    for element in list(original.find('default')):
        if element.tag!='default':hand_defaults.append(deepcopy(element))
    ur=ET.parse(UR/'ur3e.xml').getroot()
    defaults=deepcopy(ur.find('default/default'))
    classes={d.get('class'):'ur3e_'+d.get('class') for d in defaults.iter('default') if d.get('class')}
    assets={a.get('name'):'ur3e_'+a.get('name') for a in ur.find('asset') if a.get('name')}
    def shared_names(element):
        for e in element.iter():
            for attr in ['class','childclass']:
                if e.get(attr) in classes:e.set(attr,classes[e.get(attr)])
            for attr in ['mesh','material']:
                if e.get(attr) in assets:e.set(attr,assets[e.get(attr)])
    shared_names(defaults)
    # Arm collision primitives participate in arm/hand/object/environment contacts.
    ET.SubElement(defaults,'geom',contype='4',conaffinity='15',condim='3',friction='1 .1 .001')
    original.find('default').append(defaults)
    for source in ur.find('asset'):
        e=deepcopy(source)
        shared_names(e)
        if e.get('name'):e.set('name',assets[e.get('name')])
        if e.get('file'):e.set('file',str(UR/'assets'/e.get('file')))
        old_asset.append(e)
    old_act=original.find('actuator')
    finger_act={s:[deepcopy(a) for a in old_act if a.get('joint','').startswith(s+'_')
                   and not any(a.get('joint','').startswith(s+'_'+v) for v in ['pos_','rot_'])] for s in SIDES}
    old_act.clear()
    arm_meta={}
    lateral=np.array([-math.sin(heading),math.cos(heading)])
    for side in SIDES:
        hand=deepcopy(world.find(f".//body[@name='{side}_hand_C_MC']"))
        world.remove(world.find(f"body[@name='{side}_base_tx']"))
        arm=deepcopy(ur.find('worldbody/body'))
        shared_names(arm)
        prefix=side+'_arm_'
        for i,e in enumerate(arm.iter()):
            if e.get('name'):e.set('name',prefix+e.get('name'))
            elif e.tag=='geom':e.set('name',prefix+'geom_'+str(i))
            if e.tag=='geom':
                if e.get('mesh'):
                    e.set('type','mesh');e.set('group','2')
                elif e.get('class')=='ur3e_eef_collision':
                    e.set('type','cylinder');e.set('group','3')
                else:e.set('type','capsule');e.set('group','3')
        # Manufacturer elbow range is ±160°, narrower than the asset's ±180°.
        arm.find(f".//joint[@name='{prefix}elbow_joint']").set('range',f'{-math.radians(160)} {math.radians(160)}')
        arm.find(f".//joint[@name='{prefix}wrist_3_joint']").set('limited','false')
        sign=1 if side=='left' else -1
        base_xy=np.asarray(center)+sign*separation/2*lateral
        arm.set('pos',f'{base_xy[0]} {base_xy[1]} {base_height}')
        arm.set('quat',quat_z(heading+math.radians(YAW_OFFSETS[side])))
        wrist=arm.find(f".//body[@name='{prefix}wrist_3_link']")
        mount=ET.SubElement(wrist,'body',name=prefix+'coupler',pos='0 .09215 0',quat='-1 1 0 0')
        ET.SubElement(mount,'geom',name=prefix+'coupler_black',type='cylinder',
            size='.034 .006625',pos='0 0 .006625',rgba='.05 .05 .05 1',contype='4',conaffinity='15')
        ET.SubElement(mount,'geom',name=prefix+'coupler_silver',type='cylinder',
            size='.021 .00875',pos='0 0 .022',rgba='.79 .82 .93 1',contype='4',conaffinity='15')
        hand.set('pos',f'0 0 {MOUNT_OFFSET}')
        hand.set('quat',quat_z(math.radians(MOUNT_YAWS[side])))
        hand.set('childclass','sharpa_original')
        mount.append(hand)
        world.append(arm)
        for a0 in ur.find('actuator'):
            a=deepcopy(a0);shared_names(a)
            a.set('name',prefix+a.get('name'));a.set('joint',prefix+a.get('joint'))
            if a.get('joint').endswith('elbow_joint'):
                a.set('ctrlrange',f'{-math.radians(160)} {math.radians(160)}')
            if a.get('joint').endswith('wrist_3_joint'):a.set('ctrllimited','false')
            old_act.append(a)
        old_act.extend(finger_act[side])
        ET.SubElement(world,'geom',name=prefix+'mount_block',type='box',
            pos=f'{base_xy[0]} {base_xy[1]} {(.0381+base_height)/2}',size=f'.09 .09 {(base_height-.0381)/2}',
            quat=quat_z(heading),rgba='.13 .15 .17 1',contype='8',conaffinity='4')
        arm_meta[side]={'base_position':[float(base_xy[0]),float(base_xy[1]),base_height],
            'base_yaw_rad':heading+math.radians(YAW_OFFSETS[side]),'hand_mount_yaw_deg':MOUNT_YAWS[side]}
    # Preserve the original layout: right 28 hinges, left 28 hinges, two objects.
    for side in reversed(SIDES):
        body=world.find(f"body[@name='{side}_arm_base']")
        world.remove(body);world.insert(0,body)
    ET.SubElement(world,'geom',name='arm_breadboard',type='box',
        pos=f'{center[0]} {center[1]} {.0381/2}',size='.1524 .4572 .01905',
        quat=quat_z(heading),rgba='.14 .14 .14 1',contype='8',conaffinity='4')
    for g in original.iter('geom'):
        name=g.get('name','')
        if name.startswith('collision_hand_'):
            g.set('contype','2');g.set('conaffinity','4')
        elif name.startswith('right_object_'):
            g.set('contype','1');g.set('conaffinity','4')
        elif name=='floor':
            g.set('contype','8');g.set('conaffinity','4')
    key=original.find('keyframe')
    if key is not None:original.remove(key)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(original)
    ET.ElementTree(original).write(path,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(path))
    assert model.nq==70 and model.nv==68 and model.nu==56,(model.nq,model.nv,model.nu)
    assert not any((model.joint(j).name or '').startswith(('right_pos_','left_pos_','right_rot_','left_rot_')) for j in range(model.njnt))
    assert model.geom_contype[model.geom('right_hand_C_MC_visual').id]==0
    for side in SIDES:
        root_id=model.body(side+'_hand_C_MC').id
        assert model.body(model.body_parentid[root_id]).name==side+'_arm_coupler'
    # The actuator vector follows qpos for all 56 actuated hinge joints.
    assert np.array_equal(model.jnt_qposadr[model.actuator_trnid[:,0]],np.arange(56))
    return model,{'separation_m':separation,'center_xy':list(center),'heading_rad':heading,
        'base_height_above_table_m':base_height,'arms':arm_meta,'floating_wrist_joints':False,
        'joint_velocity_limits_rad_s':[math.pi]*3+[2*math.pi]*3,
        'limits_source':'https://www.universal-robots.com/manuals/EN/HTML/SW10_14/Content/prod-usr-man/complianceUR3e/H_g5_sections/appendix_g5/tech_spec_UR3e.htm'}

if __name__=='__main__':
    m,meta=build_scene(ROOT/'scene_prototype.xml')
    (ROOT/'mount_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
