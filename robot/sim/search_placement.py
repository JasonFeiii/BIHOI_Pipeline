"""Fixed-spacing dual-arm placement search using actual achieved wrist trajectories."""
import argparse
import itertools
import json
import time
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from build_arm_scene import ROOT,SEPARATION,BASE_HEIGHT,SIDES,YAW_OFFSETS,build_scene
from arm_kinematics import ArmKinematics
from prepare_reference import derivatives

VELOCITY_LIMIT=np.r_[np.full(3,np.pi),np.full(3,2*np.pi)]
SEEDS=np.array([[0,-1.57,1.57,-1.57,-1.57,0],[-1.57,-1.57,1.57,-1.57,-1.57,0],
 [1.57,-1.57,-1.57,-1.57,1.57,0],[3.14,-1.57,-1.57,-1.57,1.57,0],
 [-1.8308,-1.4903,-.2515,-2.9927,-1.4881,3.9406],
 [2.0238,-1.5676,.1041,-.0717,1.6592,-3.9320]],dtype=np.float32)

def base_poses(center,heading,height=None):
    centers=np.asarray(center);heading=np.asarray(heading)
    lateral=np.stack([-np.sin(heading),np.cos(heading)],-1)
    p=[];R=[]
    for side,sign in [('right',-1),('left',1)]:
        xy=centers+sign*SEPARATION/2*lateral
        p.append(np.c_[xy,np.full(len(xy),BASE_HEIGHT) if height is None else np.asarray(height)])
        angles=heading+np.deg2rad(YAW_OFFSETS[side])
        R.append(Rotation.from_rotvec(np.c_[np.zeros(len(angles)),np.zeros(len(angles)),angles]).as_matrix())
    return np.stack(p,1),np.stack(R,1)

def update_bases(model,center,heading,height=BASE_HEIGHT):
    bp,br=base_poses(np.asarray(center)[None],np.asarray([heading]),np.asarray([height]))
    for s,side in enumerate(SIDES):
        bid=model.body(side+'_arm_base').id
        model.body_pos[bid]=bp[0,s]
        quat=Rotation.from_matrix(br[0,s]).as_quat(scalar_first=True)
        model.body_quat[bid]=quat
        gid=model.geom(side+'_arm_mount_block').id
        model.geom_pos[gid,:2]=bp[0,s,:2]
        model.geom_pos[gid,2]=(.0381+height)/2
        model.geom_size[gid,2]=(height-.0381)/2
        model.geom_quat[gid]=Rotation.from_euler('z',heading).as_quat(scalar_first=True)
    gid=model.geom('arm_breadboard').id
    model.geom_pos[gid,:2]=center
    model.geom_pos[gid,2]=.0381/2
    model.geom_size[gid,2]=.0381/2
    model.geom_quat[gid]=Rotation.from_euler('z',heading).as_quat(scalar_first=True)

def unwrap(q):
    delta=np.diff(q,axis=0)
    wrapped=(delta+np.pi)%(2*np.pi)-np.pi
    return np.r_[q[:1],q[:1]+np.cumsum(wrapped,axis=0)]

def trajectory_metrics(p,R,target,dt):
    v,a,w,alpha=derivatives(p,R,dt)
    pe=np.linalg.norm(p-target['p'],axis=-1)
    re=Rotation.from_matrix(R@target['R'].transpose(0,2,1)).magnitude()
    ve=np.linalg.norm(v-target['v'],axis=-1);ae=np.linalg.norm(a-target['a'],axis=-1)
    we=np.linalg.norm(w-target['w'],axis=-1);ale=np.linalg.norm(alpha-target['alpha'],axis=-1)
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    return {'position_rmse_m':rms(pe),'position_max_m':float(pe.max()),
        'rotation_rmse_rad':rms(re),'rotation_max_rad':float(re.max()),
        'velocity_rmse_m_s':rms(ve),'acceleration_rmse_m_s2':rms(ae),
        'angular_velocity_rmse_rad_s':rms(we),'angular_acceleration_rmse_rad_s2':rms(ale)}

def evaluate_collisions(model,qpos,arm_geom_ids):
    data=mujoco.MjData(model);minimum=.01;bad=0;first=None
    for t,q in enumerate(qpos):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        for c in data.contact:
            if int(c.geom[0]) in arm_geom_ids or int(c.geom[1]) in arm_geom_ids:
                minimum=min(minimum,float(c.dist))
                if c.dist<-.0001:
                    bad+=1
                    if first is None:first=t
    return {'min_arm_clearance_m':minimum,'arm_collision_contacts':bad,'first_collision_frame':first}

def main(args):
    torch.set_num_threads(4);torch.manual_seed(123)
    started=time.monotonic()
    model,_=build_scene(ROOT/'scene_prototype.xml')
    kin=[ArmKinematics(model,s) for s in SIDES]
    ref=dict(np.load(ROOT/'reference_30hz.npz'))
    # A necessary embodiment correction: UR wrist housings cannot occupy the
    # same space even when the human wrists fit. Keep original targets for all
    # metrics, and solve the nearest symmetric separation of the tool centers.
    target_pos=ref['wrist_pos'].copy()
    centers=target_pos-ref['wrist_rot'][:,:,:,2]*.0529
    separation=centers[:,0]-centers[:,1]
    distance=np.linalg.norm(separation,axis=1)
    offset=np.maximum(.083-distance,0)[:,None]*.5*separation/np.maximum(distance[:,None],1e-8)
    target_pos[:,0]+=offset;target_pos[:,1]-=offset
    (ROOT/'wrist_housing_projection.json').write_text(json.dumps({
        'bounding_center_clearance_m':.083,'maximum_per_hand_shift_m':float(np.linalg.norm(offset,axis=1).max()),
        'reference_poses_for_metrics_unchanged':True},indent=2))
    info=json.loads((ROOT/'reference_info.json').read_text())
    center=np.asarray(info['mean_wrist_position'])[:2]
    fwd=np.asarray(info['mean_hand_forward'])[:2]
    theta0=float(np.arctan2(fwd[1],fwd[0]))
    candidates=[]
    for height,heading,back,lateral in itertools.product([BASE_HEIGHT,.15,.25],theta0+np.deg2rad(np.linspace(-45,45,args.headings)),
                                                 np.linspace(.22,.50,args.distances),
                                                 np.linspace(-.12,.12,args.lateral)):
        f=np.array([np.cos(heading),np.sin(heading)]);left=np.array([-f[1],f[0]])
        c=center-back*f+lateral*left
        # Both fixed bases must be behind both wrist trajectories in the layout's forward direction.
        bp,_=base_poses(c[None],np.asarray([heading]))
        margins=np.einsum('tsi,i->ts',ref['wrist_pos'][:,:,:2]-c,f)
        if margins.min()<.12:continue
        candidates.append({'center_xy':c.tolist(),'height_m':height,'heading_rad':float(heading),'behind_margin_m':float(margins.min())})
    uniform=np.unique(np.linspace(0,len(ref['time'])-1,args.frames).round().astype(int))
    dynamic=np.linalg.norm(ref['wrist_vel'],axis=-1).max(axis=1)+.05*np.linalg.norm(ref['wrist_angvel'],axis=-1).max(axis=1)
    fast=np.argsort(dynamic)[-50:]
    indices=np.unique(np.r_[uniform,fast,np.clip(fast-1,0,len(dynamic)-1),np.clip(fast+1,0,len(dynamic)-1)])
    C=len(candidates);F=len(indices);S=len(SEEDS)
    bp,br=base_poses(np.array([c['center_xy'] for c in candidates]),np.array([c['heading_rad'] for c in candidates]),np.array([c['height_m'] for c in candidates]))
    all_q=[];all_scores=[];all_maxp=[];all_maxr=[]
    dt=np.diff(ref['time'][indices])[:,None,None]
    for side,k in enumerate(kin):
        targetp=np.einsum('cij,cfj->cfi',br[:,side].transpose(0,2,1),target_pos[indices,side][None]-bp[:,side,None])
        targetR=np.einsum('cij,fjk->cfik',br[:,side].transpose(0,2,1),ref['wrist_rot'][indices,side])
        tp=torch.tensor(np.repeat(targetp[:,:,None],S,axis=2).reshape(-1,3),device='cuda',dtype=torch.float32)
        tr=torch.tensor(np.repeat(targetR[:,:,None],S,axis=2).reshape(-1,3,3),device='cuda',dtype=torch.float32)
        seed=torch.tensor(np.broadcast_to(SEEDS,(C,F,S,6)).copy().reshape(-1,6),device='cuda')
        q,ep,er=k.ik(tp,tr,seed,iterations=args.iterations)
        qs=q.cpu().numpy().reshape(C,F,S,6)
        ep=ep.cpu().numpy().reshape(C,F,S);er=er.cpu().numpy().reshape(C,F,S)
        dq=(np.diff(qs,axis=1)+np.pi)%(2*np.pi)-np.pi
        velocity=dq/dt[None]
        joint_pen=np.mean(np.maximum(abs(velocity)/VELOCITY_LIMIT-1,0)**2,axis=(1,3))*10
        jump_pen=np.mean(np.maximum(abs(dq)-.8,0)**2,axis=(1,3))*10
        scores=np.mean((ep/.005)**2+(er/np.deg2rad(2))**2,axis=1)+joint_pen+jump_pen
        scores+=np.max(ep,axis=1)/.01+np.max(er,axis=1)/.1
        all_q.append(qs);all_scores.append(scores)
        all_maxp.append(ep.max(axis=1));all_maxr.append(er.max(axis=1))
    ranking=[]
    for ci,c in enumerate(candidates):
        choices=[np.argsort(scores[ci])[:2] for scores in all_scores]
        for rs,ls in itertools.product(*choices):
            if max(all_maxp[0][ci,rs],all_maxp[1][ci,ls])>.02 or max(all_maxr[0][ci,rs],all_maxr[1][ci,ls])>.2:continue
            ranking.append((float(all_scores[0][ci,rs]+all_scores[1][ci,ls]),ci,int(rs),int(ls)))
    if not ranking:
        # A coarse approximation is not a physics success. Keep the nearest
        # finite IK candidates for continuous constrained refinement instead
        # of crashing on an empty list; final errors and collisions are saved.
        diagnostics=[]
        for ci,c in enumerate(candidates):
            choices=[np.argsort(scores[ci])[:2] for scores in all_scores]
            for rs,ls in itertools.product(*choices):
                score=float(all_scores[0][ci,rs]+all_scores[1][ci,ls])
                if not np.isfinite(score):continue
                ranking.append((score,ci,int(rs),int(ls)))
                diagnostics.append({'coarse_score':score,'candidate_index':ci,
                    'position_max_m':[float(all_maxp[0][ci,rs]),float(all_maxp[1][ci,ls])],
                    'rotation_max_rad':[float(all_maxr[0][ci,rs]),float(all_maxr[1][ci,ls])]})
        diagnostics.sort(key=lambda x:x['coarse_score'])
        (ROOT/'coarse_approximation_warning.json').write_text(json.dumps({
            'reason':'No candidate passed coarse pose thresholds; nearest approximations forwarded to continuous constrained refinement.',
            'physics_acceptance_unchanged':True,'best_candidates':diagnostics[:12]},indent=2))
        print(json.dumps({'stage':'coarse_fallback','best':diagnostics[:3]}),flush=True)
    if not ranking:raise RuntimeError('No finite IK candidate in placement search')
    ranking.sort()
    (ROOT/'coarse_ranking.json').write_text(json.dumps([
        dict(candidates[ci],score=score,seeds=[rs,ls]) for score,ci,rs,ls in ranking],indent=2))
    print(json.dumps({'stage':'coarse_complete','placements':C,'branches_evaluated':C*2*S,'seconds':time.monotonic()-started}),flush=True)
    arms={g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith(('right_arm_','left_arm_'))
          and model.geom_contype[g]}
    model.geom_margin[list(arms)]=.01;model.geom_gap[list(arms)]=.01
    collision_rank=[]
    used=set();uniform_rows=np.searchsorted(indices,uniform)
    for coarse,ci,rs,ls in ranking:
        if ci in used:continue
        c=candidates[ci];update_bases(model,c['center_xy'],c['heading_rad'],c['height_m'])
        q=ref['qpos'][uniform].copy()
        q[:,kin[0].qadr]=all_q[0][ci,uniform_rows,rs]
        q[:,kin[1].qadr]=all_q[1][ci,uniform_rows,ls]
        collision=evaluate_collisions(model,q,arms)
        collision_rank.append((collision['arm_collision_contacts']*1e5+coarse,coarse,ci,rs,ls))
        if collision['arm_collision_contacts']==0:used.add(ci)
    collision_rank.sort()
    ranking=[(coarse,ci,rs,ls) for _,coarse,ci,rs,ls in collision_rank]
    print(json.dumps({'stage':'collision_prefilter','checked_bases':len(ranking),
        'zero_collision_candidates':sum(x[0]<1e5 for x in collision_rank)}),flush=True)
    results=[]
    for rank,(coarse,ci,rs,ls) in enumerate(ranking[:args.finalists]):
        c=candidates[ci];update_bases(model,c['center_xy'],c['heading_rad'],c['height_m'])
        qout=ref['qpos'].copy();metrics={};elbow_margin=[];speed_ratio=[]
        for side,k in enumerate(kin):
            sid=[rs,ls][side]
            seed=np.repeat(SEEDS[sid][None],len(ref['time']),axis=0)
            tp=np.einsum('ij,fj->fi',br[ci,side].T,target_pos[:,side]-bp[ci,side])
            tr=np.einsum('ij,fjk->fik',br[ci,side].T,ref['wrist_rot'][:,side])
            q,ep,er=k.ik(torch.tensor(tp,device='cuda',dtype=torch.float32),
                        torch.tensor(tr,device='cuda',dtype=torch.float32),
                        torch.tensor(seed,device='cuda',dtype=torch.float32),iterations=80,step_cap=.25,damping=.001)
            q=unwrap(q.cpu().numpy());qout[:,k.qadr]=q
            p,R,elbow=k.fk(torch.tensor(q,device='cuda',dtype=torch.float32),elbow=True)
            p=p.cpu().numpy()@br[ci,side].T+bp[ci,side]
            R=np.einsum('ij,fjk->fik',br[ci,side],R.cpu().numpy())
            elbow=elbow.cpu().numpy()@br[ci,side].T+bp[ci,side]
            elbow_margin.append(float(np.einsum('fi,fi->f',p-elbow,ref['wrist_rot'][:,side,:,2]).min()))
            target={key:ref[name][:,side] for key,name in [('p','wrist_pos'),('R','wrist_rot'),('v','wrist_vel'),('a','wrist_acc'),('w','wrist_angvel'),('alpha','wrist_angacc')]}
            metrics[SIDES[side]]=trajectory_metrics(p,R,target,1/30)
            speed_ratio.append(float(np.max(abs(np.diff(q,axis=0))*30/VELOCITY_LIMIT)))
        coll=evaluate_collisions(model,qout[indices],arms)
        score=sum((v['position_rmse_m']/.005)**2+(v['rotation_rmse_rad']/np.deg2rad(2))**2+
            .1*(v['velocity_rmse_m_s']/.05)**2+.02*(v['acceleration_rmse_m_s2']/1)**2+
            .1*(v['angular_velocity_rmse_rad_s']/.25)**2+.02*(v['angular_acceleration_rmse_rad_s2']/5)**2 for v in metrics.values())
        score+=1e5*coll['arm_collision_contacts']+1e3*sum(max(.02-x,0) for x in elbow_margin)
        score+=100*sum(max(v-1,0)**2 for v in speed_ratio)
        item=dict(c,score=float(score),coarse_score=coarse,seeds=[rs,ls],metrics=metrics,
                  elbow_behind_margin_m=elbow_margin,joint_velocity_limit_ratio=speed_ratio,**coll)
        results.append(item)
        np.savez_compressed(ROOT/f'candidate_{rank:03d}.npz',qpos=qout)
        item['trajectory_file']=f'candidate_{rank:03d}.npz'
        (ROOT/'placement_progress.json').write_text(json.dumps({'completed':rank+1,'total':min(args.finalists,len(ranking)),'best_score':min(r['score'] for r in results)},indent=2))
        print(json.dumps({'candidate':rank,'score':score,'collisions':coll['arm_collision_contacts'],'speed_ratio':speed_ratio,'elbow_margin':elbow_margin}),flush=True)
    results.sort(key=lambda x:x['score'])
    # Independently check every original frame for the best candidates.
    for item in results[:5]:
        update_bases(model,item['center_xy'],item['heading_rad'],item['height_m'])
        q=np.load(ROOT/item['trajectory_file'])['qpos']
        full=evaluate_collisions(model,q,arms)
        item['full_frame_collision_check']=full
        if full['arm_collision_contacts']==0:
            best=item;break
    else:best=results[0]
    best_model,mount=build_scene(ROOT/'best_scene.xml',best['center_xy'],best['heading_rad'],base_height=best['height_m'])
    best['mount']=mount;best['search_seconds']=time.monotonic()-started
    (ROOT/'placement_results.json').write_text(json.dumps(results,indent=2))
    (ROOT/'best_placement.json').write_text(json.dumps(best,indent=2))
    q=np.load(ROOT/best['trajectory_file'])['qpos']
    np.savez_compressed(ROOT/'best_arm_reference.npz',qpos=q,time=ref['time'],source_frame=ref['source_frame'])
    print(json.dumps({'stage':'finished','best':best},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--headings',type=int,default=9)
    p.add_argument('--distances',type=int,default=5);p.add_argument('--lateral',type=int,default=5)
    p.add_argument('--frames',type=int,default=41);p.add_argument('--iterations',type=int,default=45)
    p.add_argument('--finalists',type=int,default=24)
    main(p.parse_args())
