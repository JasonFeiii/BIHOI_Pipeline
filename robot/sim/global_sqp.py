"""Whole-clip sparse SQP: wrist pose/derivative fit, joint bounds, and collisions."""
import argparse
import json
import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
import osqp
import torch
import mujoco
import mink
from build_arm_scene import ROOT,build_scene,SIDES
from arm_kinematics import ArmKinematics
from search_placement import trajectory_metrics,VELOCITY_LIMIT,evaluate_collisions

def setup_collision(model):
    arms={s:[g for g in range(model.ngeom) if (model.body(model.geom_bodyid[g]).name or '').startswith(s+'_arm_') and model.geom_contype[g]] for s in SIDES}
    environment=[g for g in range(model.ngeom) if model.geom_bodyid[g]==0 and model.geom_contype[g]]
    objects=[g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body('right_object').id]
    palms={s:[model.geom('collision_hand_'+s+'_palm_0').id] for s in SIDES}
    pairs=[(arms['right'],arms['right']),(arms['left'],arms['left']),(arms['right'],arms['left'])]
    for s in SIDES:pairs.extend([(arms[s],environment),(arms[s],objects),(arms[s],palms['left' if s=='right' else 'right'])])
    return mink.CollisionAvoidanceLimit(model,pairs,gain=1.,minimum_distance_from_collisions=.004,
        collision_detection_distance=.035),set(arms['right']+arms['left'])

def main(args):
    torch.set_num_threads(1);start=time.monotonic()
    source=json.loads((ROOT/f'mink_result_{args.candidate:03d}.json').read_text())
    ref=dict(np.load(ROOT/'reference_30hz.npz'));T=len(ref['time']);dt=1/30
    model,mount=build_scene(ROOT/f'sqp_scene_{args.candidate:03d}.xml',source['center_xy'],source['heading_rad'],base_height=source['height_m'])
    fullq=np.load(ROOT/source['trajectory_file'])['qpos'][:T].copy()
    assert len(fullq)==T
    kin=[ArmKinematics(model,s,'cpu',torch.float64) for s in SIDES]
    ids=np.r_[np.arange(6),np.arange(28,34)]
    dofs=ids.copy()
    q=fullq[:,ids].copy();qinitial=q[0].copy();N=T*12
    bp=[];br=[];zero=mujoco.MjData(model);mujoco.mj_kinematics(model,zero)
    for s in SIDES:
        b=model.body(s+'_arm_base').id;bp.append(zero.xpos[b].copy());br.append(zero.xmat[b].reshape(3,3).copy())
    collision,armgeoms=setup_collision(model)
    cfg=mink.Configuration(model,q=fullq[0])
    I=sp.eye(N,format='csc')
    d1=sp.diags([-np.ones(T-1),np.ones(T-1)],[0,1],shape=(T-1,T),format='csc')
    d2=sp.diags([np.ones(T-2),-2*np.ones(T-2),np.ones(T-2)],[0,1,2],shape=(T-2,T),format='csc')
    D1=sp.kron(d1,sp.eye(12),format='csc');D2=sp.kron(d2,sp.eye(12),format='csc')
    W0=sp.diags(np.tile([100]*3+[20]*3+[100]*3+[20]*3,T))
    W1=sp.diags(np.tile([6.32]*3+[1.26]*3+[6.32]*3+[1.26]*3,T-1))
    W2=sp.diags(np.tile([.224]*3+[.0448]*3+[.224]*3+[.0448]*3,T-2))
    lower=np.tile(np.r_[kin[0].lower.numpy(),kin[1].lower.numpy()],T)
    upper=np.tile(np.r_[kin[0].upper.numpy(),kin[1].upper.numpy()],T)
    vmax=np.tile(np.r_[VELOCITY_LIMIT,VELOCITY_LIMIT]*(np.array([1.])),T-1)*dt
    amax=np.full((T-2)*12,args.acceleration*dt*dt)
    history=[]
    def linearize(q):
        p=[];R=[];J=[]
        for s,k in enumerate(kin):
            pp,rr,jj=k.fk(torch.tensor(q[:,s*6:s*6+6]),True)
            pp=pp.numpy()@br[s].T+bp[s];rr=np.einsum('ij,tjk->tik',br[s],rr.numpy())
            jj=jj.numpy();jj[:,:3]=np.einsum('ij,tjk->tik',br[s],jj[:,:3]);jj[:,3:]=np.einsum('ij,tjk->tik',br[s],jj[:,3:])
            p.append(pp);R.append(rr);J.append(jj)
        e=np.c_[p[0]-ref['wrist_pos'][:,0],Rotation.from_matrix(R[0]@ref['wrist_rot'][:,0].transpose(0,2,1)).as_rotvec(),
                p[1]-ref['wrist_pos'][:,1],Rotation.from_matrix(R[1]@ref['wrist_rot'][:,1].transpose(0,2,1)).as_rotvec()]
        blocks=[]
        for t in range(T):
            j=np.zeros((12,12));j[:6,:6]=J[0][t];j[6:,6:]=J[1][t];blocks.append(j)
        return e,sp.block_diag(blocks,format='csc'),p,R
    for iteration in range(args.iterations):
        e,J,ps,Rs=linearize(q);ef=e.ravel()
        A0=W0@J;A1=W1@D1@J/dt;A2=W2@D2@J/(dt*dt)
        b0=W0@ef;b1=W1@D1@ef/dt;b2=W2@D2@ef/(dt*dt)
        H=A0.T@A0+A1.T@A1+A2.T@A2+.05*I
        gradient=A0.T@b0+A1.T@b1+A2.T@b2
        grow=[];hrow=[]
        for t in range(T):
            fullq[t,ids]=q[t];cfg.update(q=fullq[t])
            c=collision.compute_qp_inequalities(cfg,1.)
            if c.G is None:G=np.zeros((0,12));h=np.zeros(0)
            else:
                G=np.asarray(c.G)[:,dofs];h=np.asarray(c.h)
                valid=np.isfinite(h)&(np.linalg.norm(G,axis=1)>1e-10)
                G=G[valid];h=h[valid]
            extra=[];eh=[]
            for side,s in enumerate(SIDES):
                root=model.body(s+'_hand_C_MC').id;elb=model.body(s+'_arm_forearm_link').id
                jp=np.zeros((3,model.nv));je=np.zeros_like(jp)
                mujoco.mj_jacBody(model,cfg.data,jp,None,root);mujoco.mj_jacBody(model,cfg.data,je,None,elb)
                forward=ref['wrist_rot'][t,side,:,2]
                distance=float((cfg.data.xpos[root]-cfg.data.xpos[elb])@forward)
                extra.append(-forward@(jp-je)[:,dofs]);eh.append(distance-.01)
            grow.append(sp.csc_matrix(np.r_[G,np.array(extra)]));hrow.append(np.r_[h,eh])
        C=sp.block_diag(grow,format='csc');h=np.concatenate(hrow);M=len(h);K=len(amax)
        rho_c=min(1e9,1e6*2**iteration);rho_a=min(1e8,1e5*2**iteration)
        P=sp.block_diag([2*H,2*rho_c*sp.eye(M),2*rho_a*sp.eye(K)],format='csc')
        lin=np.r_[2*gradient,np.zeros(M+K)]
        zero_qc=sp.csc_matrix((N,M));zero_qa=sp.csc_matrix((N,K))
        trust=args.trust
        lb=np.maximum(-trust,lower-q.ravel());ub=np.minimum(trust,upper-q.ravel())
        lb[:12]=0;ub[:12]=0
        A_bounds=sp.hstack([I,zero_qc,zero_qa],format='csc')
        A_vel=sp.hstack([D1,sp.csc_matrix((D1.shape[0],M+K))],format='csc')
        A_coll=sp.hstack([C,-sp.eye(M),sp.csc_matrix((M,K))],format='csc')
        A_acc1=sp.hstack([D2,sp.csc_matrix((K,M)),-sp.eye(K)],format='csc')
        A_acc2=sp.hstack([-D2,sp.csc_matrix((K,M)),-sp.eye(K)],format='csc')
        A_slack=sp.hstack([sp.csc_matrix((M+K,N)),sp.eye(M+K)],format='csc')
        A=sp.vstack([A_bounds,A_vel,A_coll,A_acc1,A_acc2,A_slack],format='csc')
        d1q=D1@q.ravel();d2q=D2@q.ravel()
        l=np.r_[lb,-vmax-d1q,np.full(M,-np.inf),np.full(2*K,-np.inf),np.zeros(M+K)]
        u=np.r_[ub,vmax-d1q,h,amax-d2q,amax+d2q,np.full(M+K,np.inf)]
        solver=osqp.OSQP();solver.setup(P=sp.triu(P,format='csc'),q=lin,A=A,l=l,u=u,
            verbose=False,eps_abs=2e-5,eps_rel=2e-5,max_iter=8000,polishing=True)
        solution=solver.solve()
        if solution.x is None or solution.info.status_val not in [1,2]:
            print(json.dumps({'iteration':iteration,'status':solution.info.status}),flush=True)
            break
        dq=solution.x[:N].reshape(T,12);q+=dq
        fullq[:,ids]=q
        ep,_,pp,rr=linearize(q)
        native=evaluate_collisions(model,fullq,armgeoms)
        speed=float(np.max(abs(D1@q.ravel())/vmax))
        acceleration=float(np.max(abs(D2@q.ravel())/(dt*dt)))
        item={'iteration':iteration,'status':solution.info.status,'pose_scaled_cost':float(np.mean((W0@ep.ravel())**2)),
            'position_max_m':float(max(np.linalg.norm(pp[s]-ref['wrist_pos'][:,s],axis=-1).max() for s in range(2))),
            'joint_speed_ratio':speed,'joint_acceleration_max':acceleration,**native}
        history.append(item);print(json.dumps(item),flush=True)
        np.savez_compressed(ROOT/f'sqp_trajectory_{args.candidate:03d}.npz',qpos=fullq,time=ref['time'],source_frame=ref['source_frame'])
        (ROOT/f'sqp_progress_{args.candidate:03d}.json').write_text(json.dumps(history,indent=2))
        if float(np.max(abs(dq)))<1e-4 and native['arm_collision_contacts']==0:break
    e,J,pp,rr=linearize(q)
    metrics={}
    for s,side in enumerate(SIDES):
        target={key:ref[name][:,s] for key,name in [('p','wrist_pos'),('R','wrist_rot'),('v','wrist_vel'),('a','wrist_acc'),('w','wrist_angvel'),('alpha','wrist_angacc')]}
        metrics[side]=trajectory_metrics(pp[s],rr[s],target,dt)
    result={k:source[k] for k in ['center_xy','heading_rad','height_m']}
    result.update({'metrics':metrics,'history':history,'seconds':time.monotonic()-start,
        'collision':evaluate_collisions(model,fullq,armgeoms),'trajectory_file':f'sqp_trajectory_{args.candidate:03d}.npz','mount':mount})
    (ROOT/f'sqp_result_{args.candidate:03d}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,default=4)
    p.add_argument('--iterations',type=int,default=15);p.add_argument('--trust',type=float,default=.15)
    p.add_argument('--acceleration',type=float,default=20.)
    main(p.parse_args())
