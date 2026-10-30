"""Object momentum / contact-wrench response guided shooting.

This is a local model of the *real rollout*, including IK, motors, contacts and
friction. It never applies a force to the object. Its null-space is approximate,
not a certificate of force closure. Physical and numerical guards are external.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation


def reliable_rows(metrics, count):
    # 'solver_force_residual' is a magnitude in N, not a binary failure flag.
    # The original numerical guard owns its thresholds; do not invent new ones.
    failed=np.asarray(metrics['bad_solver'])
    if failed.shape!=(count,):raise ValueError('Missing per-candidate solver validity')
    return np.isfinite(failed)&(failed<1e-6)


def object_state(q, v, mass, ipos, inertia_body):
    q, v = np.asarray(q), np.asarray(v)
    r = Rotation.from_quat(q[..., [-10, -9, -8, -11]].reshape(-1, 4)).as_matrix()
    r = r.reshape(q.shape[:-1] + (3, 3))
    offset = np.einsum('...ij,j->...i', r, ipos)
    omega = np.einsum('...ij,...j->...i', r, v[..., -9:-6])
    com = q[..., -14:-11] + offset
    com_v = v[..., -12:-9] + np.cross(omega, offset)
    iw = r @ inertia_body @ np.swapaxes(r, -1, -2)
    return com, mass * com_v, np.einsum('...ij,...j->...i', iw, omega), r


def reference_state(q, dt, mass, ipos, inertia_body):
    r = Rotation.from_quat(q[:, [-10, -9, -8, -11]])
    com = q[:, -14:-11] + r.apply(np.broadcast_to(ipos, (len(q), 3)))
    vel = np.diff(com, axis=0) / dt
    omega = (r[1:] * r[:-1].inv()).as_rotvec() / dt
    vel = np.r_[vel[:1], vel]
    omega = np.r_[omega[:1], omega]
    iw = r.as_matrix() @ inertia_body @ r.as_matrix().transpose(0, 2, 1)
    return com, mass * vel, np.einsum('tij,tj->ti', iw, omega)


def ridge_response(x, y, desired, ref_direction, reliability, damping=.3):
    """One smooth regularized task inverse, plus a task-null regrasp direction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.sqrt(np.asarray(reliability, float).clip(.05, 1.))
    xm = np.average(x, axis=0, weights=w*w)
    ym = np.average(y, axis=0, weights=w*w)
    dx, dy = (x-xm)*w[:, None], (y-ym)*w[:, None]
    # z -> y; the ridge handles unobservable / contact-free action directions.
    a = np.linalg.solve(dx.T@dx + damping*np.eye(x.shape[1]), dx.T@dy).T
    b = ym-a@xm
    inv = np.linalg.solve(a@a.T+damping*np.eye(a.shape[0]), np.eye(a.shape[0]))
    inverse = a.T@inv
    null = np.eye(a.shape[1])-inverse@a
    task = inverse@(desired-b)
    mean = task + null@ref_direction
    # Unit normalized trust region; neither retries nor rounds enlarge noise.
    mean /= max(1., np.linalg.norm(mean)/np.sqrt(len(mean)))
    prediction = x@a.T+b
    return mean, null, dict(rank=int(np.linalg.matrix_rank(a)),
        fit_rmse=float(np.sqrt(np.mean((prediction-y)**2))),
        desired_error_before=float(np.linalg.norm(ym-desired)),
        predicted_error_after=float(np.linalg.norm(a@mean+b-desired)),
        task_step_norm=float(np.linalg.norm(task)),
        regrasp_null_norm=float(np.linalg.norm(null@ref_direction)),
        null_leak=float(np.linalg.norm(a@null@ref_direction)),
        singular_values=np.linalg.svd(a, compute_uv=False).tolist())


class WrenchSearch:
    def __init__(self, model, q, v, base, reference_path, raw_q, raw_p, raw_r,
                 nominal, dt, ctrl_steps, seed, beta, log, start_s=None):
        self.base=np.array(base, float, copy=True)
        self.h=len(base); self.dt=dt; self.ctrl_steps=ctrl_steps
        self.q0=np.array(q, copy=True); self.v0=np.array(v, copy=True)
        self.reference_path=np.array(reference_path, copy=True); self.beta=beta
        self.rng=np.random.default_rng(seed); self.log=log; self.round=0;self.start_s=start_s
        obj=model.body('right_object').id
        self.mass=float(model.body_mass[obj]);self.ipos=model.body_ipos[obj].copy()
        ir=Rotation.from_quat(model.body_iquat[obj][[1,2,3,0]]).as_matrix()
        self.inertia=ir@np.diag(model.body_inertia[obj])@ir.T
        self.c0,self.p0,self.l0,self.r0=object_state(q,v,self.mass,self.ipos,self.inertia)
        self.gc,self.gp,self.gl=reference_state(reference_path,dt,self.mass,self.ipos,self.inertia)
        self.p_scale=self.mass*.05
        self.l_scale=np.trace(self.inertia)/3*.35
        self.marks=tuple(sorted(set((ctrl_steps, self.h//2, self.h))))
        self.desired=np.concatenate([self.features(reference_path[k], self.gp[k], self.gl[k], self.gc[k], k) for k in self.marks])
        t=np.arange(1,self.h+1)*dt
        ramp=1.-np.exp(-t/(ctrl_steps*dt))
        phi=np.stack((ramp, ramp*(2*t/(self.h*dt)-1.)),axis=1)
        self.phi=phi/np.sqrt(np.mean(phi**2,axis=0))[None]/np.sqrt(2.)
        self.finger=[]
        for s in range(2):
            f=s*28+6
            z=np.asarray(nominal)[:,f:f+22]-self.base[:,f:f+22]
            _,_,u=np.linalg.svd(z,full_matrices=False)
            self.finger.append(u[:2]*np.sqrt(22.))
        self.finger=np.array(self.finger)
        if not np.array_equal(reference_path,raw_q):
            raise ValueError('This ablation requires the unmodified retarget object path')
        self.goals=np.array(nominal,copy=True)
        self.goal_r=np.array(raw_r[1:self.h+1],copy=True)
        for side in range(2):
            self.goals[:,side*28:side*28+3]=raw_p[1:self.h+1,side]
        self.ref_direction=self.encode_goals().reshape(-1)
        self.ref_direction/=max(1.,np.linalg.norm(self.ref_direction)/np.sqrt(len(self.ref_direction)))
        self.mean=np.zeros(32);self.null=np.eye(32)
        self.x=[];self.y=[];self.w=[];self.current={};self.fit_info=None

    def features(self,q,p,l,com,k):
        r=Rotation.from_quat(np.asarray(q)[...,[-10,-9,-8,-11]].reshape(-1,4))
        angle=(r*Rotation.from_matrix(self.r0).inv()).as_rotvec().reshape(np.asarray(q).shape[:-1]+(3,))
        # Momentum differences equal integrated *net* contact wrench after
        # subtracting gravity. Table contributions are not mislabeled as hand force.
        time=k*self.dt
        gravity=np.array([0.,0.,-9.81])*self.mass*time
        impulse=np.asarray(p)-self.p0-gravity
        angular=np.asarray(l)-self.l0
        return np.concatenate(((np.asarray(com)-self.c0)/.025,
            angle/np.deg2rad(15),impulse/self.p_scale,angular/self.l_scale),axis=-1)

    def encode_goals(self):
        dp=[];dr=[];fg=[]
        for s in range(2):
            j=s*28
            dp.append(self.goals[:,j:j+3]-self.base[:,j:j+3])
            dr.append((Rotation.from_matrix(self.goal_r[:,s])*Rotation.from_euler('XYZ',self.base[:,j+3:j+6]).inv()).as_rotvec())
            fg.append((self.goals[:,j+6:j+28]-self.base[:,j+6:j+28])@self.finger[s].T/(22*.1))
        shared_r=(dr[0]+dr[1])/2
        # Remove the first-order rigid rotational translation before encoding.
        for s in range(2):dp[s]=dp[s]-np.cross(shared_r,self.base[:,s*28:s*28+3]-self.gc[1:])
        spatial=np.c_[(dp[0]+dp[1])/2/.01,shared_r/.03,
            (dp[0]-dp[1])/2/.01,(dr[0]-dr[1])/2/.03,fg[0],fg[1]]
        return np.linalg.lstsq(self.phi,spatial,rcond=None)[0]

    def decode(self,z):
        z=np.asarray(z).reshape(-1,2,16)
        d=np.einsum('tk,nkj->ntj',self.phi,z)
        out=np.broadcast_to(self.base,(len(z),*self.base.shape)).copy()
        common=Rotation.from_rotvec((.03*d[:,:,3:6]).reshape(-1,3))
        center=np.broadcast_to(self.gc[None,1:],(len(z),self.h,3))
        for s,sign in enumerate((1.,-1.)):
            j=s*28
            lever=(out[:,:,j:j+3]-center).reshape(-1,3)
            out[:,:,j:j+3]=center+common.apply(lever).reshape(len(z),self.h,3)+.01*(d[:,:,:3]+sign*d[:,:,6:9])
            r=Rotation.from_rotvec((sign*.03*d[:,:,9:12]).reshape(-1,3))*common*Rotation.from_euler('XYZ',out[:,:,j+3:j+6].reshape(-1,3))
            a=r.as_euler('XYZ').reshape(len(z),self.h,3); anchor=out[:,:,j+3:j+6]
            out[:,:,j+3:j+6]=anchor+np.arctan2(np.sin(a-anchor),np.cos(a-anchor))
            out[:,:,j+6:j+28]+=.1*np.einsum('ntk,kj->ntj',d[:,:,12+s*2:14+s*2],self.finger[s])
        return out

    def propose(self,latent):
        import torch
        n=len(latent)-8
        eps=self.rng.standard_normal((n,32))
        # Reduced task-space variance, retained null-space exploration; max singular
        # scale <=1, so this is not expanded search.
        shape=.6*np.eye(32)+.4*self.null
        coeff=self.mean+self.beta**self.round*(eps@shape.T)
        coeff[0]=self.mean
        coeff[1]=np.zeros(32)
        samples=self.decode(coeff)
        # Retain fine finger residual exploration without coupling fingers to a
        # presumed force. The true rollout determines its contact effect.
        for s in range(2):
            j=s*28+6
            residual=(latent[8:,:,j:j+22]-latent[:1,:,j:j+22]).detach().cpu().numpy()
            samples[2:,:,j:j+22]+=.25*residual[2:]
        latent[8:]=torch.as_tensor(samples,device=latent.device,dtype=latent.dtype)
        self.last_coeff=coeff;self.current={}
        return latent

    def capture(self,q,v,k):
        if k not in self.marks:return
        q=q.detach().cpu().numpy().copy();v=v.detach().cpu().numpy().copy()
        c,p,l,_=object_state(q,v,self.mass,self.ipos,self.inertia)
        self.current[k]=self.features(q,p,l,c,k)

    def observe(self,score,metrics,valid):
        if set(self.current)!=set(self.marks):raise RuntimeError('Missing rollout momentum observation')
        y=np.concatenate([self.current[k] for k in self.marks],axis=1)[8:]
        reliable=np.isfinite(y).all(axis=1)
        reliable &= reliable_rows({'bad_solver':metrics['bad_solver'].detach().cpu().numpy()},len(score))[8:]
        w=np.where(valid.detach().cpu().numpy()[8:],1.,.2)
        self.x.extend(self.last_coeff[reliable]);self.y.extend(y[reliable]);self.w.extend(w[reliable])
        if len(self.x)>=40:
            self.mean,self.null,self.fit_info=ridge_response(np.array(self.x),np.array(self.y),self.desired,self.ref_direction,np.array(self.w))
        row=dict(time_s=self.start_s,round=self.round+1,trained_rows=len(self.x),numerically_reliable=int(reliable.sum()),
            physical_feasible=int(valid.sum()),fit=self.fit_info,search_noise_multiplier=float(self.beta**self.round),
            objective='raw_retarget',marks=list(self.marks),desired_features=self.desired.tolist(),
            learned_mean=self.mean.tolist(),scope='Local surrogate of IK+servo+physics; approximate null-space, not control certificate')
        with self.log.open('a') as f:f.write(json.dumps(row)+'\n')
        self.round+=1

    def momentum_cost(self,q,v,k):
        import torch
        from angular_tracking import rotate_body_to_world
        quat=q[:,-11:-7];omega=rotate_body_to_world(quat,v[:,-9:-6])
        key=(q.device,q.dtype)
        if not hasattr(self,'_momentum_tensors'):self._momentum_tensors={}
        if key not in self._momentum_tensors:
            self._momentum_tensors[key]=dict(
                ipos=torch.as_tensor(self.ipos,device=q.device,dtype=q.dtype),
                inertia=torch.as_tensor(self.inertia,device=q.device,dtype=q.dtype),
                gp=torch.as_tensor(self.gp,device=q.device,dtype=q.dtype),
                gl=torch.as_tensor(self.gl,device=q.device,dtype=q.dtype))
        cached=self._momentum_tensors[key]
        ipos=cached['ipos'].expand(len(q),-1)
        offset=rotate_body_to_world(quat,ipos)
        p=self.mass*(v[:,-12:-9]+torch.linalg.cross(omega,offset))
        body_i=cached['inertia']
        l=rotate_body_to_world(quat,v[:,-9:-6]@body_i.T)
        pt=cached['gp'][k]
        lt=cached['gl'][k]
        return .25*((p-pt).square().sum(-1)/self.p_scale**2+(l-lt).square().sum(-1)/self.l_scale**2).clamp(max=36.)
