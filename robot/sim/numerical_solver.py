"""Pyramidal Newton convergence fix and object numerical-integrity checks.
All changes are process-local. Existing physical feasibility gates are unchanged.
"""
import torch,warp as wp,mujoco
from mujoco_warp._src import solver
from solver_convergence import residual_done
from stable_linesearch import linesearch_delta64

def install(model):
    if int(model.opt.cone)!=int(mujoco.mjtCone.mjCONE_PYRAMIDAL) or int(model.opt.solver)!=int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise RuntimeError('Validated numerical solver only supports pyramidal Newton')
    solver.solve_done=residual_done
    solver.linesearch_iterative=linesearch_delta64
    return dict(version='delta_cost64_gradient_v50',model_scope='pyramidal Newton',
        method='Equivalent objective increments and derivatives in float64 line search; primal residual stopping',
        gradient_abs_floor=.001,max_iterations=int(model.opt.iterations),
        diagnostic_force_tolerance_N=.01,diagnostic_torque_tolerance_Nm=.001,
        guard='Reject numerically inconsistent candidates and rollback entire uncommitted execution window; no relaxed physical thresholds')

def residual(data):
    g=wp.to_torch(data.efc.Ma)[:,-12:-6]-wp.to_torch(data.qfrc_smooth)[:,-12:-6]-wp.to_torch(data.qfrc_constraint)[:,-12:-6]
    f=torch.linalg.vector_norm(g[:,:3],dim=-1);t=torch.linalg.vector_norm(g[:,3:],dim=-1)
    ok=torch.isfinite(g).all(dim=-1)&(f<.01)&(t<.001)
    return ok,f,t
