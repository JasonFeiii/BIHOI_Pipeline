"""Process-local convergence experiment; does not edit mujoco_warp on disk.

The absolute gradient floor acknowledges float32 numerical resolution. It is
not a relaxation of collision, control, velocity, or contact safety limits.
The returned solve still needs an independent residual check at its cap.
"""
import warp as wp
import mujoco_warp._src.solver as solver

@wp.kernel
def residual_done(nv: int, opt_tolerance: wp.array(dtype=float), opt_iterations: int,
                  stat_meaninertia: float, efc_grad_dot_in: wp.array(dtype=float),
                  efc_cost_in: wp.array(dtype=float), efc_prev_cost_in: wp.array(dtype=float),
                  efc_done_in: wp.array(dtype=bool), solver_niter_out: wp.array(dtype=int),
                  efc_done_out: wp.array(dtype=bool), nsolving_out: wp.array(dtype=int)):
    world = wp.tid()
    if efc_done_in[world]:
        return
    solver_niter_out[world] += 1
    original_abs_tolerance = opt_tolerance[world % opt_tolerance.shape[0]] * stat_meaninertia * float(wp.max(nv, 1))
    gradient_norm = wp.sqrt(efc_grad_dot_in[world])
    # Mixed generalized coordinates: this numerical norm is not a physical
    # control certificate. The object force/torque residual is checked separately.
    converged = gradient_norm <= wp.max(0.001, original_abs_tolerance)
    if converged or solver_niter_out[world] >= opt_iterations:
        efc_done_out[world] = True
        wp.atomic_add(nsolving_out, 0, -1)

def install():
    previous = solver.solve_done
    solver.solve_done = residual_done
    return previous
