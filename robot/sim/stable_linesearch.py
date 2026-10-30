"""Isolated pyramidal solver line-search experiment, not installed runtime code.
Line-search structure adapted from mujoco_warp under Apache-2.0.
Evaluate objective differences rather than subtracting large float32 costs.
"""
import warp as wp
from mujoco_warp._src import types

@wp.func
def safe64(a:wp.float64,b:wp.float64)->wp.float64:
    if b==wp.float64(0.0):
        return a/wp.float64(types.MJ_MINVAL)
    return a/b

@wp.func
def bracket64(x:wp.vec3d,y:wp.vec3d)->bool:
    return (x[1]<y[1] and y[1]<wp.float64(0.0)) or (x[1]>y[1] and y[1]>wp.float64(0.0))

@wp.func
def gauss_delta(quad:wp.vec3,a:wp.float64)->wp.vec3d:
    b=wp.float64(quad[1]);c=wp.float64(quad[2])
    return wp.vec3d(a*(b+a*c),b+wp.float64(2.0)*a*c,wp.float64(2.0)*c)

@wp.func
def huber64(r:wp.float64,D:wp.float64,f:wp.float64)->wp.float64:
    bound=safe64(f,D)
    if wp.abs(r)<bound:
        return wp.float64(0.5)*D*r*r
    return f*(wp.abs(r)-wp.float64(0.5)*bound)

@wp.func
def row_delta(r:wp.float64,j:wp.float64,D:wp.float64,f:wp.float64,kind:int,a:wp.float64)->wp.vec3d:
    z=a*j;x=r+z
    delta=wp.float64(0.0);slope=wp.float64(0.0);curvature=wp.float64(0.0)
    if kind==0:
        delta=wp.float64(0.5)*D*z*(wp.float64(2.0)*r+z)
        slope=D*x*j;curvature=D*j*j
    elif kind==1:
        bound=safe64(f,D)
        if wp.abs(r)<bound and wp.abs(x)<bound:
            delta=wp.float64(0.5)*D*z*(wp.float64(2.0)*r+z)
        elif r>=bound and x>=bound:
            delta=f*z
        elif r<=-bound and x<=-bound:
            delta=-f*z
        else:
            delta=huber64(x,D,f)-huber64(r,D,f)
        if wp.abs(x)<bound:
            slope=D*x*j;curvature=D*j*j
        elif x<=-bound:slope=-f*j
        else:slope=f*j
    else:
        if r<wp.float64(0.0) and x<wp.float64(0.0):
            delta=wp.float64(0.5)*D*z*(wp.float64(2.0)*r+z)
        elif r<wp.float64(0.0):delta=-wp.float64(0.5)*D*r*r
        elif x<wp.float64(0.0):delta=wp.float64(0.5)*D*x*x
        if x<wp.float64(0.0):slope=D*x*j;curvature=D*j*j
    return wp.vec3d(delta,slope,curvature)

@wp.func
def eval_delta(contact_friction_in:wp.array(dtype=types.vec5),contact_efc_address_in:wp.array2d(dtype=int),
    ne_clip:int,nef_clip:int,nefc_clip:int,impratio_invsqrt:float,type_in:wp.array(dtype=int),id_in:wp.array(dtype=int),
    D_in:wp.array(dtype=float),frictionloss_in:wp.array(dtype=float),Jaref_in:wp.array(dtype=float),jv_in:wp.array(dtype=float),quad_in:wp.array(dtype=wp.vec3),alpha:wp.float64)->wp.vec3d:
    out=wp.vec3d(wp.float64(0.0))
    for i in range(nefc_clip):
        kind=int(2)
        if i<ne_clip:kind=0
        elif i<nef_clip:kind=1
        out+=row_delta(wp.float64(Jaref_in[i]),wp.float64(jv_in[i]),wp.float64(D_in[i]),wp.float64(frictionloss_in[i]),kind,alpha)
    return out

@wp.func
def eval_three(contact_friction_in:wp.array(dtype=types.vec5),contact_efc_address_in:wp.array2d(dtype=int),
    ne_clip:int,nef_clip:int,nefc_clip:int,impratio_invsqrt:float,type_in:wp.array(dtype=int),id_in:wp.array(dtype=int),
    D_in:wp.array(dtype=float),frictionloss_in:wp.array(dtype=float),Jaref_in:wp.array(dtype=float),jv_in:wp.array(dtype=float),quad_in:wp.array(dtype=wp.vec3),a:wp.float64,b:wp.float64,c:wp.float64):
    x=eval_delta(contact_friction_in,contact_efc_address_in,ne_clip,nef_clip,nefc_clip,impratio_invsqrt,type_in,id_in,D_in,frictionloss_in,Jaref_in,jv_in,quad_in,a)
    y=eval_delta(contact_friction_in,contact_efc_address_in,ne_clip,nef_clip,nefc_clip,impratio_invsqrt,type_in,id_in,D_in,frictionloss_in,Jaref_in,jv_in,quad_in,b)
    z=eval_delta(contact_friction_in,contact_efc_address_in,ne_clip,nef_clip,nefc_clip,impratio_invsqrt,type_in,id_in,D_in,frictionloss_in,Jaref_in,jv_in,quad_in,c)
    return x,y,z

@wp.kernel
def linesearch_delta64(
  # Model:
  nv: int,
  opt_impratio: wp.array(dtype=float),
  opt_tolerance: wp.array(dtype=float),
  opt_ls_tolerance: wp.array(dtype=float),
  opt_ls_iterations: int,
  stat_meaninertia: float,
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_efc_address_in: wp.array2d(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_search_dot_in: wp.array(dtype=float),
  efc_jv_in: wp.array2d(dtype=float),
  efc_quad_in: wp.array2d(dtype=wp.vec3),
  efc_quad_gauss_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  njmax_in: int,
  # Data out:
  efc_alpha_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  impratio = opt_impratio[worldid % opt_impratio.shape[0]]
  efc_type = efc_type_in[worldid]
  efc_id = efc_id_in[worldid]
  efc_D = efc_D_in[worldid]
  efc_frictionloss = efc_frictionloss_in[worldid]
  efc_Jaref = efc_Jaref_in[worldid]
  efc_jv = efc_jv_in[worldid]
  efc_quad = efc_quad_in[worldid]
  efc_quad_gauss = efc_quad_gauss_in[worldid]
  tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
  ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
  ne_clip = min(njmax_in, ne_in[worldid])
  nef_clip = min(njmax_in, ne_clip + nf_in[worldid])
  nefc_clip = min(njmax_in, nefc_in[worldid])
  impratio_invsqrt = 1.0 / wp.sqrt(impratio)

  # Calculate p0
  snorm = wp.math.sqrt(efc_search_dot_in[worldid])
  scale = stat_meaninertia * wp.float(wp.max(1, nv))
  gtol = wp.float64(tolerance * ls_tolerance * snorm * scale)
  p0 = gauss_delta(efc_quad_gauss,wp.float64(0.0))
  p0 += eval_delta(
    contact_friction_in,
    contact_efc_address_in,
    ne_clip,
    nef_clip,
    nefc_clip,
    impratio_invsqrt,
    efc_type,
    efc_id,
    efc_D,
    efc_frictionloss,
    efc_Jaref,
    efc_jv,
    efc_quad,
    wp.float64(0.0),
  )

  # Calculate lo bound
  lo_alpha_in = -safe64(p0[1], p0[2])
  lo_in = gauss_delta(efc_quad_gauss, lo_alpha_in)
  lo_in += eval_delta(
    contact_friction_in,
    contact_efc_address_in,
    ne_clip,
    nef_clip,
    nefc_clip,
    impratio_invsqrt,
    efc_type,
    efc_id,
    efc_D,
    efc_frictionloss,
    efc_Jaref,
    efc_jv,
    efc_quad,
    lo_alpha_in,
  )

  # Initialize bounds
  lo_less = lo_in[1] < p0[1]
  lo = wp.where(lo_less, lo_in, p0)
  lo_alpha = wp.where(lo_less, lo_alpha_in, wp.float64(0.0))
  hi = wp.where(lo_less, p0, lo_in)
  hi_alpha = wp.where(lo_less, wp.float64(0.0), lo_alpha_in)

  # Launch main linesearch iterative loop
  alpha = wp.float64(0.0)
  for _ in range(opt_ls_iterations):
    lo_next_alpha = lo_alpha - safe64(lo[1], lo[2])
    hi_next_alpha = hi_alpha - safe64(hi[1], hi[2])
    mid_alpha = wp.float64(0.5) * (lo_alpha + hi_alpha)

    lo_next, hi_next, mid = eval_three(
      contact_friction_in,
      contact_efc_address_in,
      ne_clip,
      nef_clip,
      nefc_clip,
      impratio_invsqrt,
      efc_type,
      efc_id,
      efc_D,
      efc_frictionloss,
      efc_Jaref,
      efc_jv,
      efc_quad,
      lo_next_alpha,
      hi_next_alpha,
      mid_alpha,
    )
    lo_next += gauss_delta(efc_quad_gauss, lo_next_alpha)
    hi_next += gauss_delta(efc_quad_gauss, hi_next_alpha)
    mid += gauss_delta(efc_quad_gauss, mid_alpha)

    # swap lo:
    swap_lo_lo_next = bracket64(lo, lo_next)
    lo = wp.where(swap_lo_lo_next, lo_next, lo)
    lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
    swap_lo_mid = bracket64(lo, mid)
    lo = wp.where(swap_lo_mid, mid, lo)
    lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
    swap_lo_hi_next = bracket64(lo, hi_next)
    lo = wp.where(swap_lo_hi_next, hi_next, lo)
    lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
    swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

    # swap hi:
    swap_hi_hi_next = bracket64(hi, hi_next)
    hi = wp.where(swap_hi_hi_next, hi_next, hi)
    hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
    swap_hi_mid = bracket64(hi, mid)
    hi = wp.where(swap_hi_mid, mid, hi)
    hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
    swap_hi_lo_next = bracket64(hi, lo_next)
    hi = wp.where(swap_hi_lo_next, lo_next, hi)
    hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
    swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

    # if we did not adjust the interval, we are done
    # also done if either low or hi slope is nearly flat
    ls_done = (not swap_lo and not swap_hi) or (lo[1] < wp.float64(0.0) and lo[1] > -gtol) or (hi[1] > wp.float64(0.0) and hi[1] < gtol)

    # update alpha if we have an improvement
    improved = lo[0] < p0[0] or hi[0] < p0[0]
    lo_better = lo[0] < hi[0]
    alpha = wp.where(improved and lo_better, lo_alpha, alpha)
    alpha = wp.where(improved and not lo_better, hi_alpha, alpha)
    if ls_done:
      break

  efc_alpha_out[worldid] = float(alpha)
