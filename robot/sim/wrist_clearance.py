"""One signed-distance policy for candidate, execution and resume boundaries."""
import mujoco

MIN_CLEARANCE_M = .001

def wrist_geometries(model):
    return [[g for g in range(model.ngeom)
             if model.geom_contype[g]
             and (model.body(model.geom_bodyid[g]).name or '') == side + '_arm_wrist_2_link']
            for side in ('right', 'left')]

def clearance(model, data, geoms=None):
    geoms = wrist_geometries(model) if geoms is None else geoms
    if not geoms[0] or not geoms[1]:
        raise ValueError('Missing cross-wrist collision geometry')
    return min(float(mujoco.mj_geomDistance(model, data, x, y, .02, None))
               for x in geoms[0] for y in geoms[1])

def allowed(model, data, geoms=None):
    return clearance(model, data, geoms) > MIN_CLEARANCE_M
