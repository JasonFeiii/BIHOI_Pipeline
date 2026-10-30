"""Cache only the host-side broadphase function factory for static model keys."""
from functools import wraps

def factory_cache(original):
    cache={}
    @wraps(original)
    def cached(model):
        # These are exactly the m-fields referenced by wp.static in the
        # installed _broadphase_filter factory. Array values remain runtime inputs.
        key=(id(model),int(model.opt.broadphase_filter),tuple(model.geom_aabb.shape),
             tuple(model.geom_rbound.shape),tuple(model.geom_margin.shape))
        if key not in cache:cache[key]=(model,original(model))
        return cache[key][1]
    cached.cache=cache
    return cached

def install():
    import inspect
    import mujoco_warp._src.collision_driver as driver
    original=driver._broadphase_filter
    if getattr(original,'bihoi_static_factory_cache',False):return
    source=inspect.getsource(original)
    import hashlib
    if hashlib.sha256(source.strip().encode()).hexdigest()!='d3e6c97720ad9a9e34ff8b53fc75fab8a65e9e9b18f1e4d4d3b2f959edf4757d':
        raise RuntimeError('Broadphase factory source changed; cache requires revalidation')
    for required in ['m.geom_aabb.shape[0]','m.geom_rbound.shape[0]','m.geom_margin.shape[0]','m.opt.broadphase_filter']:
        if required not in source:raise RuntimeError('Unexpected broadphase factory implementation')
    wrapped=factory_cache(original);wrapped.bihoi_static_factory_cache=True
    driver._broadphase_filter=wrapped
    return original
