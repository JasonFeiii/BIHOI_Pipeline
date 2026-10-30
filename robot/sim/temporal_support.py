"""Validate recorded handovers with the same temporal budgets as execution.

This reconstructs observations and state transitions; stored valid flags alone
cannot authorize a checkpoint. It does not change any physical limit.
"""
from pathlib import Path
import numpy as np
from handover_control import HandoverState, observe


def replay_measurements(rows):
    """Small deterministic core, also used by counterexample tests."""
    state = None
    phase_ok = False
    result = []
    previous_bottom = None
    for row in rows:
        if row is None:
            state = None
            phase_ok = False
            result.append(False)
            continue
        support = row['support']
        elapsed = row['elapsed']
        if state is None or elapsed == 1:
            reset_ok = state is None or state.finished(previous_bottom)
            state = HandoverState(support, np.asarray(row['anchor']).copy())
            phase_ok = bool(elapsed == 1 and reset_ok)
        ok, _ = state.observe_step(row['hits'], row['normal'], row['relative_both'],
                                   row['bottom'], row['vertical_speed'], row['table'])
        consistent = (state.support == support and state.elapsed_frames == elapsed
                      and state.table_frames == row['table_frames']
                      and np.allclose(state.anchor_local, row['anchor'], atol=1e-7, rtol=0)
                      and bool(row['table']) == bool(row['recorded_table']))
        phase_ok = bool(phase_ok and ok and consistent and row['force_valid'] and row['recorded_valid'])
        result.append(phase_ok)
        previous_bottom = row['bottom']
    return np.asarray(result, dtype=bool)


def verify_temporal_support(run, native, bottoms):
    import mujoco
    count = len(native['time'])
    result = np.zeros(count, dtype=bool)
    required = ('handover_control_valid', 'handover_support', 'handover_anchor_local',
                'handover_table_frames', 'handover_elapsed_frames', 'handover_table_contact')
    if any(k not in native for k in required):
        return result
    scenes = list(Path(run).glob('model/sharpa/bimanual/*/0/scene.xml'))
    if len(scenes) != 1:
        return result
    model = mujoco.MjModel.from_xml_path(str(scenes[0]))
    data = mujoco.MjData(model)
    objects = {g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body('right_object').id}
    hands = {g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('collision_hand_')}
    rows = []
    for i in range(count):
        support = int(native['handover_support'][i])
        if support not in (0, 1):
            rows.append(None)
            continue
        data.qpos[:] = native['qpos'][i]
        data.qvel[:] = native['qvel'][i]
        data.ctrl[:] = native['ctrl'][i]
        mujoco.mj_forward(model, data)
        o = observe(model, data, data.qpos, objects, hands, support)
        force_valid = True
        if native.get('actual_force_available', np.zeros(count, bool))[i]:
            force_valid = False
            keys = ('actual_hand_normal_force', 'actual_object_contact_force', 'actual_object_constraint_force')
            if all(k in native for k in keys):
                normal = native['actual_hand_normal_force'][i]
                residual = np.linalg.norm(native['actual_object_contact_force'][i] - native['actual_object_constraint_force'][i, :3])
                force_valid = bool(np.isfinite(normal).all() and np.isfinite(residual) and residual < .01)
                o['normal'] = normal
        rows.append(dict(support=support, elapsed=int(native['handover_elapsed_frames'][i]),
                         anchor=native['handover_anchor_local'][i], hits=o['hits'], normal=o['normal'],
                         relative_both=o['relative_both'], bottom=float(bottoms[i]),
                         vertical_speed=float(data.qvel[-10]), table=bool(o['table']),
                         recorded_table=bool(native['handover_table_contact'][i]),
                         table_frames=int(native['handover_table_frames'][i]),
                         recorded_valid=bool(native['handover_control_valid'][i]), force_valid=force_valid))
    return replay_measurements(rows)
