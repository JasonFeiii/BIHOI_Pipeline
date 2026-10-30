"""Causal joint-command shaping used inside prediction and execution alike."""
import torch


def smooth_targets(targets, q0, v0, dt, velocity_limit, acceleration_limit,
                   acceleration0=None, jerk_limit=None, tracking_rate=25.):
    """Keep the accepted boundary state; bounds are per-second quantities.

    Arm mode uses a damped tracker with bounded acceleration change. Finger
    mode uses a velocity tracker. Neither changes the tracking reference or
    any simulated state. Hard speed/position bounds take precedence at limits.
    """
    q = q0.expand_as(targets[..., 0, :]).clone()
    v = v0.expand_as(q).clone()
    a = torch.zeros_like(q) if acceleration0 is None else acceleration0.expand_as(q).clone()
    out = []
    for i in range(targets.shape[-2]):
        error = targets[..., i, :] - q
        if jerk_limit is None:
            goal_v = (tracking_rate * error).clamp(-velocity_limit, velocity_limit)
            next_a = ((goal_v - v) / dt).clamp(-acceleration_limit, acceleration_limit)
        else:
            target_v = v0 if i == 0 else (targets[..., i, :] - targets[..., i-1, :]) / dt
            target_v = target_v.clamp(-velocity_limit, velocity_limit)
            goal_a = (tracking_rate**2 * error + 2*tracking_rate*(target_v-v)).clamp(-acceleration_limit, acceleration_limit)
            next_a = torch.maximum(torch.minimum(goal_a, a + jerk_limit*dt), a - jerk_limit*dt)
        next_v = (v + next_a*dt).clamp(-velocity_limit, velocity_limit)
        a = (next_v-v)/dt
        q = q + next_v*dt
        v = next_v
        out.append(q)
    return torch.stack(out, dim=-2)
