"""Soft guidance which permits one hand to release and regrasp."""
import torch
def one_hand_permitted_cost(side_touch,anchor_error,contact_required,anchor_weight):
    touch=side_touch.clamp(0,1)
    # Lack of all contacts is a soft risk signal, not a feasibility rule.
    # Only hands actually touching are encouraged to remain near their anchor.
    no_hands=(touch.sum(-1)<.5).to(touch.dtype)
    missing_cost=(2. if contact_required else .05)*no_hands
    return missing_cost+anchor_weight*(anchor_error*touch).sum(-1)
def no_hand_cost(side_touch,contact_required):
    return (2. if contact_required else .05)*(side_touch.sum(-1)<.5).to(side_touch.dtype)
