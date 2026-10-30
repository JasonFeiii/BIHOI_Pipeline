# Local changes to Do as I Do `retargeting`

Source: [malik-group/do-as-i-do](https://github.com/malik-group/do-as-i-do) at
`824591b808c342b20079c3b4198a8c2bdf88c74e`, subdirectory `retargeting/retargeting/`, MIT license (`LICENSE`).
Only the Python package is vendored. The robot models under `assets/robots/` (STL/OBJ meshes) are not
committed; `python scripts/fetch_external.py do-as-i-do` checks out the same commit into
`.external/do-as-i-do`, and `robot/retarget.py` reads the Sharpa assets from there.

All files except the ones below are byte-identical to upstream (line endings normalized to LF).

| File | Change |
| --- | --- |
| `pipeline/solve_ik.py` | New `run_forward_validation=True` argument; `False` saves `trajectory_kinematic.npz` and skips the dynamics rollout (used by `robot/retarget.py`). |
| `utils/sampling.py` | Reward normalization uses `std(unbiased=False)`, so a top-k of one sample does not produce NaN weights. |
| `pipeline/optimize_contact_init.py` | New file (not upstream): open-hand contact initialization of the dual-arm bowl demo, used by `robot/sim/open_hand_pipeline.py`. Its data root is `robot_paths.DATA/demo/physics_contact_bowl`. |
