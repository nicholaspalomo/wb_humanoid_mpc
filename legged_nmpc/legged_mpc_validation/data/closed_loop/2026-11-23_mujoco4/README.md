# 2026-11-23_mujoco4

The closed-loop runs of the five recorded configurations after the simulator moved from MuJoCo 3.3.4 to 4.0.1
(`lib/mujoco_vendor`). Each run is held to the label it was last recorded under. The documents' provenance says
`unknown` for the commit and the worktree state: they were recorded in a git worktree of the checkout, whose `.git`
file the dev container cannot follow. The machine is the Intel Core Ultra 7 265K (20 logical cores, 30.2 GiB), GCC
13.3.0, HPIPM `GENERIC`. The runs used 4 solver threads, one at a time:

```bash
make closed-loop-metrics ROBOT=unitree_g1 LABEL=2026-11-23_mujoco4 BASELINE=2026-11-13_legged_rename
make closed-loop-metrics ROBOT=unitree_g1_kalman LABEL=2026-11-23_mujoco4 BASELINE=2026-11-13_legged_rename
make closed-loop-metrics ROBOT=anymal_b LABEL=2026-11-23_mujoco4 BASELINE=2026-11-13_legged_rename
make closed-loop-metrics ROBOT=anymal_c_wheels LABEL=2026-11-23_mujoco4 BASELINE=2026-11-22_wheel_steps
make closed-loop-metrics ROBOT=tron1 LABEL=2026-11-23_mujoco4 BASELINE=2026-11-22_wheel_steps
```

| Configuration | Baseline | Runs compared | Violations | Improvements |
| --- | --- | --- | --- | --- |
| `unitree_g1` | `2026-11-13_legged_rename` | 7 | 0 | 0 |
| `unitree_g1_kalman` | `2026-11-13_legged_rename` | 3 | 0 | 0 |
| `anymal_b` | `2026-11-13_legged_rename` | 3 | 0 | 0 |
| `anymal_c_wheels` | `2026-11-22_wheel_steps` | 4 | 0 | 0 |
| `tron1` | `2026-11-22_wheel_steps` | 3 | 0 | 0 |

Every run survives, and all but two print the survival, velocity and yaw rate errors, maximum tilt, stance-foot slip
and final heading of their baseline to four digits. `unitree_g1_kalman` in `walk_0p5` ends with a heading of 0.236 rad
(0.186 before), a velocity RMS error of 0.380 m/s (0.379), a maximum tilt of 0.146 rad (0.150) and a maximum
stance-foot slip of 3.2 mm (2.8 mm); it is the one configuration that estimates its state from the simulated
sensors. `tron1` in `standing` differs in the fourth digit.
