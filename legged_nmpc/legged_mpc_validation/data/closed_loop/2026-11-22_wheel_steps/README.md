# 2026-11-22_wheel_steps

The closed-loop runs of the two wheeled configurations after their gait files switched to the event-driven steps of
`RollingStepScheduler` (`gait_scheduler: "rolling_lateral_debt"`,
[docs/rolling_contact](../../../../docs/rolling_contact/README.md), section 10). Each run is held to the label it was
last recorded under. The documents' provenance says `unknown` for the commit and the worktree state: they were recorded
in a git worktree of the checkout, whose `.git` file the dev container cannot follow. The machine is the Intel Core
Ultra 7 265K (20 logical cores, 30.2 GiB), GCC 13.3.0, HPIPM `GENERIC`. The runs used 4 solver threads, one at a time:

```bash
make closed-loop-metrics ROBOT=anymal_c_wheels LABEL=2026-11-22_wheel_steps BASELINE=2026-11-13_legged_rename
make closed-loop-metrics ROBOT=tron1 LABEL=2026-11-22_wheel_steps BASELINE=2026-11-13_legged_rename
```

| Configuration | Baseline | Runs compared | Violations | Improvements |
| --- | --- | --- | --- | --- |
| `anymal_c_wheels` | `2026-11-13_legged_rename` | 3 | 0 | 0 |
| `tron1` | `2026-11-13_legged_rename` | 3 | 0 | 0 |

The wheeled ANYmal C's `turn_in_place_720` now steps its diagonal pairs where the wheels' lateral debt reaches 5 cm, in
place of the gait file's `turning_trot`: it turns through 11.97 of 12.57 rad (11.78 before), with a yaw rate RMS error
of 0.083 rad/s (0.098) and a maximum tilt of 0.0097 rad (0.0119). `walk_0p5` drives on its four wheels and takes no
step, as before. Its new scenario `lateral_0p2`, 0.2 m/s to the left for 10 s, has no baseline: it survives, with 50
stance phases over its four wheels, and has a velocity RMS error of 0.042 m/s, a maximum tilt of 0.032 rad and a
heading drift of 0.046 rad.

Tron1's runs match their baseline: a drive or a turn in place builds no lateral debt, so it never steps in them. A first
recording, with the turn's pivot at the base rather than at the mean of the contacts, stepped Tron1 three times in
`turn_in_place_720` (a tilt of 0.078 rad against 0.0046 rad, outside the bands); its axle passes a few centimeters off
the base, and the scheduler now turns about the axle.
