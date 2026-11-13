# 2026-11-12_anymal_c_wheels

The closed-loop runs of the wheeled ANYbotics ANYmal C (`anymal_c_wheels`, centroidal MPC on four rolling contacts,
ground-truth contact estimator and state estimator) on its three scenarios, recorded on commit 79a84bef with the
uncommitted stage 4 work. Each document's provenance gives the worktree state, "269 changed paths, diff sha256
d8428687e3330470", the machine (Intel Core Ultra 7 265K, 20 logical cores, 30.2 GiB), GCC 13.3.0, HPIPM `GENERIC` and
the container image `unknown` (`VALIDATION_CONTAINER_IMAGE` was not passed). The runs used 4 solver threads and the gait
file `humanoid_common_mpc/config/command/wheeled_quadruped_gait.textproto`:

```bash
make closed-loop-metrics ROBOT=anymal_c_wheels LABEL=2026-11-12_anymal_c_wheels RECORD_STATES=1
```

`RECORD_STATES=1` also wrote `data/benchmark/states/anymal_c_wheels_walk_0p5_states.txt`, which the solve benchmark
replays. This README holds the label's metrics; the package READMEs and the validation README link here.

| Metric | standing | walk_0p5 | turn_in_place_720 |
| --- | --- | --- | --- |
| survived | yes | yes | yes |
| base height mean / RMS error [m] | 0.6343 / 0.0016 | 0.6338 / 0.0021 | 0.6347 / 0.0074 |
| tilt RMS / max [rad] | 0.0037 / 0.0038 | 0.0062 / 0.0278 | 0.0049 / 0.0119 |
| velocity RMS error [m/s] | 0.0003 | 0.0525 | 0.0349 |
| yaw-rate RMS error [rad/s] | 0.0000 | 0.0000 | 0.0982 |
| heading final / max [rad] | 0.0000 / 0.0000 | 0.0000 / 0.0000 | -0.094 / 11.776 |
| stance-foot slip max / RMS [mm] | 0.5 / 0.5 | 6.2 / 6.2 | 5.4 / 4.2 |
| stance phases | 4 | 4 | 232 |
| joint torque RMS [N m] | 18.70 | 18.88 | 25.43 |
| initial-state gap rotation max [rad] / norm max | 7.5e-6 / 5.6e-4 | 0.0011 / 0.0885 | 0.0018 / 0.0865 |
| quaternion norm max deviation | 1.1e-16 | 1.1e-16 | 2.2e-16 |
| solve time mean / p99 / max [ms] | 4.95 / 6.84 / 7.85 | 4.92 / 7.01 / 7.36 | 5.54 / 7.46 / 8.64 |
| SQP steps full / reduced / zero | 800 / 0 / 0 | 1360 / 0 / 0 | 4181 / 1 / 0 |
| full steps refused above g_max, longest violation growth | 0, 2 | 0, 2 | 1, 5 |
| constraint violation mean / max | 3.6e-5 / 5.5e-5 | 4.1e-5 / 0.0032 | 0.0022 / 0.10 |
| failed solves, resets, full resets, simulator resets, unhealthy cycles | 0 | 0 | 0 |
| non-finite values | 0 | 0 | 0 |

In `walk_0p5` the gait table switches from `stance` to `drive`, and the robot rolls on its four wheels without a step:
each wheel has one stance phase, which never ends (a wheel that lifted would add one). The base holds 0.496 m/s over
the last 10 s of the 15 s command (0.489 m/s over the whole command, the start included) and stops 7.44 m from where
the command started, 0.07 mm to the side and 8e-6 rad off its heading. The 6.2 mm of slip is the distance the wheel
material at the contact point slid over the drive (`ContactSlipMeasure::kMaterialSlip`).

In `turn_in_place_720` the gait table switches from `drive` to `turning_trot` on the yaw-rate command, and the robot
turns by stepping: diagonal pairs of wheels swing for 0.30 s, each swing followed by 0.15 s on all four wheels, 228
touch-downs in the 50 s of turning. The base turns at 0.484 rad/s against the 0.5 rad/s command (mean from 5 s to 20 s
into the command) through 11.78 of the commanded 12.57 rad before the turn back, ends 0.094 rad off its starting
heading and 12 mm from where it started, and stays within 0.14 m of it. The yaw-rate RMS error, 0.098 rad/s, is about
ANYmal B's in the same scenario (0.094 rad/s, `2026-11-10_anymal_b`) and Tron1's (0.096 rad/s, `2026-11-12_tron1`).

Standing, the base sits 1.6 mm below the nominal 0.63588 m with a static tilt of 0.0037 rad, and the wheels hold it
within 0.5 mm of slip.

The solve benchmark replaying this label's states (`make benchmark-mpc-solve ROBOT=anymal_c_wheels LABEL=<label>`)
measured 4.92 ms mean and 6.85 ms p99, 55 % of the 12.5 ms MPC period, over 825 timed solves with none failed; its
document was not kept.

The label is its own baseline for the bands: `make compare-closed-loop CANDIDATE=2026-11-12_anymal_c_wheels
BASELINE=2026-11-12_anymal_c_wheels ROBOT=anymal_c_wheels` compares the three runs with no violation and no
improvement, and the robot keeps the default bands (`closedLoopMetricBandsFor()`).

## How the turn was found

The configuration of the package unit, on `wheeled_gait.textproto` (stance and drive only) with wheel kd 1 N m s/rad,
survived all three scenarios but did not turn: in `turn_in_place_720` the heading reached 0.99 rad of the 12.57 rad
commanded, the base sank to 0.581 m mean (RMS error 0.059 m), the constraint violation averaged 0.12 and 3580 of 4182
full steps were refused above g_max; its walk_0p5 held 0.488 m/s and its standing run matched the one above. On four
parallel axles a yaw slides the front and hind wheels sideways at 0.36 m times the yaw rate, which the no-slip rows
forbid (`humanoid_nmpc/docs/rolling_contact/README.md`, section 8). The runs that followed, each on the turn:

| Change | Heading max [rad] | Tilt max [rad] | Outcome |
| --- | --- | --- | --- |
| `zero_velocity` soft (weight 100) instead of hard, drive only | 0.56 | 0.129 | survived; the solver planned the slip, but the wheels rolled at the rolling speed and the base did not turn |
| `trot` of `quadruped_gait.textproto` (no four-wheel phase), `task_space_foot_cost` listed | 4.75 | 0.871 | survived on the survival rule, but the base pitched by 0.7 rad within 2 s of turning and stayed down |
| the same with wheel kd 8 N m s/rad | 1.41 | 0.928 | fell at 25.0 s |
| `turning_trot` (0.15 s on four wheels after each swing), wheel kd 1 N m s/rad | 11.74 | 0.0127 | survived |
| the same with wheel kd 4 N m s/rad (shipped) | 11.78 | 0.0119 | survived; walk_0p5 at 0.496 m/s against 0.488 m/s with kd 1 |

The shipped values set by these runs are the gait file `wheeled_quadruped_gait.textproto`, `task_space_foot_cost` in
the task file's cost list and the wheel kd of 4 N m s/rad in `joint_pd_gains.textproto`; every other value of the
package was used as the package unit shipped it.
