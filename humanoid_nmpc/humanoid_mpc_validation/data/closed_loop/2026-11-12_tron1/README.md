# 2026-11-12_tron1

The closed-loop runs of LimX Tron1 (`tron1`, centroidal MPC on two rolling contacts, ground-truth contact estimator and
state estimator) on its three scenarios, recorded on commit 79a84bef with the uncommitted stage 4 work. Each
document's provenance gives the worktree state, "264 changed paths, diff sha256 51097e844743b1d9", the machine (Intel
Core Ultra 7 265K, 20 logical cores, 30.2 GiB), GCC 13.3.0, HPIPM `GENERIC` and the container image `unknown`
(`VALIDATION_CONTAINER_IMAGE` was not passed). The runs used 4 solver threads:

```bash
make closed-loop-metrics ROBOT=tron1 LABEL=2026-11-12_tron1 RECORD_STATES=1
```

`RECORD_STATES=1` also wrote `data/benchmark/states/tron1_walk_0p5_states.txt`, which the solve benchmark replays.
This README lists the label's metrics; the package READMEs and the validation README link here.

| Metric | standing | walk_0p5 | turn_in_place_720 |
| --- | --- | --- | --- |
| survived | yes | yes | yes |
| base height mean / RMS error [m] | 0.7483 / 0.0002 | 0.7488 / 0.0026 | 0.7532 / 0.0050 |
| tilt RMS / max [rad] | 0.0004 / 0.0004 | 0.0048 / 0.0315 | 0.0035 / 0.0046 |
| velocity RMS error [m/s] | 0.0002 | 0.0798 | 0.0021 |
| yaw-rate RMS error [rad/s] | 0.0000 | 0.0001 | 0.0956 |
| heading final / max [rad] | 0.0001 / 0.0001 | 0.0016 / 0.0016 | -0.479 / 10.874 |
| stance-foot slip max / RMS [mm] | 0.003 / 0.003 | 1.2 / 1.2 | 458.9 / 458.3 |
| stance phases | 2 | 2 | 2 |
| joint torque RMS [N m] | 11.20 | 11.24 | 11.05 |
| initial-state gap rotation max [rad] / norm max | 7.0e-6 / 8.0e-5 | 0.0117 / 0.0775 | 0.0018 / 0.0088 |
| quaternion norm max deviation | 1.1e-16 | 1.1e-16 | 2.2e-16 |
| solve time mean / p99 / max [ms] | 2.02 / 3.07 / 3.68 | 2.13 / 3.25 / 3.75 | 2.16 / 3.18 / 3.74 |
| SQP steps full / reduced / zero | 800 / 0 / 0 | 1359 / 1 / 0 | 4181 / 1 / 0 |
| full steps refused above g_max, longest violation growth | 0, 3 | 1, 3 | 1, 3 |
| constraint violation mean / max | 6.4e-8 / 1.2e-7 | 6.2e-5 / 0.043 | 4.0e-5 / 0.0085 |
| failed solves, resets, full resets, simulator resets, unhealthy cycles | 0 | 0 | 0 |
| non-finite values | 0 | 0 | 0 |

Both wheels stay in measured contact through every run: each wheel has one stance phase, which never ends. The stance
phases count the touch-downs, so a wheel that lifted would add one.

In `walk_0p5` the gait table of `wheeled_gait.textproto` switches from `stance` to `drive`, and the robot rolls on its
wheels without a step. The base reaches the command within about 1 s, averages 0.506 m/s over the last 10 s of the 15 s
command (0.49 m/s over the whole 15 s, the start included), and stops 7.59 m from where the command started, 6 mm to
the side and 0.0016 rad off its heading. The 1.2 mm of slip over that drive is the distance the wheel material at the
contact point slid, the stance-foot slip of a rolling contact (`ContactSlipMeasure::kMaterialSlip`): a wheel's contact
point moves with the wheel, so its drift from touch-down is no slip measure.

In `turn_in_place_720` the base turns at about 0.45 rad/s against the 0.5 rad/s command (0.450 rad/s mean from 5 s to
20 s into the command), through 10.87 of the commanded 12.57 rad before the turn back, ends 0.48 rad off its starting
heading and stays within 26 mm of where it started. The slip of 0.46 m per wheel, about 9 mm/s over the 50 s of
turning, fits the wheels of the MuJoCo scene: each is a flat cylinder 5 cm wide (`urdf/tron1.xml`), which MuJoCo touches
to the floor at its two rims, 25 mm on either side of the wheel's middle plane (checked in MuJoCo 4.0.1 on the shipped
scene). Turning in place, the two rims travel circles of different radii at one wheel speed, so the wheel scrubs, and
the material on the middle plane, where the MPC's model puts the contact, slides at the yaw rate times the offset of
the rim that touches the floor on the outside of the turn, about 20 mm here. The same offset widens the turn's effective track beyond the model's,
which fits the yaw rate falling about 10 % short of the command.
The yaw-rate RMS error, 0.096 rad/s, is about ANYmal B's in the same scenario (0.094 rad/s, `2026-11-10_anymal_b`).

Standing, the base sits 0.2 mm below the nominal 0.7485 m and the wheels slip less than 3 micrometers.

The solve benchmark replaying this label's states (`make benchmark-mpc-solve ROBOT=tron1 LABEL=<label>`) measured
2.12 ms mean and 3.15 ms p99, 25 % of the 12.5 ms MPC period, on the states of the rerun (2.11 ms and 3.16 ms over
825 timed solves with none failed on the first recording's); its document was not kept.

The label is its own baseline for the bands: `make compare-closed-loop CANDIDATE=2026-11-12_tron1
BASELINE=2026-11-12_tron1 ROBOT=tron1` compares the three runs with no violation and no improvement, and Tron1 keeps the
default bands (`closedLoopMetricBandsFor()`).

The runs set the wheel kd of `robot_models/limx_tron1/tron1_centroidal_mpc/config/controller/joint_pd_gains.textproto`
(5 N m s/rad; 1 N m s/rad drove at 0.524 m/s against the 0.5 m/s command, its tilt RMS 0.0104 rad). The documents were
recorded again on the shipped files after later edits of that file and of `wheeled_gait.textproto`, so each document's
`configuration_sha256` matches the files as shipped. The rerun reproduced every metric of the first recording except the
solve times, which vary from run to run; the table lists the rerun's. The package README says which runs set which
values.
