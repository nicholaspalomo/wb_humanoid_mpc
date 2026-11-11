# 2026-11-10_anymal_b

The closed-loop runs of ANYbotics ANYmal B (`anymal_b`, centroidal MPC, ground-truth contact estimator and state
estimator) on its three scenarios, recorded on commit 01991174 with the uncommitted stage 3 work. Each document's
provenance gives the worktree state, "128 changed paths, diff sha256 58a387208099a7fa", the machine (Intel Core Ultra
7 265K, 20 logical cores, 30.2 GiB), GCC 13.3.0, HPIPM `GENERIC` and the container image `unknown`
(`VALIDATION_CONTAINER_IMAGE` was not passed). The runs used 4 solver threads:

```bash
make closed-loop-metrics ROBOT=anymal_b LABEL=2026-11-10_anymal_b RECORD_STATES=1
```

`RECORD_STATES=1` also wrote `data/benchmark/states/anymal_b_walk_0p5_states.txt`, which the solve benchmark replays.
This README holds the label's metrics; the package READMEs and the validation README link here.

| Metric | standing | walk_0p5 | turn_in_place_720 |
| --- | --- | --- | --- |
| survived | yes | yes | yes |
| base height mean / RMS error [m] | 0.5171 / 0.0074 | 0.5110 / 0.0045 | 0.5141 / 0.0049 |
| tilt RMS / max [rad] | 0.0003 / 0.0003 | 0.0246 / 0.0818 | 0.0026 / 0.0060 |
| velocity RMS error [m/s] | 0.0000 | 0.1127 | 0.0019 |
| yaw-rate RMS error [rad/s] | 0.0000 | 0.0476 | 0.0940 |
| stance-foot slip max / RMS [mm] | 2.0 / 2.0 | 2.3 / 0.6 | 5.0 / 1.8 |
| joint torque RMS [N m] | 7.31 | 12.59 | 12.21 |
| solve time mean / p99 [ms] | 3.70 / 5.15 | 4.19 / 5.51 | 4.19 / 5.50 |
| SQP steps full / reduced / zero | 800 / 0 / 0 | 1357 / 3 / 0 | 4182 / 0 / 0 |
| failed solves, resets, simulator resets | 0 | 0 | 0 |

In `walk_0p5` the gait table of `quadruped_gait.textproto` goes from stance straight to the trot (LF_RH, then RF_LH).
The base averages 0.42 m/s over the 15 s of the 0.5 m/s command and 0.45 m/s over its last 10 s, and ends 0.11 rad
off its starting heading. An
earlier recording of this label, with static_walk between stance and trot and the horizontal linear momentum weight
at 150, walked about 2.4 s in static_walk and averaged 0.38 m/s (velocity RMS error 0.156 m/s). In `turn_in_place_720`
it turns through 24.6 of the commanded 25.1 rad (720 degrees and back) and ends 2 mm from where the command started.
The standing base height is 7 mm above the nominal 0.5097 m. About 2 mm of that is the feet: at the default posture
the shanks are tilted, so each foot sphere touches the ground about 2 mm below its `FOOT` frame.

The solve benchmark replaying this label's states (`make benchmark-mpc-solve ROBOT=anymal_b LABEL=<label>`) measured
4.20 ms mean and 8.78 ms p99, 70 % of the 12.5 ms MPC period; its document was not kept.

The label is its own baseline for the bands: `make compare-closed-loop CANDIDATE=2026-11-10_anymal_b
BASELINE=2026-11-10_anymal_b ROBOT=anymal_b` compares the three runs with no violation and no improvement, and ANYmal B
keeps the default bands (`closedLoopMetricBandsFor()`), since two runs of one configuration give the same documents.

The runs set the tuning of `robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc` and of the quadruped gait table
(the task file, the PD gains and `quadruped_gait.textproto` say which runs moved which value).
