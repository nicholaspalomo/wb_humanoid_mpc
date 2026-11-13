# 2026-11-11_g1_kalman

The runs `unitree_g1_kalman` is held to and its own, on the G1 scene with joint damping (c877a515) and the simulator's
accelerometer reading the mean over the control period (`MujocoSimConfig::accelerometerWindow`,
[robot_runtime/mujoco_sim_interface](../../../../../robot_runtime/mujoco_sim_interface/README.md#the-accelerometer)).
Recorded on commit a0c28bfb with the uncommitted accelerometer work. Each document's provenance gives the worktree state:
"18 changed paths, diff sha256 31bd7ee0238dda75" for `unitree_g1`, and "21 changed paths, diff sha256
0e715c12e0eb13a9" for `unitree_g1_kalman`, recorded second, with `unitree_g1`'s three documents in the label. The
machine was an Intel Core Ultra 7 265K (20 logical cores, 30.2 GiB), with GCC 13.3.0, HPIPM `GENERIC` and the container
image `unknown`. The runs are the Make target's test with `--jobs=4 --local_test_jobs=2`:

```bash
make closed-loop-metrics ROBOT=unitree_g1 LABEL=2026-11-11_g1_kalman "SCENARIO=standing:*/walk_0p5:*/turn_in_place_720"
make closed-loop-metrics ROBOT=unitree_g1_kalman LABEL=2026-11-11_g1_kalman BASELINE=2026-11-11_g1_kalman
```

`unitree_g1`'s documents equal those of `2026-11-08_g1_fall_fix` metric for metric: the ground-truth estimator does not
read the accelerometer. `unitree_g1_kalman` survives all three scenarios, and its comparison with `unitree_g1` within the
default bands found no violation and no improvement in any of them:

| Metric | standing: ground truth / kalman | walk_0p5 | turn_in_place_720 |
| --- | --- | --- | --- |
| survived | yes / yes | yes / yes | yes / yes |
| base height mean [m] | 0.7907 / 0.7907 | 0.7446 / 0.7454 | 0.7649 / 0.7667 |
| base height RMS error [m] | 0.0004 / 0.0004 | 0.0499 / 0.0490 | 0.0270 / 0.0252 |
| tilt RMS / max [rad] | 0.0002 / 0.0002, 0.0009 / 0.0009 | 0.0936 / 0.0906, 0.1454 / 0.1499 | 0.0458 / 0.0462, 0.0898 / 0.0920 |
| velocity RMS error [m/s] | 0.0005 / 0.0005 | 0.3776 / 0.3793 | 0.1312 / 0.1326 |
| yaw-rate RMS error [rad/s] | 0.0000 / 0.0000 | 0.7707 / 0.7879 | 0.5005 / 0.5002 |
| stance-foot slip max [mm] | 0.0001 / 0.0115 | 4.5 / 2.8 | 2.7 / 2.5 |
| solve time p99 [ms] | 9.88 / 10.25 | 10.35 / 10.28 | 10.41 / 10.28 |

The slip of `unitree_g1_kalman` is read from the controller's observation, so it includes the drift of the estimated
horizontal base position. Against the truth, the estimate of the base ended 0.01 mm (standing, 10 s), 1.5 mm
(`walk_0p5`, 17 s) and 9.3 mm (`turn_in_place_720`, 52 s) away, and the RMS error of the estimated base velocity was
below 0.0001, 0.5 to 1.2 and 0.09 to 0.6 cm/s per axis. These come from a per-cycle log of a run of the same code with a
temporary logger, not from the documents.

Label `2026-11-07_kalman_variant` held this variant to bands of its own, a tilt of 0.04 rad and a slip of 0.2 m; with
these runs it is held to the default bands and must survive `walk_0p5` like every configuration.
