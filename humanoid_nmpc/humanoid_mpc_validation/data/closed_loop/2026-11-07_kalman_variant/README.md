# 2026-11-07_kalman_variant

The run the bands of `unitree_g1_kalman` were set from (`closedLoopMetricBandsFor()`, `MetricBands.cpp`), recorded on
commit e1c06dae with the uncommitted stage 2 estimator work (the worktree state is in each document's provenance):

```bash
make closed-loop-metrics ROBOT=unitree_g1 LABEL=2026-11-07_kalman_variant "SCENARIO=standing:*/walk_0p5:*/turn_in_place_720"
make closed-loop-metrics ROBOT=unitree_g1_kalman LABEL=2026-11-07_kalman_variant BASELINE=2026-11-07_kalman_variant
```

`unitree_g1` (ground truth) survived `standing` and fell 12.2 s into `walk_0p5` and 52.1 s into `turn_in_place_720`
(tilt past 1 rad). `unitree_g1_kalman` survived `standing` and fell 1.9 s into `walk_0p5` and 1.6 s into
`turn_in_place_720` (the base below half its initial height). Against `unitree_g1`, its `standing` run is worse in tilt
(RMS 0.0280 against 0.0014 rad) and in stance-foot slip (0.153 m against 1e-7 m: the drift of the estimated base
position, since the metric reads the controller's observation); the other metrics are within the default bands. The
walking and turning runs are compared on survival alone, since the baseline fell.
