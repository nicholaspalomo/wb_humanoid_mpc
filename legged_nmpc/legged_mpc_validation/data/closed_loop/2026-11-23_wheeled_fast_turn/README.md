# 2026-11-23_wheeled_fast_turn

The wheeled ANYmal C with the `RollingStepScheduler` that steps the group with the largest debt when several groups are
overdue (commit `1ab49816`, clean), on the gait file's 0.15 s stance and 0.3 s swing. The robot now runs `turn_1radps`
and `arc` as well. The machine is the Intel Core Ultra 7 265K (20 logical cores, 30.2 GiB), GCC 13.3.0, HPIPM
`GENERIC`, 4 solver threads. The simulator of another session ran on the machine during these runs. The target ran
through `bazel test --jobs=2 --local_test_jobs=2` with the arguments of:

```bash
make closed-loop-metrics ROBOT=anymal_c_wheels LABEL=2026-11-23_wheeled_fast_turn BASELINE=2026-11-23_mujoco4
```

The four runs the baseline has show no violation and no improvement.

Before the fix, `turn_1radps` fell 3.1 s into the turn, at 6.67 s. The recorded contacts show LF and RH stepping every
0.45 s and no lift-off of RF or LH. At 1 rad/s a wheel builds 5.4 cm of debt in the 0.15 s stance, above the 5 cm bound,
so both groups were overdue at each lift-off. The scheduler broke that tie by the lead and chose LF + RH. The base sank
while the RF and LH wheels moved up to 0.54 m from their nominal places.

This label was first recorded on a 0.05 s stance (commit `17bff597`), which keeps the debt of one stance below the
bound. That recording is in the Git history. With the fix, the 0.15 s stance turns further at 1 rad/s and stands closer
to the nominal height of 0.636 m, so the gait file uses it.

| Scenario | Scheduler, stance | Survived | Mean height [m] | Max tilt [rad] | Heading [rad] | Yaw rate RMS error [rad/s] | Max stance slip [mm] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `turn_1radps` | before, 0.15 s | no | 0.497 | 0.300 | 1.96 | 0.511 | 97.4 |
| `turn_1radps` | fixed, 0.05 s | yes | 0.630 | 0.025 | 11.97 | 0.173 | 5.5 |
| `turn_1radps` | fixed, 0.15 s (this label) | yes | 0.638 | 0.014 | 12.57 | 0.171 | 8.2 |
| 1 m/s at 1 rad/s | before, 0.15 s | no | 0.443 | 0.261 | 1.11 | 1.194 | 780.7 |
| 1 m/s at 1 rad/s | fixed, 0.05 s | yes | 0.629 | 0.060 | 11.60 | 0.202 | 10.7 |
| 1 m/s at 1 rad/s | fixed, 0.15 s | yes | 0.637 | 0.080 | 12.29 | 0.192 | 19.5 |

The commands last 12.6 s. 1 m/s at 1 rad/s was a temporary scenario, so it has no document here. Each row is one run.
