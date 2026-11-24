# 2026-11-23_wheeled_fast_turn

The wheeled ANYmal C after its gait file's `rolling_lateral_debt.min_stance_duration` went from 0.15 s to 0.05 s, with
the two scenarios the robot now runs as well, `turn_1radps` (its 1 rad/s command limit) and `arc`. The documents'
provenance says commit `5f1e4f58`, clean; the commit was then amended to `17bff597` for clang-format alone. The machine
is the Intel Core Ultra 7 265K (20 logical cores, 30.2 GiB), GCC 13.3.0, HPIPM `GENERIC`, 4 solver threads:

```bash
make closed-loop-metrics ROBOT=anymal_c_wheels LABEL=2026-11-23_wheeled_fast_turn BASELINE=2026-11-23_mujoco4
```

The four runs the baseline has show no violation and one improvement: `lateral_0p2`'s tilt RMS is 0.0141 rad against
0.0204 rad. `standing` and `walk_0p5` take no step and are unchanged.

On the 0.15 s stance, `turn_1radps` fell 3.1 s into the turn: the base sank steadily from 0.636 m and dropped below
half its height at 6.67 s. By the gait's timing each diagonal pair stayed down for 0.6 s between its steps, a lateral
debt of 22 cm against 14 cm on the 0.05 s stance (legged_nmpc/docs/rolling_contact/README.md, section 10). The joint
angles, foot positions and step times were not recorded, so the limit it ran into is not identified. A drive combined
with a fast yaw (1 m/s at 1 rad/s) was not run. A 0.25 s swing on the 0.05 s stance
survived too, at a mean height of 0.617 m and a maximum tilt of 0.050 rad, so the swing stays at 0.3 s.

| Scenario | Survived | Mean height [m] | Max tilt [rad] | Yaw rate RMS error [rad/s] | Velocity RMS error [m/s] | Max stance slip [mm] |
| --- | --- | --- | --- | --- | --- | --- |
| `turn_1radps`, 0.15 s stance | no, at 6.67 s | 0.497 | 0.300 | 0.511 | 0.150 | 97.4 |
| `turn_1radps` | yes | 0.630 | 0.025 | 0.173 | 0.051 | 5.5 |
| `arc`, 0.15 s stance | yes | 0.635 | 0.033 | 0.068 | 0.043 | 6.4 |
| `arc` | yes | 0.633 | 0.035 | 0.072 | 0.045 | 5.8 |
| `turn_in_place_720`, baseline | yes | 0.636 | 0.010 | 0.083 | 0.036 | 5.9 |
| `turn_in_place_720` | yes | 0.631 | 0.014 | 0.086 | 0.037 | 3.9 |

`turn_1radps` turns through 11.97 of the commanded 12.6 rad, and `turn_in_place_720` reaches a cumulative heading of
11.62 rad against the baseline's 11.97 rad, within the default bands. The turning runs stand 3 to 6 mm lower than on
the 0.15 s stance.
