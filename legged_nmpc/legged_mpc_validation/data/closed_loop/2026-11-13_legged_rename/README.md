# 2026-11-13_legged_rename

The closed-loop runs of five configurations after the packages, the namespace `ocs2::humanoid`, the macros and the proto
packages were renamed from `humanoid` to `legged` (`humanoid_nmpc` to `legged_nmpc` and so on). The rename changes no
code path, so each run is held to the label it was last recorded under. Recorded on commit 54ce1bff with the uncommitted
rename. Each document's provenance gives the worktree state (between 1967 and 1983 changed paths, since every run adds
the documents of the runs before it to the worktree), the machine (Intel Core Ultra 7 265K, 20 logical cores, 30.2 GiB),
GCC 13.3.0, HPIPM `GENERIC` and the container image `unknown` (`VALIDATION_CONTAINER_IMAGE` was not passed). The runs
used 4 solver threads and `--jobs=4 --local_test_jobs=2`, one at a time:

```bash
make closed-loop-metrics ROBOT=anymal_b LABEL=2026-11-13_legged_rename BASELINE=2026-11-10_anymal_b
make closed-loop-metrics ROBOT=unitree_g1 LABEL=2026-11-13_legged_rename BASELINE=2026-11-11_g1_kalman
make closed-loop-metrics ROBOT=unitree_g1_kalman LABEL=2026-11-13_legged_rename BASELINE=2026-11-11_g1_kalman
make closed-loop-metrics ROBOT=tron1 LABEL=2026-11-13_legged_rename BASELINE=2026-11-12_tron1
make closed-loop-metrics ROBOT=anymal_c_wheels LABEL=2026-11-13_legged_rename BASELINE=2026-11-12_anymal_c_wheels
```

| Configuration | Baseline | Runs compared | Violations | Improvements |
| --- | --- | --- | --- | --- |
| `anymal_b` | `2026-11-10_anymal_b` | 3 | 0 | 0 |
| `unitree_g1` | `2026-11-11_g1_kalman` | 3 | 0 | 0 |
| `unitree_g1_kalman` | `2026-11-11_g1_kalman` | 3 | 0 | 0 |
| `tron1` | `2026-11-12_tron1` | 3 | 0 | 0 |
| `anymal_c_wheels` | `2026-11-12_anymal_c_wheels` | 3 | 0 | 0 |

The compared runs are `standing`, `walk_0p5` and `turn_in_place_720`. Beyond the bands, every metric of the 15 compared
documents equals its baseline's exactly, except the solve times, which vary from run to run, and the
`sqp_step.constraint_violation` statistics, which differ from the baseline's in the last bit or two in 8 of them (a
relative difference of at most 3.5e-16). Two earlier recordings with the same configuration files,
`2026-11-08_g1_fall_fix` and `2026-11-11_g1_kalman`'s `unitree_g1`, differ from each other in the same bits, so this is
run-to-run rounding and not the rename: survival, base height, tilt, velocity, yaw rate, heading, slip, joint torque,
the initial-state gap, the SQP step counts and the failure counters are identical.

`unitree_g1` ran its seven scenarios because the command passed no `SCENARIO`. The four that its baseline does not have,
`walk_0p3`, `lateral_0p2`, `turn_1radps` and `arc`, survived and are kept in the label without a comparison.

Two fields differ from the baselines by construction. `schema` is `legged_mpc_validation.closed_loop_metrics.v2`, where
the older documents say `humanoid_mpc_validation.closed_loop_metrics.v2`: the fields are the same, and the comparison
does not read the name. `configuration_sha256` differs for every textproto, because the rename rewrote their
`# proto-file:` and `# proto-message:` headers, and the paths of the shared gait files start with `legged_nmpc/`. Each
configuration file of the runs differs from its commit 54ce1bff version only in the renamed names, and the URDF and MJCF
files hash as before. The older labels keep the paths they were recorded with.

The other configurations, `drc_atlas`, `engineai_sa01`, `unitree_r1` and `unitree_g1_wb`, have no baseline label; their
closed-loop tests pass on the renamed tree.
