# anymal_b_centroidal_mpc

The centroidal MPC configuration of ANYbotics ANYmal B: the typed textprotos the MPC node, the robot process and the
validation read, the launch files of its Make targets, and the Pinocchio playground `test_pinocchio_model`. The robot,
its weights and their derivation are described in [`../README.md`](../README.md); the URDF, the MuJoCo model, their
sources and the BSD-3 license are in [`../anymal_b_description`](../anymal_b_description/README.md).

| File | Schema | What it sets |
| --- | --- | --- |
| `config/mpc/task.textproto` | `TaskFile` | the contact set, legs, costs, constraints, solver, swing and the robot-process settings |
| `config/command/reference.textproto` | `ReferenceFile` | the default posture, the nominal base height and the command limits |
| `config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | kp 400 N m/rad and kd 10 N m s/rad on all 12 joints |
| `launch/robot.textproto`, `launch/mpc.textproto`, `launch/dummy_sim.textproto` | launch files | the robot process, the MPC node and the dummy simulator, with `--robot_name=anymal_b` |

## Contact set and gaits

Four 3-DoF contacts in this order: `LF_FOOT`, `RF_FOOT`, `LH_FOOT` and `RH_FOOT`, the URDF's own `FOOT` frames, each on
its leg's `<leg>_KFE` joint, and one leg of the same name per contact. There is no gait file in this package: the launch
files' `gait_file` and the validation's `RobotConfiguration::gaitFile` name the shared
`legged_nmpc/legged_common_mpc/config/command/quadruped_gait.textproto` (mode bits LF = 8, RF = 4, LH = 2, RH = 1),
whose `gait_switching` table goes from stance to trot and flying_trot.

The centroidal MPC is the only formulation configured. The whole-body MPC refuses any contact set other than two 6-DoF
feet (`legged_nmpc/legged_wb_mpc/src/WBMpcInterface.cpp`, `ModelSettings::isBipedContactSet()`), so this robot has
no whole-body package.

## Running it

```bash
make launch-anymal-b-dummy-sim        # the MPC against the dummy simulator; -vnc for the VNC desktop
make launch-anymal-b-sim              # the MPC with the robot process in MuJoCo (run on the host)
make launch-anymal-b-sandbox          # the URDF in Rerun, a slider per joint
make closed-loop-metrics ROBOT=anymal_b LABEL=<label>     # the headless closed loop of legged_mpc_validation
make test-pinocchio-model-anymal-b    # print the Pinocchio model built from task.textproto
```

## Validation results

ANYmal B survives `standing`, `walk_0p5` and `turn_in_place_720` with the ground-truth contact estimator and state
estimator, and trots at 0.42 m/s on average against the 0.5 m/s command of `walk_0p5`. The metrics of label
`2026-11-10_anymal_b` are in
[its README](../../../legged_nmpc/legged_mpc_validation/data/closed_loop/2026-11-10_anymal_b/README.md).

A change to a file here changes these runs: record a new label and compare it with `make compare-closed-loop
CANDIDATE=<label> BASELINE=2026-11-10_anymal_b ROBOT=anymal_b`.
