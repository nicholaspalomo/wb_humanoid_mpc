# tron1_centroidal_mpc

The centroidal MPC configuration of the LimX Tron1 wheel-foot biped: the typed textprotos the MPC node, the robot
process and the validation read, and the Pinocchio playground `test_pinocchio_model`. The robot is described in
[`../README.md`](../README.md); the URDF, the MuJoCo model, their sources and the Apache-2.0 license are in
[`../tron1_description`](../tron1_description/README.md).

| File | Schema | What it sets |
| --- | --- | --- |
| `config/mpc/task.textproto` | `TaskFile` | the contact set, legs, costs, constraints, solver, swing and the robot-process settings |
| `config/command/reference.textproto` | `ReferenceFile` | the default posture, the nominal base height and the command limits |
| `config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | kp 400 N m/rad and kd 10 N m s/rad on the six leg joints; kp 0 and kd 5 N m s/rad on the two wheels |
| `launch/robot.textproto`, `launch/mpc.textproto`, `launch/dummy_sim.textproto` | launch files | the robot process, the MPC node and the dummy simulator, with `--robot_name=tron1` |

## Contact set and gaits

Two rolling contacts in this order: `wheel_L_contact` and `wheel_R_contact` (`model_settings.contacts_rolling`), one
per wheel, each on its wheel's own joint `wheel_L_Joint` or `wheel_R_Joint` at the wheel center, 7.07 mm outboard of
the joint frame along the axle, with a wheel radius of 0.127 m. Each is on a leg of its own, `left` and `right`. The
force of a wheel acts at its lowest point, and in stance the wheel's material there does not slip
([`rolling_contact`](../../../legged_nmpc/docs/rolling_contact/README.md)). The robot balances on its two wheels as
a wheeled inverted pendulum: it pitches by moving the wheels under its center of mass and turns by driving them at
different speeds.

There is no gait file in this package. The shared
`legged_nmpc/legged_common_mpc/config/command/wheeled_gait.textproto` has the two gaits a wheeled robot needs,
stance and drive, both on every wheel, and a `gait_switching` table from stance to drive.

The wheel joints are continuous: their position has no weight in `state_weights` and `final_state_weights` and no
joint-limit row, and the input cost weighs their velocities at a hundredth of the leg joints' (10 against 1000,
scaling 1e-3), so that the no-slip rows decide how fast a wheel turns.

The centroidal MPC is the only formulation configured. The whole-body MPC refuses any contact set other than two 6-DoF
feet (`legged_nmpc/legged_wb_mpc/src/WBMpcInterface.cpp`, `ModelSettings::isBipedContactSet()`), so this robot has
no whole-body package.

## Running it

```bash
make launch-tron1-dummy-sim        # the MPC against the dummy simulator; -vnc for the VNC desktop
make launch-tron1-sim              # the MPC with the robot process in MuJoCo (run on the host)
make launch-tron1-sandbox          # the URDF in Rerun, a slider per joint
make closed-loop-metrics ROBOT=tron1 LABEL=<label>   # the headless closed loop of legged_mpc_validation
make test-pinocchio-model-tron1    # print the Pinocchio model built from task.textproto
```

The launch files read the shared `legged_nmpc/legged_common_mpc/config/command/wheeled_gait.textproto`. The robot
process runs the simulator's ground-truth state estimator: the kalman estimator refuses rolling contacts.

## Validation results

Tron1 survives `standing`, `walk_0p5` and `turn_in_place_720` with the ground-truth contact estimator and state
estimator, drives at 0.506 m/s against the 0.5 m/s command of `walk_0p5` and keeps both wheels in measured contact
throughout. The metrics of label `2026-11-12_tron1` are in
[its README](../../../legged_nmpc/legged_mpc_validation/data/closed_loop/2026-11-12_tron1/README.md); the wheel kd
of 5 N m s/rad comes from those runs ([`../README.md`](../README.md), "Validation").

A change to a file here changes these runs: record a new label and compare it with `make compare-closed-loop
CANDIDATE=<label> BASELINE=2026-11-12_tron1 ROBOT=tron1`.
