# ANYbotics ANYmal C on wheels

The wheeled ANYmal C is a quadruped of ANYbotics' ANYmal C legs with a driven wheel at the end of each leg, as
described by the community package [anymal_with_wheels](https://github.com/LuanDev3/anymal_with_wheels). Each leg has
three joints, hip abduction/adduction (`HAA`), hip flexion/extension (`HFE`) and knee flexion/extension (`KFE`), and a
continuous wheel joint (`wheel`), on legs named `LF`, `RF`, `LH` and `RH` (left/right, front/hind). Each wheel is a
0.098 m cylinder on the leg's foot adapter, so the robot stands and drives on its four wheels.

This directory holds two Bazel packages:

| Package | Contents |
| --- | --- |
| [`anymal_c_wheels_description`](anymal_c_wheels_description/README.md) | the URDF (`urdf/anymal_c_wheels.urdf`) and the MuJoCo model (`urdf/anymal_c_wheels.xml`, which declares the virtual gantry's `gantry` weld from the world to `base`), the decimated meshes and ANYbotics' BSD-3 `LICENSE`; its `README.md` lists the sources, their commits, every change made to them and the wheel geometry the MPC needs |
| [`anymal_c_wheels_centroidal_mpc`](anymal_c_wheels_centroidal_mpc/README.md) | the centroidal MPC configuration: `config/mpc/task.textproto`, `config/command/reference.textproto` and `config/controller/joint_pd_gains.textproto`, the launch files `launch/{robot,mpc,dummy_sim}.textproto`, and the Pinocchio playground `test_pinocchio_model` |

The centroidal MPC plans on one rolling contact per wheel (`model_settings.contacts_rolling`,
[`rolling_contact`](../../legged_nmpc/docs/rolling_contact/README.md)) with the wheeled quadrupeds' gait file
`legged_nmpc/legged_common_mpc/config/command/wheeled_quadruped_gait.textproto`: it drives on its four wheels and
turns in a trot. The state and input weights start from
ANYmal B's and the OCS2 example's ANYmal C: the momentum and base-pose weights as they are, the force weights scaled by
the square of the mass ratio to the OCS2 example's ANYmal C, the leg joint velocity weights from the wheel Jacobians,
and no weight on a wheel's position.

## Running it

```bash
make launch-anymal-c-wheels-dummy-sim      # centroidal MPC against the dummy simulator (the MPC model's own rollout)
make launch-anymal-c-wheels-sim            # centroidal MPC in MuJoCo: the robot process in the robot-sim container (run on the host)
make launch-anymal-c-wheels-sandbox        # the URDF in Rerun, a slider per joint (-vnc: on the VNC desktop)
make deploy-robot ROBOT=anymal_c_wheels HOST=<robot> NETWORK=<file>     # the robot side on the robot's computer
make test-pinocchio-model-anymal-c-wheels  # print the Pinocchio model, its joints and frames
```

Each launch target has a `-vnc` variant (`make launch-anymal-c-wheels-sim-vnc`) with every view on the VNC desktop;
see `.devcontainer/README.md`. The launch files are `anymal_c_wheels_centroidal_mpc/launch/{robot,mpc,dummy_sim}.textproto`
and `anymal_c_wheels_description/launch/sandbox.textproto`; the MPC node reads the wheeled quadrupeds' gait file.

The wheeled ANYmal C runs on the centroidal MPC only. The whole-body MPC refuses every contact set other than two 6-DoF
feet, and the kalman state estimator refuses rolling contacts (their contact point moves over the wheel), so the robot
process runs the ground-truth estimator of the simulator. The locomotion heuristics are not derived for it
(`tools/locomotion_heuristics/derive_parameters.py`, `UNDERIVED_ROBOTS`): their formulas assume two feet on footholds
fixed in the world.

## Validation

The headless lockstep closed loop in MuJoCo (`legged_nmpc/legged_mpc_validation`, configuration
`anymal_c_wheels`) runs `standing`, `walk_0p5` (a 0.5 m/s drive on the wheels, on the `drive` gait) and
`turn_in_place_720` (on the `turning_trot`), and fails when the robot falls in `walk_0p5`. Label
`2026-11-12_anymal_c_wheels` recorded the shipped configuration (`make closed-loop-metrics ROBOT=anymal_c_wheels
LABEL=2026-11-12_anymal_c_wheels RECORD_STATES=1`), and the robot is held to the default bands around it. It stands
still, drives at 0.496 m/s against the 0.5 m/s command with every wheel in measured contact throughout, and turns at
0.484 rad/s against 0.5 rad/s through 11.78 of the commanded 12.57 rad. The metrics, the solve benchmark and the runs
that did not turn (on the drive gait, with a soft no-slip constraint and on a trot without four-wheel phases) are in
[the label's README](../../legged_nmpc/legged_mpc_validation/data/closed_loop/2026-11-12_anymal_c_wheels/README.md).
`legged_nmpc/legged_centroidal_mpc:testAnymalCWheelsLockstepLoop` runs a shorter drive and turn in the same loop.

The closed-loop runs set three values of the configuration: the gait file (`wheeled_quadruped_gait.textproto` instead
of the wheeled bipeds' `wheeled_gait.textproto`, whose drive gait cannot yaw a base on four parallel axles),
`task_space_foot_cost` in the task file's cost list, which places a swinging wheel, and the wheel kd of
`config/controller/joint_pd_gains.textproto`, raised from 1 to 4 N m s/rad (the drive went from 0.488 to 0.496 m/s).
