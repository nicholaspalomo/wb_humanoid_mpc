# anymal_c_wheels_centroidal_mpc

The centroidal MPC configuration of the wheeled ANYbotics ANYmal C: the typed textprotos the MPC node, the robot
process and the validation read, and the Pinocchio playground `test_pinocchio_model`. The robot is described in
[`../README.md`](../README.md); the URDF, the MuJoCo model, their sources and the BSD-3 license are in
[`../anymal_c_wheels_description`](../anymal_c_wheels_description/README.md).

| File | Schema | What it sets |
| --- | --- | --- |
| `config/mpc/task.textproto` | `TaskFile` | the contact set, legs, costs, constraints, solver, swing and the robot-process settings |
| `config/command/reference.textproto` | `ReferenceFile` | the default posture, the nominal base height and the command limits |
| `config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | kp 400 N m/rad and kd 10 N m s/rad on the twelve leg joints; kp 0 and kd 4 N m s/rad on the four wheels |
| `launch/robot.textproto`, `launch/mpc.textproto`, `launch/dummy_sim.textproto` | launch files | the robot process, the MPC node and the dummy simulator, with `--robot_name=anymal_c_wheels` |

## Contact set and gaits

Four rolling contacts in the order `LF_wheel_contact`, `RF_wheel_contact`, `LH_wheel_contact` and `RH_wheel_contact`
(`model_settings.contacts_rolling`), one per wheel, each on its wheel's own joint `<leg>_wheel` at the wheel center,
the origin of the wheel link, with a wheel radius of 0.098 m. Each is on a leg of its own, `LF`, `RF`, `LH` and `RH`.
The force of a wheel acts at its lowest point, and in stance the wheel's material there does not slip
([`rolling_contact`](../../../legged_nmpc/docs/rolling_contact/README.md)).

There is no gait file in this package. The wheeled quadrupeds' gait file
`legged_nmpc/legged_common_mpc/config/command/wheeled_quadruped_gait.textproto` has stance, drive (both on every
wheel) and turning_trot, and a `gait_switching` table from stance to drive on a linear command and on to turning_trot
on a yaw-rate command above 0.15 rad/s.

None of the four wheels steers: every axle is the base's y axis at the default posture. Driving forward is rolling.
A turn of the base moves each wheel center sideways at the yaw rate times its distance from the base along x, 0.36 m,
which the lateral no-slip row of the wheel forbids; the legs can give that motion only within their HAA range, and a
turn by sliding the wheels sideways (skid steering) needs a slip model the formulation does not have. The robot turns
in turning_trot instead: diagonal pairs of wheels swing for 0.30 s, each swing followed by 0.15 s on all four wheels,
and `task_space_foot_cost` places each swinging wheel at its nominal place turned with the base
(`nominal_foothold` anchor `nominal_stance`). In stance a wheel may still roll along its heading.

The wheel joints are continuous: their position has no weight in `state_weights` and `final_state_weights` and no
joint-limit row, and the input cost weighs their velocities at 10 against the leg joints' 623 to 1635 (scaling 1e-3),
so that the no-slip rows decide how fast a wheel turns.

The centroidal MPC is the only formulation configured. The whole-body MPC refuses any contact set other than two 6-DoF
feet (`legged_nmpc/legged_wb_mpc/src/WBMpcInterface.cpp`, `ModelSettings::isBipedContactSet()`), so this robot has
no whole-body package.

## Running it

```bash
make launch-anymal-c-wheels-dummy-sim        # the MPC against the dummy simulator; -vnc for the VNC desktop
make launch-anymal-c-wheels-sim              # the MPC with the robot process in MuJoCo (run on the host)
make launch-anymal-c-wheels-sandbox          # the URDF in Rerun, a slider per joint
make test-pinocchio-model-anymal-c-wheels    # print the Pinocchio model built from task.textproto
```

The launch files read `legged_nmpc/legged_common_mpc/config/command/wheeled_quadruped_gait.textproto`. The robot
process runs the simulator's ground-truth state estimator: the kalman estimator refuses rolling contacts.

The weights are ANYmal B's and the OCS2 example's, as the task file says. The closed-loop runs of label
`2026-11-12_anymal_c_wheels` (`legged_nmpc/legged_mpc_validation`) set the gait file, `task_space_foot_cost` in the
cost list and the wheel kd of 4 N m s/rad (1 N m s/rad, the description's drop test's, drove at 0.488 m/s against the
0.5 m/s command); [the label's README](../../../legged_nmpc/legged_mpc_validation/data/closed_loop/2026-11-12_anymal_c_wheels/README.md)
has the runs that set them.
