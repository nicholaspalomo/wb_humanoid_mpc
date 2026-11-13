# ANYbotics ANYmal B

ANYmal B is a 30.4 kg quadruped by ANYbotics: four legs of three joints each, hip abduction/adduction (`HAA`, about the
base's x axis), hip flexion/extension (`HFE`, y) and knee flexion/extension (`KFE`, y), on legs named `LF`, `RF`, `LH`
and `RH` (left/right, front/hind). Each leg ends in a rubber ball foot, which the MPC models as a 3-DoF point contact on
the URDF's `<leg>_FOOT` frame. It is the first robot of this repository that is not a biped.

This directory holds two Bazel packages, each with its own README:

| Package | Contents |
| --- | --- |
| [`anymal_b_description`](anymal_b_description/README.md) | the URDF (`urdf/anymal_b.urdf`) and the MuJoCo model (`urdf/anymal_b.xml`, which declares the virtual gantry's `gantry` weld from the world to `base`), the decimated meshes and ANYbotics' BSD-3 `LICENSE`; its `README.md` lists the sources, their commits and every change made to them |
| [`anymal_b_centroidal_mpc`](anymal_b_centroidal_mpc/README.md) | the centroidal MPC configuration: `config/mpc/task.textproto`, `config/command/reference.textproto` and `config/controller/joint_pd_gains.textproto`, and the Pinocchio model playground `test_pinocchio_model` |

## Running it

```bash
make launch-anymal-b-dummy-sim    # centroidal MPC against the dummy simulator (the MPC model's own rollout)
make launch-anymal-b-sim          # centroidal MPC in MuJoCo: the robot process in the robot-sim container (run on the host)
make launch-anymal-b-sandbox      # the URDF in Rerun, a slider per joint
make deploy-robot ROBOT=anymal_b HOST=<robot> NETWORK=<file>     # the robot side on the robot's computer
make test-pinocchio-model-anymal-b    # print the Pinocchio model, its joints and frames
make closed-loop-metrics ROBOT=anymal_b LABEL=<label>             # the headless closed loop, every scenario of anymal_b
```

Each launch target has a `-vnc` variant (`make launch-anymal-b-sim-vnc`) with every view on the VNC desktop; see
`.devcontainer/README.md`. The launch files are `anymal_b_centroidal_mpc/launch/{robot,mpc,dummy_sim}.textproto` and
`anymal_b_description/launch/sandbox.textproto`; the MPC node reads the shared quadruped gait file. The MuJoCo robot
process starts with the ground-truth contact estimator (`cheater_sim`) on the four foot spheres and without foot force
sensors, which the simulator reads only for two 6-DoF feet. The Rerun bridge's "left" and "right" foot force panels
stay empty for this robot: its contact force series are named after the contacts (`contact_forces/LF_FOOT_normal`,
...), which the `contact_forces/*` panel draws.

## The centroidal MPC configuration

The MPC plans on four 3-DoF contacts, `LF_FOOT`, `RF_FOOT`, `LH_FOOT` and `RH_FOOT` in that order, which are the URDF's
own `FOOT` links on the `<leg>_KFE` joints, with one leg of the same name per foot (`LF`, `RF`, `LH`, `RH`). The gaits
are those of the shared quadruped gait file, `legged_nmpc/legged_common_mpc/config/command/quadruped_gait.textproto`
(mode bits LF = 8, RF = 4, LH = 2, RH = 1). The terms are the state and input costs, the hard `zero_wrench`,
`zero_velocity` and `normal_velocity` constraints, the soft joint-limit and friction-cone constraints, and
`task_space_foot_cost`, which lands each swing foot at its nominal stance below the hip (`nominal_foothold` anchor
`nominal_stance`). The locomotion heuristics stay off.

The weights, the friction coefficient, the horizon, the time step, the swing and the default posture start from the
OCS2 `legged_robot` example of ANYmal C (`ocs2_legged_robot/config/mpc/task.info`). The headless closed loop in MuJoCo
(`legged_nmpc/legged_mpc_validation`, label `2026-11-10_anymal_b`) changed four things from that start: the joint
PD gains (kp 80 and kd 2 let the trot sag 5 cm and reach 0.17 m/s of a 0.5 m/s command), the horizontal linear
momentum weight (0.34 m/s at 15, 0.42 m/s at 400), the yaw weight and the foot task-space cost (without it the trot
drifted 0.30 rad off its heading over a 15 s walk):

| Quantity | Value | Derivation |
| --- | --- | --- |
| mass | 30.4214 kg | the URDF, as Pinocchio sums it |
| default posture | HAA -0.25 / +0.25 rad (left / right), HFE +0.6 / -0.6 rad and KFE -0.85 / +0.85 rad (front / hind) | the OCS2 example's; ANYmal B's joint axes are ANYmal C's, so the knees point inward with every foot below its hip |
| base height | 0.5097 m | the URDF's forward kinematics of the default posture with the four `FOOT` frames on the ground |
| state weights | the OCS2 example's Q, with the horizontal linear momentum raised from 15 to 400 and the yaw from 100 to 1000 | the momenta are mass-normalized, so ANYmal B uses the same weights; the two raised weights come from the closed-loop runs below |
| contact force weights | 2.937e-3 per axis | OCS2's 1e-3 times (52.135 kg / 30.421 kg)^2, ANYmal C's mass over ANYmal B's, so that a force in units of the body weight costs the same |
| joint velocity weights | 1.299 (HAA), 1.221 (HFE), 0.566 (KFE) | the diagonal of J^T (5 I) J, OCS2's foot-velocity weight through each leg's Jacobian at the default posture |
| friction, horizon, time step | 0.5, 1.0 s, 0.015 s | the OCS2 example's |
| swing | apex 0.1 m, lift-off 0.2 m/s, touch-down -0.4 m/s | the OCS2 example's |

The joint PD gains are kp 400 N m/rad and kd 10 N m s/rad on every joint, with the URDF's 80 N m torque limit; with the
MuJoCo model's joint damping of 2 N m s/rad they pass `robot_models/tests/test_joint_pd_stability.py`. The robot
process runs the simulator's ground-truth contact estimator (`cheater_sim`), the default state estimator and the
`weld_constraint` gantry, as the bipeds do.

ANYmal B runs on the centroidal MPC only. The whole-body MPC refuses every contact set other than two 6-DoF feet
(`WBMpcInterface`, `ModelSettings::isBipedContactSet()`), and the centroidal MPC refuses the cost and constraint terms
written for two feet by name (`checkCentroidalContactSet()`).

## Validation

The headless lockstep closed loop in MuJoCo (`legged_nmpc/legged_mpc_validation`, configuration `anymal_b`) runs
`standing`, `walk_0p5` and `turn_in_place_720`, and fails when the robot falls in `walk_0p5`. Label
`2026-11-10_anymal_b` recorded the shipped configuration (`make closed-loop-metrics ROBOT=anymal_b
LABEL=2026-11-10_anymal_b RECORD_STATES=1`), and ANYmal B is held to the default bands around it. It survives all
three, trots at 0.42 m/s on average against the 0.5 m/s command and turns through 24.6 of the commanded 25.1 rad; the
metrics and the solve benchmark are in
[the label's README](../../legged_nmpc/legged_mpc_validation/data/closed_loop/2026-11-10_anymal_b/README.md).

## Trot speed

ANYmal B ships with its legs regularized toward the default posture (`leg_joint_reference` `default_posture`: the
state cost's joint weight 20 and the input cost's zero joint velocities) and without a foothold heuristic, and its trot
of `walk_0p5` reaches 0.45 m/s of the 0.5 m/s command over the last 10 s. Raising the horizontal linear momentum weight
trades tilt for speed (0.44 m/s and 0.103 rad at 600, 0.45 m/s and 0.153 rad at 1000).

The robot follows its plan: the planned CoM velocity 0.1 s ahead of each solve is within 0.0013 m/s of the measured
one over the steady trot. The plan itself falls short. At four solves 8.0 to 8.3 s into the command, the converged
plan's mean CoM velocity over the 1 s horizon was 0.460 to 0.463 m/s against a reference of 0.498 m/s. Adding
1/2 400 (v_x - 0.5)^2 to the problem and solving again raised it by 0.012 m/s; the change of each term per m/s of that
rise was:

| Term | Change per m/s |
| --- | --- |
| input cost, joint velocities | +14.2 |
| input cost, contact forces | +5.7 |
| state cost, joint positions | +4.1 |
| state cost, lateral momentum | +1.3 |
| foot task-space costs, all four | +0.1 |
| state cost, base position x | -1.9 |
| state cost, forward momentum | -15.5 |

The joint velocities of a trotting leg move its foot relative to the base at about the walking speed, backwards in
stance and forwards in swing, so a zero joint-velocity reference costs more the faster the robot walks; the default
joint positions do the same for the stride length. The joint-velocity weights are what the derivation above says, the
diagonal of J^T (5000 I) J at the default posture with no mass scaling (1299, 1221 and 566, recomputed from the URDF
with Pinocchio).

`leg_joint_reference: "foot_motion"` (`legged_common_mpc/reference_manager/LegJointReference.h`) regularizes each
leg toward the joints that put its foot where the references do. With it and `translational_stepping` at
`forward_stance_fraction` 0.5 the trot reached 0.499 m/s over the last 10 s, with a peak tilt of 0.074 rad (0.082 on
the default posture), and the turn in place went from 24.6 to 25.9 of the commanded 25.1 rad. It is not shipped: the
stronger joint references trigger the Gauss-Newton SQP's cycling with the hard swing normal-velocity rows
(`legged_nmpc/docs/quaternion_base_orientation/README.md`, section 6, risk 12). In the walk the solves refused 25 full
steps above `g_max` instead of 3, the p99 of the constraint violation after the step rose from 0.023 to 0.30 and that
of the swing normal-velocity residual from 0.16 to 1.0 m/s, and in `testMpcEquivalence_anymal_b` the iterated SQP no
longer converges within 100 iterations and the real-time solves no longer hold the swing rows (a median residual of
1.4 m/s). The reference can be shipped once that defect is fixed.

The trot of `quadruped_gait.textproto` has no flight phase: its two diagonal modes switch at once every 0.35 s.
