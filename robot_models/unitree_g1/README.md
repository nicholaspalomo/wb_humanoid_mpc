# Unitree G1

The 29-DoF Unitree G1 humanoid: six-joint legs, a three-joint waist and seven-joint arms. The MPC holds the six wrist
joints fixed (`model_settings.fixed_joint_names` in `task.textproto`), so it optimizes over the remaining 23.

This directory holds three Bazel packages:

| Package | Contents |
| --- | --- |
| `g1_description` | the URDF and MuJoCo models in `urdf/` (the MPC uses `g1_29dof.urdf` and `g1_29dof.xml`, which declares the virtual gantry's `gantry` weld from the world to `pelvis`) and the meshes; its own `README.md` is Unitree's description of the model variants |
| `g1_centroidal_mpc` | the centroidal MPC's configuration: `config/mpc/task.textproto`, `config/command/reference.textproto` and `config/controller/joint_pd_gains.textproto` |
| `g1_wb_mpc` | the whole-body MPC's configuration, the same three files |

The configuration files are typed textprotos, parsed strictly; `robot_models/README.md` says what each one sets, how to
tune them and how the tuning GUI saves them. Either MPC, centroidal or whole-body, applies an edited task file's
`RELOAD_HOT` fields (and the reference file's command limits) before its next solve; `RELOAD_START_UP` fields take
effect at the next start. The robot process reloads the PD gains and the task file's `contact_estimator` and
`contact_wrench_gate`. The GUI's **Save** writes both the laptop's copy and the robot's (`robot_models/README.md`,
"Tuning").

## Running it

```bash
make launch-g1-dummy-sim      # centroidal MPC against the ideal-tracking dummy simulator
make launch-g1-sim            # centroidal MPC in MuJoCo: the robot process in the robot-sim container (run on the host)
make launch-wb-g1-dummy-sim   # whole-body MPC against the dummy simulator
make launch-wb-g1-sim         # whole-body MPC in MuJoCo
make launch-g1-sandbox        # the URDF in Rerun, a slider per joint
make deploy-robot ROBOT=unitree_g1 HOST=<robot> NETWORK=<file>     # or ROBOT=unitree_g1_wb: the robot's computer
```

Each has a `-vnc` variant (`make launch-g1-sim-vnc`) that starts the VNC server first; see `.devcontainer/README.md`.
The launch files are `g1_centroidal_mpc/launch/` and `g1_wb_mpc/launch/` (`robot.textproto`, `mpc.textproto`,
`dummy_sim.textproto`) and `g1_description/launch/sandbox.textproto`
(`humanoid_nmpc/docs/distributed_runtime/README.md`, "Launching").

## ACoM tracking: the network exists but is NOT VALIDATED

<!-- LINT.IfChange(acom_status) -->
G1 has a trained Angular Center of Mass network, `AcomSirenWeightsG1.h`, registered in `AngularCenterOfMass.cpp` under
`model_settings.robot_name` `g1`. It is **NOT VALIDATED, and must stay off**: G1's centroidal `task.textproto` does not list
`com_and_acom_tracking_cost` in its `costs` (a comment there says why), and G1 has no contact planner configuration that
would run the network as a heading model. The whole-body MPC refuses the cost whatever the robot.

The reason is accuracy. Over the joint-limit box, the network's joint Jacobian is 40 % off the centroidal connection
on average, against 22 % for the validated Atlas network, and at its worst configurations it is further from the
target than no network at all; at G1's nominal stance it is 37 % off. It was also trained on a joint box trimmed by
10 % of each range, which left G1's nominal knee angle of 0.1 rad outside the training set; the dataset generator now
samples the full box. `testAcomAngularVelocityConsistency` grades the header against its own measured bounds, which
only guard against it getting worse, and fails if `com_and_acom_tracking_cost` is listed while G1 is marked unvalidated
there.

Enabling it means retraining (`bazel run //humanoid_learning/acom:train_main -- --robot g1 --install_header`) until the
header meets Atlas's acceptance bounds, then marking it validated in the test. See `humanoid_learning/acom/README.md`,
section 6.3.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:acom_acceptance_robots) -->
