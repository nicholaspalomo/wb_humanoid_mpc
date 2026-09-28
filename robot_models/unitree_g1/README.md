# Unitree G1

The 29-DoF Unitree G1 humanoid: six-joint legs, a three-joint waist and seven-joint arms. The MPC holds the six wrist
joints fixed (`model_settings.fixedJointNames` in `task.yaml`), so it optimizes over the remaining 23.

This directory holds three ament packages:

| Package | Contents |
| --- | --- |
| `g1_description` | the URDF and MuJoCo models in `urdf/` (the MPC uses `g1_29dof.urdf` and `g1_29dof.xml`), meshes, launch and rviz files; its own `README.md` is Unitree's description of the model variants |
| `g1_centroidal_mpc` | the centroidal MPC's `config/mpc/task.yaml`, its command and controller configuration, and launch files |
| `g1_wb_mpc` | the whole-body MPC's configuration and launch files |

## Running it

```bash
make launch-g1-dummy-sim      # centroidal MPC against the ideal-tracking dummy simulator
make launch-g1-sim            # centroidal MPC in MuJoCo
make launch-wb-g1-dummy-sim   # whole-body MPC against the dummy simulator
make launch-wb-g1-sim         # whole-body MPC in MuJoCo
```

Each has a `-vnc` variant (`make launch-g1-sim-vnc`) that starts the VNC server first; see `.devcontainer/README.md`.

## ACoM tracking: the network exists but is NOT VALIDATED

<!-- LINT.IfChange(acom_status) -->
G1 has a trained Angular Center of Mass network, `AcomSirenWeightsG1.h`, registered in `AngularCenterOfMass.cpp` under
`model_settings.robotName` `g1`. It is **NOT VALIDATED, and must stay off**: G1's centroidal `task.yaml` does not list
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
