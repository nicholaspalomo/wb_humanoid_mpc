# DRC Atlas Model

This directory contains the configurations and descriptions for the DRC Atlas robot.

## Running it

```bash
make launch-drc-atlas-sim          # centroidal MPC in MuJoCo: the robot process in the robot-sim container (on the host)
make launch-drc-atlas-dummy-sim    # centroidal MPC against the dummy simulator
make launch-drc-atlas-sandbox      # the URDF in Rerun, a slider per joint
make deploy-robot ROBOT=drc_atlas HOST=<robot> NETWORK=<file>   # the robot side on the robot's computer
make test-pinocchio-model-atlas    # print the Pinocchio model
```

Each has a `-vnc` variant; see `.devcontainer/README.md`. The launch files are `drc_atlas_centroidal_mpc/launch/`
(`robot.textproto`, `mpc.textproto`, `dummy_sim.textproto`) and `drc_atlas_description/launch/sandbox.textproto`
(`humanoid_nmpc/docs/distributed_runtime/README.md`, "Launching").

## MPC Configurations

The MPC's configuration is in `drc_atlas_centroidal_mpc/config/`: `mpc/task.textproto`, `mpc/contact_planning.textproto`,
`command/reference.textproto` and `controller/joint_pd_gains.textproto`. Each is a typed textproto, parsed strictly;
`robot_models/README.md` says what each file sets, how to tune them and how the tuning GUI saves them.

### Tracking Formulations

You can choose between different tracking formulations in `task.textproto`:

- `com_and_acom_tracking_cost` in the `costs` list: selects between regulating the floating base pose directly and regulating the Center of Mass (CoM) and Angular Center of Mass (ACoM). It replaced the boolean `use_com_and_acom_tracking`, which a task file may no longer carry (the parser refuses it and says what to write instead).
  - not listed: the MPC state quadratic cost explicitly tracks the floating base pose (`p_base`, `theta_base`).
  - listed (Atlas ships this): the MPC regulates the CoM (`p_com`) and ACoM (`theta_acom`) with `ComAndAcomTrackingCost`. The base-pose weights - `base_position` and `base_orientation` (`p_base` and the base Euler angles) of both the running cost's `state_weights` and the terminal cost's `final_state_weights` - are automatically zeroed to prevent penalizing the same error twice, and the procedural arm swing is switched off. The normalized centroidal momentum weights (`normalized_linear_momentum`, `normalized_angular_momentum`) are left as they are. With the quadratic terminal cost the terminal node gets its own CoM + ACoM term, weighted by `terminal_cost_scaling`; Atlas ends its horizon on the DCM cost instead (`dcm_terminal_cost` in `costs`, in place of `terminal_cost`). You can tune the tracking weights in the `com_weights { scaling x y z }` and `acom_weights { scaling yaw pitch roll }` blocks of `task.textproto`. See `humanoid_learning/acom/README.md`, section 4.

- `contact_input_parameterization` (a name): how the contact inputs are parameterized. It defaults to `wrench` when the task file omits it; Atlas's `task.textproto` names `basis_vectors`. It replaced the boolean `use_contact_basis_vector_inputs`, which a task file may no longer carry (the parser refuses it and says what to write instead).
  - `wrench`: The MPC optimizer directly optimizes contact wrenches (3D forces and 3D torques) at each end-effector, and the contact wrench cone is a soft constraint of the formulation lists.
  - `basis_vectors` (Atlas ships this): The MPC optimizer optimizes the scalings `\lambda \ge 0` of wrench-cone generators. Every `\lambda \ge 0` gives a wrench inside the contact wrench cone, which replaces the explicit cone constraint; the bound itself is a barrier built for every contact. With the default generator set (`contacts { basis_generator_set: "conservative_inner_approximation" }`, as shipped) only a small part of that cone is reachable; `exact_wrench_cone` spans all of it. See `humanoid_nmpc/docs/contact_basis_vectors/README.md`.

### ACoM network

<!-- LINT.IfChange(acom_status) -->
Atlas's ACoM network (`AcomSirenWeightsAtlas.h`) is the reference, and the only one **validated** for closed-loop use:
its joint Jacobian approximates the centroidal connection to 0.219 mean relative error, against acceptance bounds of
0.30 mean and 0.50 worst that `testAcomAngularVelocityConsistency` enforces. That is why Atlas is the one robot that
ships with `com_and_acom_tracking_cost` in its `costs` and with `heading_double_integrator` in its contact planner. See
`humanoid_learning/acom/README.md`, section 6.3.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:acom_acceptance_robots) -->
