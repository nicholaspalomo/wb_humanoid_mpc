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

The MPC configurations can be found in `drc_atlas_centroidal_mpc/config/mpc/task.yaml`.

### Tracking Formulations

You can choose between different tracking formulations in `task.yaml`:

- `com_and_acom_tracking_cost` in the `costs` list: selects between regulating the floating base pose directly and regulating the Center of Mass (CoM) and Angular Center of Mass (ACoM). It replaced the boolean `useComAndAcomTracking`, which a task file may no longer carry (start-up refuses it).
  - not listed: the MPC state quadratic cost explicitly tracks the floating base pose (`p_base`, `theta_base`).
  - listed (Atlas ships this): the MPC regulates the CoM (`p_com`) and ACoM (`theta_acom`) with `ComAndAcomTrackingCost`. The 6x6 base-pose block of the state weights - the diagonal block at rows and columns 6..11 (`p_base` and the base Euler angles) of both the running cost `Q` and the terminal cost `Q_final` - is automatically zeroed to prevent penalizing the same error twice, and the procedural arm swing is switched off. The top-left 6x6 block (the normalized centroidal momentum) is left as it is. With the quadratic terminal cost the terminal node gets its own CoM + ACoM term, weighted by `terminalCostScaling`; Atlas ends its horizon on the DCM cost instead (`dcm_terminal_cost` in `costs`, in place of `terminal_cost`). You can tune the tracking weights using the `Q_com` and `Q_acom` 3x3 diagonal matrices in `task.yaml`. See `humanoid_learning/acom/README.md`, section 4.

- `contactInputParameterization` (a name): how the contact inputs are parameterized. It defaults to `wrench` when `task.yaml` omits it; Atlas's `task.yaml` names `basis_vectors`. It replaced the boolean `useContactBasisVectorInputs`, which a task file may no longer carry (start-up refuses it).
  - `wrench`: The MPC optimizer directly optimizes contact wrenches (3D forces and 3D torques) at each end-effector, and the contact wrench cone is a soft constraint of the formulation lists.
  - `basis_vectors` (Atlas ships this): The MPC optimizer optimizes the scalings `\lambda \ge 0` of wrench-cone generators. Every `\lambda \ge 0` gives a wrench inside the contact wrench cone, which replaces the explicit cone constraint; the bound itself is a barrier built for every contact. With the default generator set (`contacts.basisGeneratorSet: conservative_inner_approximation`, as shipped) only a small part of that cone is reachable; `exact_wrench_cone` spans all of it. See `humanoid_nmpc/docs/contact_basis_vectors/README.md`.

### ACoM network

<!-- LINT.IfChange(acom_status) -->
Atlas's ACoM network (`AcomSirenWeightsAtlas.h`) is the reference, and the only one **validated** for closed-loop use:
its joint Jacobian approximates the centroidal connection to 0.219 mean relative error, against acceptance bounds of
0.30 mean and 0.50 worst that `testAcomAngularVelocityConsistency` enforces. That is why Atlas is the one robot that
ships with `com_and_acom_tracking_cost` in its `costs` and with `heading_double_integrator` in its contact planner. See
`humanoid_learning/acom/README.md`, section 6.3.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:acom_acceptance_robots) -->
