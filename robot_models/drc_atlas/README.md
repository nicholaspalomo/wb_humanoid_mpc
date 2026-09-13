# DRC Atlas Model

This directory contains the configurations and descriptions for the DRC Atlas robot.

## MPC Configurations

The MPC configurations can be found in `drc_atlas_centroidal_mpc/config/mpc/task.yaml`.

### Tracking Formulations

You can toggle between different tracking formulations using the following parameters in `task.yaml`:

- `useComAndAcomTracking` (bool): Toggles between regulating the floating base pose directly vs. regulating the Center of Mass (CoM) and Angular Center of Mass (ACoM).
  - `false` (default): The MPC state quadratic cost explicitly tracks the floating base pose (`p_base`, `theta_base`).
  - `true`: The MPC state cost explicitly regulates the CoM (`p_com`) and ACoM (`theta_acom`). The top-left 6x6 block of the state quadratic tracking cost `Q` (corresponding to the floating base) will automatically be zeroed out to prevent double-penalizing the state. You can tune the tracking weights using the `Q_com` and `Q_acom` 3x3 diagonal matrices in `task.yaml`.

- `useContactBasisVectorInputs` (bool): Toggles between the standard wrench inputs and the contact basis vector inputs.
  - `false` (default): The MPC optimizer directly optimizes contact wrenches (3D forces and 3D torques) at each end-effector.
  - `true`: The MPC optimizer optimizes the basis vector scalings `\lambda \ge 0`. This formulation embeds the wrench cone constraints implicitly inside the dynamics.
