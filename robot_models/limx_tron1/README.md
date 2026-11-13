# LimX Dynamics Tron1

Tron1 is a 22.3 kg biped by LimX Dynamics. Its wheel-foot model `WF_TRON1A` has two legs of four joints each: hip
abduction/adduction (`abad`, about the base's x axis), hip flexion (`hip`, y), knee flexion (`knee`, y) and a wheel
(`wheel`, y) at the end of the shank, on legs named after the URDF's `L` and `R` (left and right). Each wheel is a
0.127 m hub-motor wheel on a continuous joint, so the robot stands and drives on its two wheels as a wheeled inverted
pendulum: it balances in pitch by moving the wheels under its center of mass and turns by driving them at different
speeds.

This directory holds two Bazel packages:

| Package | Contents |
| --- | --- |
| [`tron1_description`](tron1_description/README.md) | the URDF (`urdf/tron1.urdf`) and the MuJoCo model (`urdf/tron1.xml`, which declares the virtual gantry's `gantry` weld from the world to `base_Link`), the decimated meshes and LimX Dynamics' Apache-2.0 `LICENSE`; its `README.md` lists the sources, their commit, every change made to them and the wheel geometry the MPC needs |
| [`tron1_centroidal_mpc`](tron1_centroidal_mpc/README.md) | the centroidal MPC configuration: `config/mpc/task.textproto`, `config/command/reference.textproto` and `config/controller/joint_pd_gains.textproto`, the launch files `launch/{robot,mpc,dummy_sim}.textproto`, and the Pinocchio playground `test_pinocchio_model` |

The centroidal MPC plans on one rolling contact per wheel (`model_settings.contacts_rolling`,
[`rolling_contact`](../../legged_nmpc/docs/rolling_contact/README.md)) with the shared wheeled gait file
`legged_nmpc/legged_common_mpc/config/command/wheeled_gait.textproto`. The state, input and contact weights start
from ANYmal B's: the momentum, base-pose and leg joint weights as they are, the force weights scaled by the square of
the mass ratio to the OCS2 example's ANYmal C, and no weight on a wheel's position.

## Running it

```bash
make launch-tron1-dummy-sim      # centroidal MPC against the dummy simulator (the MPC model's own rollout)
make launch-tron1-sim            # centroidal MPC in MuJoCo: the robot process in the robot-sim container (run on the host)
make launch-tron1-sandbox        # the URDF in Rerun, a slider per joint
make deploy-robot ROBOT=tron1 HOST=<robot> NETWORK=<file>     # the robot side on the robot's computer
make test-pinocchio-model-tron1  # print the Pinocchio model, its joints and frames
make closed-loop-metrics ROBOT=tron1 LABEL=<label>   # the headless closed loop, every scenario of tron1
```

Each launch target has a `-vnc` variant (`make launch-tron1-sim-vnc`) with every view on the VNC desktop; see
`.devcontainer/README.md`. The launch files are `tron1_centroidal_mpc/launch/{robot,mpc,dummy_sim}.textproto` and
`tron1_description/launch/sandbox.textproto`; the MPC node reads the shared wheeled gait file.

Tron1 runs on the centroidal MPC only. The whole-body MPC refuses every contact set other than two 6-DoF feet, and the
kalman state estimator refuses rolling contacts (their contact point moves over the wheel), so the robot process runs
the ground-truth estimator of the simulator. The locomotion heuristics are not derived for it
(`tools/locomotion_heuristics/derive_parameters.py`, `UNDERIVED_ROBOTS`): their formulas assume two feet on footholds
fixed in the world.

## Validation

The headless lockstep closed loop in MuJoCo (`legged_nmpc/legged_mpc_validation`, configuration `tron1`) runs
`standing`, `walk_0p5` (a 0.5 m/s drive on the wheels, on the `drive` gait) and `turn_in_place_720`, and fails when the
robot falls in `walk_0p5`. Label `2026-11-12_tron1` recorded the shipped configuration (`make closed-loop-metrics
ROBOT=tron1 LABEL=2026-11-12_tron1 RECORD_STATES=1`), and Tron1 is held to the default bands around it. The centroidal
MPC balances it on its two rolling contacts in all three: it stands still, drives at 0.506 m/s against the 0.5 m/s
command, turns at about 0.45 rad/s against 0.5 rad/s, and keeps both wheels in measured contact throughout. The metrics,
the solve benchmark and why the turn falls short (the flat wheel cylinders of the MuJoCo scene scrub) are in
[the label's README](../../legged_nmpc/legged_mpc_validation/data/closed_loop/2026-11-12_tron1/README.md).
`legged_nmpc/legged_centroidal_mpc:testTron1LockstepLoop` runs a shorter stand, drive and turn in the same loop.

The closed-loop runs set one value of the configuration: the wheel kd of `config/controller/joint_pd_gains.textproto`,
raised from 1 to 5 N m s/rad. At 1 the drive ran at 0.524 m/s against the 0.5 m/s command, with a tilt RMS of 0.0104
rad against 0.0048 rad at 5. The task, reference and gait files are as the configuration was first written; the first
closed-loop runs on them already survived all three scenarios.
