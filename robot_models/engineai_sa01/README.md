# EngineAI SA01

A 12-DoF, 33.1 kg bipedal robot: two six-joint legs on a floating base, with no arms, no waist and no head. It is the
smallest and simplest robot in this repository, and the only legs-only one.

This directory holds the two ament packages the stack needs:

| Package | Contents |
| --- | --- |
| `engineai_sa01_description` | `urdf/zq_sa01.urdf`, the MuJoCo model `urdf/zq_sa01.xml`, `meshes/`, `rviz/urdf_config.rviz`, `launch/display.launch.py` |
| `engineai_sa01_centroidal_mpc` | `config/mpc/task.yaml`, `config/mpc/contact_planning.yaml`, `config/command/reference.yaml`, `config/controller/joint_pd_gains.yaml`, the two sim launch files, and `test/testPinocchioModel.cpp` |

Both are registered in `setup_env.sh`, so `source setup_env.sh` makes them visible to `ros2 launch`.

## Running it

```bash
make launch-sa01-sandbox      # RViz URDF viewer with joint sliders
make launch-sa01-dummy-sim    # centroidal MPC against the ideal-tracking dummy simulator
make launch-sa01-sim          # centroidal MPC in MuJoCo
make test-pinocchio-model-sa01  # print the Pinocchio model, joint order and contact frames
```

Each has a `-vnc` variant (`make launch-sa01-sim-vnc`) that starts the VNC server and the Mesa software GL environment
first; see `.devcontainer/README.md`.

## Kinematics

Joints are declared left leg first, and that declaration order is the order of the MPC state vector and of every
`"(i,j)"` block in `task.yaml` and `reference.yaml`.

| Joint | Function | Axis | Origin (from parent) | Range [rad] | Torque [N·m] |
| --- | --- | --- | --- | --- | --- |
| `leg_?1_joint` | hip roll | x | `0 ±0.075 0` from `base_link` | ±0.523 | 140 |
| `leg_?2_joint` | hip yaw | z | coincident with hip roll | ±0.300 | 140 |
| `leg_?3_joint` | hip pitch | y | `0 0 -0.1311` | ±1.204 | 140 |
| `leg_?4_joint` | knee | y | `0 0 -0.30` | 0.0 … 2.268 | 140 |
| `leg_?5_joint` | ankle pitch | y | `0 0 -0.37` | −1.0 … 0.6 | 24 |
| `leg_?6_joint` | ankle roll | x | coincident with ankle pitch | ±0.600 | 24 |

`?` is `l` or `r`; the right leg mirrors the left about y. The floating base link is `base_link` — there is no
`pelvis` and no `torso`, which is why `telemetryFrames` names `base_link` and why `task_space_torso_cost` is absent
from the cost list.

## Derived constants

Everything below was measured with MuJoCo on `zq_sa01.xml` rather than estimated, and the same numbers appear in the
config files with a comment pointing back here. `make test-pinocchio-model-sa01` reprints them from the Pinocchio
model, which is the check that the URDF and the MJCF still agree.

| Quantity | Value | Where it is used |
| --- | --- | --- |
| Total mass | 33.113 kg | — |
| Leg length, hip pitch to sole | 0.725 m | step and swing scaling |
| Base above sole, legs straight | 0.8561 m | RViz grid offset |
| Nominal crouch | hip −0.30, knee 0.70, ankle −0.40 | `reference.yaml` `defaultJointState`, `task.yaml` `initialState` |
| Base above sole at that crouch | 0.8135 m | `defaultBaseHeight`, `initialState` (8,0), MJCF spawn height |
| CoM above sole at that crouch | 0.6124 m | `dcm_terminal_cost.comHeight`, `contact_planning` `shared.comHeight` |
| Sole | 0.27 × 0.10 m, 0.055 m below the ankle | `contacts.contact_frame_translation` |
| Contact frame, from the ankle | `x 0.060, y 0, z −0.055` | `contacts.contact_frame_translation` |
| Hip-to-hip spacing | 0.150 m | step widths, collision radii |
| Natural stance (legs vertical) | 0.160 m sole centre to sole centre | `nominalStepWidth`, `hlip.stepWidth` |

The crouch is a 5.0% reduction from the straight-leg height, matching the 5.2% Atlas stands at. The three sagittal
angles sum to zero so the sole stays flat, but they are **not** the usual symmetric `−θ / 2θ / −θ`: the thigh (0.30 m)
and shank (0.37 m) are unequal, so a symmetric split puts the feet 2 cm ahead of the hips and leaves the robot
standing on its heels. The −0.30 / 0.70 / −0.40 split puts the centre of mass 4 mm behind the centre of the support
polygon instead of 28 mm.

## Contact model

Each sole is four corner spheres of radius 10 mm, the same scheme Atlas and the Unitree G1 use — point contacts
rather than a box, which keeps the contact set small and the normals clean. Their centres are inset one radius from
the sole rectangle and sit 45 mm below the ankle, so the contact plane is at exactly −0.055 m and the support polygon
is `x ∈ [−0.125, 0.125]`, `y ∈ [±0.035]` in the contact frame.

The MPC's `contacts.contact_rectangle` matches that polygon except for `x_max`, which is capped — see below.

## Known issues to watch in the first bring-up

**The ankle is the weakest link on this robot, and the MPC does not know it.** The ankle pitch actuator is 24 N·m
against 324.8 N of body weight, so in single support the ankle can only hold the centre of pressure within 74 mm of
itself; in double support, carrying half the weight each, within 148 mm. The sole is 270 mm long and its toe is
185 mm ahead of the ankle. Nothing in the formulation bounds ankle torque — the wrench cone bounds the centre of
pressure geometrically only — so the solver would happily plan a forward CoP the ankle cannot hold and the robot
would pitch over its toes. `contact_rectangle.x_max` is therefore set to the double-support budget of 0.09 m rather
than the geometric 0.125 m, and `task.yaml`'s external-torque cost weights the ankle joints five times higher than
R1's. Turn the `zmp` viewer marker on (`z`) and watch whether the ZMP runs to the toe. If the 24 N·m in the URDF
turns out to be a motor-side figure ahead of a gear reduction, raise `x_max` to 0.125 and drop the extra weighting.

For the same reason `joint_pd_gains.yaml` carries an explicit `torque_limit` on every joint, which G1 and R1 both
omit. `CentroidalMpcMrtJointController` clamps the commanded joint torque against that key and falls back to a
hard-coded 500 N·m when it is missing — twenty times what SA01's ankle can deliver. The values there are the URDF
effort limits and match the `actuatorfrcrange` of `zq_sa01.xml` joint for joint.

**SA01 has proportionally the longest foot here** — 0.27 m of sole on a 0.725 m leg, where Atlas manages 0.24 m on
0.878 m. A degree of foot droop drops the toe 2.2 mm, so `task_space_foot_cost_weights.orientation_x / _y` are at
1000 and `swing_trajectory_config.swingPitchAngle` is available (at 0.0) if the toe still catches.

**SA01 has the narrowest hips here** — 0.150 m, about two thirds of the other three robots. Lateral base motion runs
out sooner, which is why `Q(11,11)` (base roll) is the largest base-pose weight and why every lateral number in
`contact_planning.yaml` is scaled down rather than copied.

## What is configured, and what is deliberately off

SA01 ships the baseline formulation the Unitree R1 runs and is tuned against: base-pose tracking, the gait schedule,
wrench inputs, a quadratic terminal cost. The weights are R1's, remapped to SA01's joint order — R1 is the closest
robot by mass (28.8 kg against 33.1 kg). They are a starting point, not a tuned set.

Every newer feature is off, and each one is pre-parameterised where it is switched on, so they can be enabled one at
a time and validated in MuJoCo:

| Toggle | State | To enable |
| --- | --- | --- |
| `useContactBasisVectorInputs` | off | one line; `basisNonNegativityBarrier` and `basisScalingRegularization` are already set |
| `useDcmTerminalCost` | off | one line; `dcm_terminal_cost` is parameterised with SA01's real CoM height |
| `useContactPlanning` | off | one line; `contact_planning.yaml` carries SA01's cadence, step widths and reach, and `task_space_foot_cost_weights.pos_x / pos_y` must be raised off 0 at the same time so the planned footholds reach something |
| `useComAndAcomTracking` | **unavailable** | needs a trained ACoM SIREN network; see below |

### ACoM tracking is not available for SA01

`AngularCenterOfMass::createForRobot` dispatches on `model_settings.robotName` and only `atlas` and `g1` have trained
weights, so setting `useComAndAcomTracking: true` throws `Unknown robot 'engineai_sa01'` at start-up. Enabling it
means training SA01 weights with `humanoid_learning/acom` and adding an `AcomSirenWeightsSa01.h` to that dispatch.
The same dependency is why `contact_planning.yaml` omits `heading_double_integrator` from its `dynamics` list: the
planner's heading model is the ACoM. Because ACoM tracking is off, `Q`'s base-pose block (6..11) is live and `Q_com` /
`Q_acom` are absent rather than present and dead.

### Arm swing

Onboarding SA01 needed one change to the shared MPC: `model_settings.armJointNames` used to be mandatory, and
`ModelSettings` aborted at load time if its four entries did not resolve to joints of the MPC model. A legs-only
robot cannot satisfy that. The block is now optional — omitting it entirely leaves `hasArmSwingJoints` false and the
procedural arm swing disabled — while naming only some of the four, or naming a joint that is not in the MPC model,
still fails loudly as before.
