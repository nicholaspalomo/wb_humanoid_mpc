# Rolling Contacts

A wheel touches the ground at its lowest point, and that point is a different point of the rim at every instant. The
centroidal MPC models a wheel's contact as a rolling contact (`ModelSettings::ContactKind::kRolling`): a force-only
3-DoF contact whose force acts at the wheel's lowest point and whose stance constraint stops the wheel material at that
point from slipping, while the wheel itself rolls. A task file selects it by listing the wheel in
`model_settings.contacts_rolling`; the contacts of `contacts_3dof` stay points fixed on their links and those of
`contact_names_6dof` soles, and nothing changes for them. The LimX Tron1 (`robot_models/limx_tron1`) stands and drives
on two rolling contacts, and the wheeled ANYmal C (`robot_models/anybotics_anymal_c_wheels`) on four.

This document derives the geometry, the stance constraint and its Jacobian, the swing, the change in the centroidal
dynamics and the costs of the wheel joints, lists what the formulation does not cover, and describes the event-driven
steps that move a wheeled robot where its wheels cannot roll.

## 1. Geometry

The MPC model has one frame per rolling contact, named after the contact, at the wheel center. It is attached to the
child link of the wheel's own joint (`parent_joint_name`) at `pos_wheel_rt_link_ewrt_link`, so the frame turns with the
wheel. Only two things are read off it: the position of its origin $\mathbf{p}_w$ and its world twist
$(\mathbf{v}_w, \boldsymbol{\omega}_w)$, whose angular velocity includes the wheel's spin about its axle. The frame's
orientation is never used, because the contact point does not depend on how far the wheel has turned.

The ground is flat with the world z axis as its normal, $\mathbf{n} = \mathbf{e}_z$ (`normalTerrainEwrtWorld()`). For a
wheel of radius $r$ whose axle is horizontal the lowest point is

$$\mathbf{p}_c = \mathbf{p}_w - r\,\mathbf{n},$$

`posContactRtWorldEwrtWorld()` in `legged_common_mpc/contact/RollingContact.h`, the header of the helpers every
consumer converts with. The swing planner, the footholds, the references, the telemetry and the metrics are written for
$\mathbf{p}_c$; the terms evaluated on the wheel-center frame add the offset $-r\,\mathbf{n}$ themselves.

## 2. No slip

The velocity of the wheel material at the contact point is the velocity of the wheel center plus the rotation of the
wheel about it:

$$\mathbf{v}_c = \mathbf{v}_w + \boldsymbol{\omega}_w \times (-r\,\mathbf{n}) = \mathbf{v}_w + r\,[\mathbf{n}]_\times \boldsymbol{\omega}_w,$$

with $[\mathbf{n}]_\times$ the skew-symmetric matrix of $\mathbf{n}$ (`linvelContactRtWorldEwrtWorld()`). This is not
the velocity of the point $\mathbf{p}_c$, which travels over the ground with the wheel center; it is the velocity of
whatever material of the rim is at $\mathbf{p}_c$. A wheel that rolls without slipping has $\mathbf{v}_c = 0$, so

$$\mathbf{v}_w = r\,\boldsymbol{\omega}_w \times \mathbf{n}.$$

The z component of $\boldsymbol{\omega}_w \times \mathbf{n}$ is zero, so the wheel center does not move vertically. Its
horizontal velocity follows from the horizontal components of $\boldsymbol{\omega}_w$. With $\mathbf{a}$ the unit axle
and $\mathbf{t} = \mathbf{a} \times \mathbf{n} / \lVert \mathbf{a} \times \mathbf{n} \rVert$ the rolling direction, the
spin about the axle drives the wheel center along $\mathbf{t}$ at $r$ times the spin rate, which is the wheel joint's
velocity plus the rotation of the leg about the same axis. A sideways velocity along $\mathbf{a}$ would need a rotation
of the wheel about $\mathbf{t}$, a change of camber, so a wheel with a steady camber cannot slide sideways. The
rotation about $\mathbf{n}$ does not appear at all: the wheel may turn on the spot, which is how a leg steers it and
how a two-wheeled robot turns, by driving its wheels at different speeds.

## 3. The stance constraint

`RollingContactConstraintCppAd` (`legged_centroidal_mpc/constraint/`) is the state-input equality constraint of a
rolling contact in stance, active while the mode schedule has the contact in contact. It has three rows, in world axes:

$$\mathbf{g} = A_x \left(\mathbf{p}_c - h_{\mathrm{ref}}\,\mathbf{e}_z\right) + A_v\,\mathbf{v}_c.$$

$A_x$ and $A_v$ are the translation blocks of the stance-foot gains, the ones a point foot's `zero_velocity` rows use
(`footTwistConfig()` of `model_settings.foot_constraint`):

| Matrix | Entries |
| --- | --- |
| $A_x$ | $\mathrm{diag}(0, 0, k_z)$ with $k_z$ = `position_error_gain_z` |
| $A_v$ | $\mathrm{diag}(k_{v,xy}, k_{v,xy}, k_{v,z})$ with `linear_velocity_error_gain_xy` and `linear_velocity_error_gain_z` |

$h_{\mathrm{ref}}$ is the reference height of the contact point: the swing trajectory planner's, which in stance is the
ground the planner was built on (`LeggedPreComputation::getDesPosFootRtWorldEwrtWorldZ()`), or the height of the
contact's world-aligned patch for a contact the swing planner does not plan. The x and y rows are the no-slip condition
scaled by $k_{v,xy}$. The z row is $k_{v,z}\,v_{w,z} + k_z\,(p_{w,z} - r - h_{\mathrm{ref}})$, a first-order servo of the
contact point onto the ground, the same row a point foot has.

The task file keeps listing `zero_velocity` in `hard_constraints` (or `soft_constraints`): on a rolling contact that
entry builds this constraint instead of `ZeroVelocityConstraintCppAd`, registered as
`<contact>_rollingNoSlip` (`contact_term::kRollingNoSlipName`). The soft form takes
`foot_constraint.soft_constraint_weight`, and a hot reload of the gains reaches it through `ZeroVelocityGainsApplier`
and `FootConstraintSoftWeightsApplier` as it reaches a foot's.

## 4. Jacobian

$r$ and $\mathbf{n}$ are constants, so with the twist $\mathbf{T}_w = (\mathbf{v}_w, \boldsymbol{\omega}_w)$

$$\mathbf{v}_c = \begin{bmatrix} I_3 & r\,[\mathbf{n}]_\times \end{bmatrix} \mathbf{T}_w, \qquad
\frac{\partial \mathbf{v}_c}{\partial(\mathbf{x}, \mathbf{u})} = J_v + r\,[\mathbf{n}]_\times J_\omega,$$

with $J_v$ and $J_\omega$ the rows of the twist's Jacobian with respect to the state and the input, which the CppAD
kinematics of the frame return exactly (`getLinearApproximationTwistEesRtWorldEwrtWorld()`). The position rows
need $\partial \mathbf{p}_c / \partial \mathbf{x} = J_p$, the Jacobian of the frame's position
(`getLinearApproximationPosEesRtWorldEwrtWorld()`). Together

$$\frac{\partial \mathbf{g}}{\partial \mathbf{x}} = A_x J_p + A_v \left(J_{v,x} + r\,[\mathbf{n}]_\times J_{\omega,x}\right),
\qquad
\frac{\partial \mathbf{g}}{\partial \mathbf{u}} = A_v \left(J_{v,u} + r\,[\mathbf{n}]_\times J_{\omega,u}\right).$$

The linearization is exact: unlike the orientation rows of a 6-DoF foot there is no map held at its value at the
linearization point. `testRollingContactConstraint` checks it against central finite differences of the value at
random states and inputs of Tron1, and the value against the material-point velocity Pinocchio computes on its own.

## 5. Swing

A rolling contact can leave the ground. In swing it is a point contact for the swing terms: `normal_velocity` servoes
the wheel center's height onto the swing reference of the contact point plus $r$,

$$v_{w,z} - \dot{z}_{\mathrm{ref}} + k_z\left(p_{w,z} - (z_{\mathrm{ref}} + r)\right) = 0,$$

which `LeggedPreComputation` builds by adding the radius to the reference height of rolling contacts only, and
`task_space_foot_cost` compares the contact point $\mathbf{p}_w - r\,\mathbf{n}$ with its reference (its CppAD library
is named with the suffix `_rollingContact`, so that no library of a fixed point's residual is loaded in its place). The
wheel center and the point below it share their velocity, so the velocity residuals are the wheel center's. The shipped
wheeled gait files (`legged_common_mpc/config/command/wheeled_gait.textproto` for Tron1,
`wheeled_quadruped_gait.textproto` for the wheeled ANYmal C) keep every wheel on the ground while the robot drives, and
swing a wheel where the command asks for motion the wheels cannot roll (section 10).
`testRollingContactSwing` covers a wheel in swing, and `testAnymalCWheelsLockstepLoop` turns the wheeled ANYmal C on it
in MuJoCo.

A fixed contact is a landmark while it stands: a swing starts where it was measured, and the nominal step of the other
foot is measured from it. A rolling contact is not, because it rolls with the base. The reference manager
(`SwitchedModelReferenceManager`) places a rolling contact in stance at time $t$ at

$$\mathbf{p}_c^{\mathrm{ref}}(t) = \mathbf{p}_{\mathrm{base}}(t) + R_z(\psi(t))\,\mathbf{d},$$

with $\mathbf{p}_{\mathrm{base}}(t)$ and $\psi(t)$ the base position and yaw predicted from the last measurement by the
commanded twist, and $\mathbf{d}$ the contact point relative to the base at the nominal posture, in the heading frame
(`configureNominalStance()`). Its velocity is $\mathbf{v}_{\mathrm{cmd}} + \omega_{\mathrm{cmd}}\,\mathbf{e}_z \times
(\mathbf{p}_c^{\mathrm{ref}} - \mathbf{p}_{\mathrm{base}})$. A swing that lifts off after the last solve starts from this
point at the lift-off; a lift-off before the last solve starts from the position measured then, as for a fixed contact.
`getSwingFootReference()` returns the point in stance too, so that `task_space_foot_cost` under `active_phases:
"swing_and_stance"` holds each wheel under its nominal place while the robot drives. The landing target of a swing is the
`nominal_stance` foothold, which is the same point at the touch-down. Fixed contacts keep their references unchanged
(`testRollingContactReferences`).

## 6. Centroidal dynamics

The centroidal dynamics sum the moments of the contact forces about the center of mass,

$$\dot{\mathbf{L}} = \sum_i (\mathbf{p}_{c,i} - \mathbf{p}_{\mathrm{com}}) \times \mathbf{f}_i,$$

with each force at its contact point. For a fixed point that is the contact frame's origin. For a rolling contact the
frame is the wheel center, a radius above the point the force acts at: taking the moment at the wheel center would
add $r\,\mathbf{e}_z \times \mathbf{f}$, which for a traction force of 20 N on Tron1's 0.127 m wheels is a pitch moment
of 2.5 N m. `CentroidalModelInfoTpl::rollingContactWheelRadii` (lib/ocs2) lists the radius of each 3-DoF contact, 0 for
a fixed point, and `getPosContactRtComEwrtWorld()` subtracts it along world z. `CentroidalMpcInterface` fills the vector
only when the contact set has a rolling contact, so every other robot keeps the empty vector and the CppAD tapes it
always had. `testRollingContactSet` compares the angular rows of the flow map with this sum computed by hand.

## 7. Wheel joints in the costs

A wheel's angle has no reference: the wheel rolls as far as the robot drives. The state cost therefore gives the
position of a rolling contact's wheel joint no weight; `state_weights.joint_positions` and
`final_state_weights.joint_positions` must give it 0, and start-up and hot reload reject any other value, reporting the
joint name (`stateWeightsFromConfig()`). The wheel joint is a URDF `continuous` joint, which the Pinocchio build turns into
a revolute joint with infinite limits (`robot_model/UrdfContinuousJoints.h`), so the joint-limit constraint has no row
for it. Start-up refuses a rolling contact whose wheel joint the URDF declares revolute with position limits, as the
upstream Tron1 description does, reporting the joint name and its limits (`checkRollingContactJoints()`).

The input cost weighs each wheel joint's velocity against its reference. In stance the reference is the rolling speed
of the commanded base twist at the contact point,

$$\dot{q}_{\mathrm{ref}} = \frac{\left(\mathbf{v}_{\mathrm{cmd}} + \boldsymbol{\omega}_{\mathrm{cmd}} \times \mathbf{p}_{c/\mathrm{base}}\right) \cdot \mathbf{t}}{r},$$

computed for rolling contacts in `SwitchedModelReferenceManager::getDesiredInput()`, with
$\mathbf{t} = \mathbf{a} \times \mathbf{n} / \lVert \mathbf{a} \times \mathbf{n} \rVert$ for the wheel joint axis
$\mathbf{a}$, the direction the wheel center moves for a positive joint velocity. $\mathbf{p}_{c/\mathrm{base}}$ and
$\mathbf{t}$ are those of the nominal posture turned to the target's heading at the node, which is the target
configuration on a level target with the nominal joints, so no forward kinematics runs per node. A forward command $v$
gives each wheel $v / r$ with the sign of its axis, a turn $\omega$ gives the two wheels of a biped opposite references
$\mp \omega\,y_c / r$ in proportion to their half tracks $y_c$, and a wheel in swing gets 0. Weights that are small next
to the leg joints' let the no-slip rows, not the cost, decide how fast a wheel turns.

## 8. Limits

- The ground is flat and horizontal. On a slope the lowest point is not $-r\,\mathbf{e}_z$ from the wheel center, and
  the no-slip rows would be written in the wrong plane.
- The contact point is exact for a horizontal axle. A wheel cambered by an angle $\gamma$ touches the ground at a point
  $r(1 - \cos\gamma)$ lower and $r \sin\gamma$ to the side of $\mathbf{p}_w - r\,\mathbf{e}_z$; the model ignores both.
  The legs of Tron1 and of the wheeled ANYmal C keep their axles close to horizontal.
- The wheel is a thin disc. A wide tire touches the ground along a line, which can take a moment about the rolling
  direction; the model gives the contact a force and no moment.
- Rolling resistance and the bearing friction are not modeled.
- A wheel cannot slip, so the base can yaw on its wheels only about a point on every axle's line. A wheel at
  $(x_c, y_c)$ from the base center moves at $\omega(-y_c, x_c)$ when the base yaws at $\omega$, and its contact point
  may move only along its heading. On Tron1 both axles lie on one line through the center ($x_c \approx 0$), so a
  differential wheel speed turns it in place. On the wheeled ANYmal C the four axles are parallel and $x_c = \pm 0.36$
  m, so a yaw would move every wheel sideways at $0.36\,\omega$: the robot drives straight on its wheels and turns
  by stepping (section 10), in which a stance wheel pivots about its contact point and rolls along its heading while
  the stance legs turn the base, and each swing re-places a wheel. Neither robot rolls sideways, since no wheel
  steers. Skid steering, which turns a four-wheel base by sliding its wheels sideways, needs a slip
  model this formulation does not have: with `zero_velocity` as a soft constraint the solver plans the sideways slip,
  but the plan's wheel speeds are the rolling speeds, and in MuJoCo the base turned through 0.56 rad of a 12.6 rad
  command (`legged_mpc_validation`, label `2026-11-12_anymal_c_wheels`).

## 9. Refused terms

`checkCentroidalContactSet()` refuses, on a contact set with a rolling contact, every selection that keeps a contact
point fixed on its link or places a contact on a tilted patch, with one `InvalidArgument` that lists the rolling
contacts and the selections. They are:

<!-- LINT.IfChange(rolling_refused_terms) -->
- the soft constraints `contact_complementarity`, `force_weighted_slip` and `ground_penetration` of the
  contact-implicit formulation, which read the corners of a footprint and the slip of a point fixed on its link;
- `foot_collision` and `foot_collision_cbf`, the collision spheres of the biped's legs;
- `contact_swing` (and the swing landing alignment it builds) and `terrain_non_penetration`, the terms of a tracked
  multi-contact plan;
- `contact_moment_xy`, the center of pressure of a 6-DoF sole;
- `contact_input_parameterization: "basis_vectors"`, whose generator sets span a 6-DoF contact's wrench cone;
- `contact_schedule_source: "contact_planner"` and `"multi_contact_plan"`, which place each contact on a patch of its
  own;
- `leg_joint_reference: "foot_motion"`, which moves a point fixed on its link.
<!-- LINT.ThenChange(//legged_nmpc/legged_centroidal_mpc/src/CentroidalContactSet.cpp:rolling_refused_terms) -->

The terms the centroidal MPC refuses on every set other than the biped's (`external_torque_cost`, `dcm_terminal_cost`,
`knee_joint_mimic` and the base-pose and wrench heuristics) are refused on a rolling set as well. The whole-body MPC
runs on the biped's two 6-DoF feet only and refuses a rolling set with every other non-biped set.

## 10. Event-driven steps

A wheel's contact point may move only along the wheel's heading $\mathbf{t}$ (section 2). The commanded base twist
$(\mathbf{v}, \omega)$ moves the nominal place of contact $i$, which stands at $\mathbf{d}_i$ from the mean of the
contacts' nominal places in the heading frame (section 5), at $\mathbf{v} + \omega\,\mathbf{e}_z \times \mathbf{d}_i$.
The turn is about that mean rather than the base: Tron1's axle passes a few centimeters off its base, and about the base
a turn in place would move both wheels sideways, while about the axle they only roll and the base circles the axle by
the offset. The part of that velocity
along the axle, $\mathbf{a}_i = \mathbf{e}_z \times \mathbf{t}_i$, is what no rolling covers:

$$\dot{e}_i = \left(\mathbf{v} + \omega\,\mathbf{e}_z \times \mathbf{d}_i\right) \cdot \mathbf{a}_i.$$

$e_i$ is the lateral debt of the wheel, the distance its nominal place has moved along its axle since the wheel last
landed (`RollingStepScheduler::lateralDebtRate()`). The gait file's `gait_scheduler: "rolling_lateral_debt"` has
`RollingStepScheduler` (`legged_common_mpc/gait/`) write the mode schedule from it, in place of the gait switching table
(`rolling_step_scheduler_config.proto`):

- A step group (a wheel of Tron1, a diagonal pair of the wheeled ANYmal C) swings for `swing_duration` where the debt
  of one of its contacts reaches `max_lateral_debt`. It lands at its nominal foothold (section 5), so its debt is zero
  again.
- The groups swing one at a time, with `min_stance_duration` on every wheel between two swings. Of two groups due
  together, the one whose nominal places stand furthest ahead in the direction they move swings first: Tron1 steps
  with the wheel on the side it moves to, then with the other.
- The debts are integrated from the filtered command since each touchdown, open loop. Each solve predicts them over
  the horizon with its command and plans every swing the horizon holds, so the solver sees a swing a horizon ahead and
  moves the center of mass over the stance wheels in time. No swing is planned sooner than `min_lift_off_lead` after the
  solve: the schedule a solve writes is the next solve's (ProceduralMpcMotionManager). A swing that lifts off within
  `commit_window` of a solve is committed and keeps its times whatever the command does; the others are planned again
  at every solve.
- A command faster than the steps can follow delays each swing to the end of the stance after the last one. The steps
  then follow each other back to back, and each one clears a debt above the bound: the step length grows with the
  speed until the leg's reach limits the tracking. With $G$ groups, a group then stays down for
  $(G - 1)\,T_\mathrm{swing} + G\,T_\mathrm{stance}$ between its steps, and builds a debt of $|\dot{e}_i|$ times that.
  The wheeled ANYmal C's gait file keeps `min_stance_duration` at 0.05 s for this reason: at its 1 rad/s command limit
  ($|\dot{e}_i| = 0.36$ m/s) a pair stays down for 0.4 s, a debt of 14 cm. On turning_trot's 0.15 s stance it stays
  down for 0.6 s, a debt of 22 cm, and the base sank until the robot fell (legged_mpc_validation's `turn_1radps`). The
  debts follow from the timing; which limit the longer step runs into was not measured.

The rates of the shipped robots, from their geometry:

| Command | Tron1, $\mathbf{d}_i = (0, \pm y)$ | Wheeled ANYmal C, $\mathbf{d}_i = (\pm 0.36, \pm y)$ m |
| --- | --- | --- |
| $v_x$ | 0, rolls | 0, rolls |
| $v_y$ | $v_y$ on both wheels, steps | $v_y$ on every wheel, steps |
| $\omega$ | 0, turns by differential wheel speed | $\pm 0.36\,\omega$, front against hind, steps |

So a straight drive, and a turn of Tron1 alone or with a drive, never steps, and every other command combines rolling
with steps: the wheels roll the part of each contact's motion along its heading, and the steps take the rest. The
wheeled ANYmal C's front and hind wheels build their debts in opposite directions in a turn, and each diagonal pair has
one of each, so the pairs swing in turn.

What it does not do: the debt is the commanded one, so a push that moves the base sideways builds none, and a wheel
that the base drags out of its nominal place by a tracking error steps only on the command's schedule. A wheel lands
at its nominal place, so the debt between two steps lags behind by up to the bound; a landing that leads by the debt
the next stance will build would center it. `gait_scheduler: "gait_switching_table"` switches back to the gait table,
whose gaits keep every wheel on the ground, except the wheeled ANYmal C's `turning_trot`, which trots through every
yaw-rate command above 0.15 rad/s.

## 11. Runtime

The robot process applies the plan through inverse dynamics at the contact frames, so a rolling contact's force has to
be moved from the wheel's lowest point to the wheel center first. `CentroidalMpcMrtJointController` sets the moment of
each rolling contact's wrench to

$$\boldsymbol{\tau}_w = (-r\,\mathbf{n}) \times \mathbf{f}$$

(`torqueWheelEwrtWorld()`) before the inverse dynamics projects it with the LOCAL_WORLD_ALIGNED Jacobian of the
wheel-center frame. The component of that moment about the axle $\mathbf{a}$ is $-r\,\mathbf{f} \cdot \mathbf{t}$, so
the wheel joint's feedforward torque is $r$ times the traction along the rolling direction, the torque that drives the
wheel against the ground. The radii are read once at construction; a contact with radius 0 keeps its wrench, and the
control cycle allocates nothing.

The other runtime consumers of the contact set treat a rolling contact as follows:

- State estimation: `stateEstimatorContacts()` gives a rolling contact as an `estimation::ContactPointKind::kRolling`
  point on its wheel link at the wheel center, with its radius. The `kalman` estimator rejects it with
  InvalidArgument identifying the contact, because its contact measurement is a point at rest on its link;
  `ground_truth`, the simulator default, takes it.
- MuJoCo: the contact frame exists only in the MPC's model and is attached to the wheel joint, so `resolveContactBodies()`
  finds the wheel body through that joint, and the ground-truth contact mask is set while the wheel's collision
  cylinder carries a normal force, whichever point of the rim touches.
- Visualization: the scene draws a rolling contact's force arrow, its planned path and its foothold at the wheel's
  lowest point, and the telemetry computes the ZMP from the lowest points. The vertical velocity and acceleration
  plots of a contact are those of its frame, which a rolling contact's lowest point shares on flat ground.

## 12. Code and tests

| What | Where |
| --- | --- |
| Contact kind, wheel radius, task-file entry | `legged_common_mpc/common/ModelSettings.h`, `model_settings_config.proto` (`contacts_rolling`) |
| Geometry helpers | `legged_common_mpc/contact/RollingContact.h`, `testRollingContact` |
| Stance constraint | `legged_centroidal_mpc/constraint/RollingContactConstraintCppAd.h`, `testRollingContactConstraint` |
| Swing reference offset | `legged_common_mpc/LeggedPreComputation.cpp`, `testRollingContactSwing` |
| Stance references, wheel velocity reference | `legged_common_mpc/reference_manager/SwitchedModelReferenceManager.cpp`, `testRollingContactReferences` |
| Event-driven steps | `legged_common_mpc/gait/RollingStepScheduler.h`, `ProceduralMpcMotionManager.cpp`, `testRollingStepScheduler`, `testRollingStepSchedulerConfig`, `testTron1LockstepLoop`, `testAnymalCWheelsLockstepLoop` |
| Lever arm | `lib/ocs2/pinocchio/centroidal_model` (`rollingContactWheelRadii`), `testRollingContactLeverArm`, `testRollingContactSet` |
| Refused terms | `legged_centroidal_mpc/src/CentroidalContactSet.cpp`, `testRollingContactSet` |
| Wheel joints | `robot_model/UrdfContinuousJoints.h`, `StateInputWeightsFromConfig.cpp`, `JointLimitsSoftConstraint.cpp`, `testRollingContactModel` |
| MRT feedforward | `legged_centroidal_mpc/src/mrt/CentroidalMpcMrtJointController.cpp`, `test_rolling_contact_mrt_feedforward`, `test_centroidal_mpc_mrt_joint_controller_allocations` |
| Estimator contacts and refusal | `legged_state_estimator/ContactKinematics.h`, `RobotSetUpSteps.cpp`, `test_state_estimator_contacts`, `test_state_estimator_registry`, `test_lockstep_state_estimator` |
| MuJoCo contact bodies and mask | `mujoco_sim_interface/src/MujocoContactUtils.cpp`, `test_mujoco_contact_utils` |
| Scene and telemetry | `visualization/src/SceneBuilder.cpp`, `TelemetryBuilder.cpp`, `test_scene_builder`, `test_telemetry_builder` |

Every branch for a rolling contact is taken only when the contact set has one, so the LQ approximations of the robots
without wheels are bit-identical with the code compiled in (`testCentroidalContactSetParity`,
`testMrtJointControllerActionParity`, `//lib/ocs2:test_sqp_flat_parity`).
